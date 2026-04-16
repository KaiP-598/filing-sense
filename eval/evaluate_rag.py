"""Evaluate the RAG pipeline on FinQA test set.

Runs three evaluations:
    1. Retrieval comparison: BM25-only vs FAISS-only vs hybrid
    2. End-to-end accuracy: retrieve + generate + check answer
    3. Prints a results summary

Usage (on GPU machine):
    python -m eval.evaluate_rag --model_path Qwen/Qwen2.5-3B --num_examples 100
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from datasets import load_dataset

from src.chunking import chunk_dataset
from src.indexing import build_index, DualIndex, _tokenize_simple
from src.retrieval import retrieve, _search_bm25, _search_faiss, _reciprocal_rank_fusion, load_reranker


# ---------------------------------------------------------------------------
# Answer extraction and comparison
# ---------------------------------------------------------------------------

def extract_answer(model_output: str) -> str | None:
    """Extract the final answer from the model's free-text response.

    Tries in order:
        1. Look for "Answer: <value>" pattern
        2. Fallback: grab the last number in the output
        3. Check for yes/no
    """
    if not model_output:
        return None

    # Strategy 1: explicit "Answer:" tag
    match = re.search(r'[Aa]nswer\s*:\s*([-\d.,]+%?)', model_output)
    if match:
        return match.group(1)

    # Strategy 2: last number in output (often the final calculated result)
    numbers = re.findall(r'-?\d+\.?\d*', model_output)
    if numbers:
        return numbers[-1]

    # Strategy 3: yes/no for comparison questions
    lower = model_output.lower()
    if 'yes' in lower:
        return 'yes'
    if 'no' in lower:
        return 'no'

    return None


def normalize_number(s: str) -> float | None:
    """Parse a string into a float, handling %, commas, etc."""
    try:
        cleaned = s.replace('%', '').replace(',', '').strip()
        return float(cleaned)
    except (ValueError, AttributeError):
        return None


def is_correct(predicted: str, ground_truth: str, tolerance: float = 0.01) -> bool:
    """Check if predicted answer matches ground truth.

    For numbers: match within 1% relative tolerance.
    For yes/no: exact string match.
    """
    if not predicted:
        return False

    pred_lower = predicted.lower().strip()
    gold_lower = ground_truth.lower().strip()

    # Handle yes/no questions
    if gold_lower in ('yes', 'no'):
        return pred_lower == gold_lower or gold_lower in pred_lower

    # Parse both as numbers
    pred_num = normalize_number(predicted)
    gold_num = normalize_number(ground_truth)

    if pred_num is None or gold_num is None:
        return False

    # Handle percentage mismatch: model says 24.7, ground truth is 0.247
    if abs(pred_num) > 1 and abs(gold_num) < 1 and gold_num != 0:
        pred_num = pred_num / 100

    # Handle reverse: model says 0.247, ground truth is 24.7
    if abs(pred_num) < 1 and abs(gold_num) > 1 and pred_num != 0:
        pred_num = pred_num * 100

    # Compare with relative tolerance
    if gold_num == 0:
        return abs(pred_num) < 0.01
    return abs(pred_num - gold_num) / abs(gold_num) < tolerance


# ---------------------------------------------------------------------------
# Retrieval evaluation (no LLM needed)
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    index: DualIndex,
    test_examples: list[dict],
    mode: str = "hybrid",
    top_k: int = 5,
) -> dict:
    """Evaluate retrieval quality by checking if retrieved chunks contain
    the information needed to answer the question.

    Since test chunks aren't in the train index, we check if any retrieved
    chunk's text has significant token overlap with the ground-truth context.

    Args:
        index: the built DualIndex over train data
        test_examples: list of test examples with 'query' and 'context' fields
        mode: "bm25", "faiss", or "hybrid"
        top_k: number of chunks to retrieve
    """
    hits = 0
    reciprocal_ranks = []
    total = len(test_examples)

    for ex in test_examples:
        query = ex["query"]
        # The ground-truth context — the text that contains the answer
        gold_context = ex.get("context", "")
        gold_tokens = set(gold_context.lower().split())

        # Retrieve based on mode
        if mode == "bm25":
            raw_results = _search_bm25(index, query, top_k=top_k)
        elif mode == "faiss":
            raw_results = _search_faiss(index, query, top_k=top_k)
        else:  # hybrid
            bm25_results = _search_bm25(index, query, top_k=20)
            faiss_results = _search_faiss(index, query, top_k=20)
            fused = _reciprocal_rank_fusion([bm25_results, faiss_results])
            raw_results = fused[:top_k]

        # Check each retrieved chunk for overlap with ground-truth context
        found_rank = None
        for rank, (idx, _score) in enumerate(raw_results):
            chunk_tokens = set(index.chunks[idx].text.lower().split())
            # Overlap: what fraction of gold tokens appear in the chunk?
            if gold_tokens:
                overlap = len(gold_tokens & chunk_tokens) / len(gold_tokens)
            else:
                overlap = 0

            # Threshold: if >50% of ground-truth tokens found, it's a match
            if overlap > 0.5:
                found_rank = rank
                break

        if found_rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / (found_rank + 1))
        else:
            reciprocal_ranks.append(0.0)

    recall = hits / total if total > 0 else 0
    mrr = sum(reciprocal_ranks) / total if total > 0 else 0

    return {
        "mode": mode,
        "total": total,
        "hits": hits,
        "recall@k": recall,
        "mrr": mrr,
    }


# ---------------------------------------------------------------------------
# End-to-end evaluation (needs LLM)
# ---------------------------------------------------------------------------

def evaluate_end_to_end(
    index: DualIndex,
    test_examples: list[dict],
    model,
    tokenizer,
    model_name: str = "qwen2.5-3b",
    top_k: int = 5,
    reranker=None,
    max_examples: int | None = None,
) -> dict:
    """Full pipeline evaluation: retrieve → generate → check answer.

    Args:
        index: built DualIndex
        test_examples: test set with 'query' and 'exe_ans' fields
        model: loaded HuggingFace model
        tokenizer: model's tokenizer
        model_name: for logging
        top_k: number of chunks to retrieve
        reranker: optional cross-encoder reranker
        max_examples: limit eval to N examples (for quick testing)
    """
    from src.retrieval import retrieve as full_retrieve
    from src.generation import generate_answer

    examples = test_examples[:max_examples] if max_examples else test_examples
    total = len(examples)
    correct = 0
    results_log = []

    start_time = time.time()

    for i, ex in enumerate(examples):
        query = ex["query"]
        gold_answer = ex["exe_ans"]

        # Step 1: Retrieve
        retrieved = full_retrieve(
            index, query,
            top_n_final=top_k,
            reranker=reranker,
            use_reranker=reranker is not None,
        )

        # Step 2: Generate
        gen_result = generate_answer(
            query, retrieved,
            model=model, tokenizer=tokenizer,
            model_name=model_name,
        )

        # Step 3: Extract and compare
        predicted = extract_answer(gen_result.answer)
        is_match = is_correct(predicted, gold_answer)

        if is_match:
            correct += 1

        # Check if retrieval found the right context
        gold_context = ex.get("context", "")
        gold_tokens = set(gold_context.lower().split())
        retrieval_hit = False
        for chunk in retrieved:
            chunk_tokens = set(chunk.chunk.text.lower().split())
            if gold_tokens and len(gold_tokens & chunk_tokens) / len(gold_tokens) > 0.5:
                retrieval_hit = True
                break

        results_log.append({
            "idx": i,
            "query": query,
            "gold": gold_answer,
            "predicted": predicted,
            "correct": is_match,
            "retrieval_hit": retrieval_hit,
            "model_output": gen_result.answer[:500],
            "retrieved_texts": [c.chunk.text[:200] for c in retrieved[:3]],
        })

        # Progress update every 50 examples
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{total}] accuracy so far: {correct/(i+1):.1%} ({rate:.1f} examples/sec)")

    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0

    return {
        "model": model_name,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_seconds": elapsed,
        "results_log": results_log,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline on FinQA")
    parser.add_argument("--model_path", type=str, default=None,
                        help="HuggingFace model path (e.g. Qwen/Qwen2.5-3B). "
                             "If not provided, only retrieval eval runs.")
    parser.add_argument("--num_examples", type=int, default=None,
                        help="Limit eval to N test examples (default: all)")
    parser.add_argument("--embed_model", type=str, default="BAAI/bge-small-en-v1.5",
                        help="Embedding model for FAISS")
    parser.add_argument("--use_reranker", action="store_true",
                        help="Use cross-encoder reranker")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to LoRA adapter (e.g. outputs/lora-qwen2.5-3b). "
                             "If provided, merges adapter into base model for eval.")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results JSON")
    args = parser.parse_args()

    # --- Load data ---
    print("Loading FinQA dataset...")
    ds = load_dataset("wandb/finqa-data-processed")
    train_data = ds["train"]
    test_data = ds["test"]
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

    # --- Build index from train set ---
    print("\nChunking train data...")
    chunks = chunk_dataset(train_data)
    print(f"  {len(chunks)} chunks")

    print("\nBuilding index...")
    dual_index = build_index(chunks, embed_model_name=args.embed_model)

    # --- Prepare test examples ---
    test_examples = list(test_data)
    if args.num_examples:
        test_examples = test_examples[:args.num_examples]
    print(f"\nEvaluating on {len(test_examples)} test examples")

    # --- Retrieval evaluation (no LLM needed) ---
    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION")
    print("=" * 60)

    retrieval_results = {}
    for mode in ["bm25", "faiss", "hybrid"]:
        print(f"\n  Running {mode}...")
        result = evaluate_retrieval(dual_index, test_examples, mode=mode)
        retrieval_results[mode] = result
        print(f"  {mode}: recall@5={result['recall@k']:.1%}, MRR={result['mrr']:.3f}")

    # --- End-to-end evaluation (needs LLM) ---
    e2e_result = None
    if args.model_path:
        print("\n" + "=" * 60)
        print("END-TO-END EVALUATION")
        print("=" * 60)

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        if args.adapter_path:
            # LoRA eval: load base model + merge adapter
            print(f"\nLoading base model: {args.model_path}")
            print(f"  + LoRA adapter: {args.adapter_path}")
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(base_model, args.adapter_path)
            model = model.merge_and_unload()  # merge LoRA into base for faster inference
            print("  LoRA adapter merged into base model")
        else:
            # Standard eval: load model directly
            print(f"\nLoading model: {args.model_path}")
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )

        reranker = load_reranker() if args.use_reranker else None

        print(f"\nRunning end-to-end eval...")
        e2e_result = evaluate_end_to_end(
            dual_index, test_examples,
            model=model, tokenizer=tokenizer,
            model_name=args.model_path,
            reranker=reranker,
            max_examples=args.num_examples,
        )
        print(f"\n  Accuracy: {e2e_result['accuracy']:.1%} "
              f"({e2e_result['correct']}/{e2e_result['total']})")
        print(f"  Time: {e2e_result['elapsed_seconds']:.0f}s")

    # --- Save results ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "retrieval": retrieval_results,
        "end_to_end": {
            "model": e2e_result["model"] if e2e_result else None,
            "accuracy": e2e_result["accuracy"] if e2e_result else None,
            "total": e2e_result["total"] if e2e_result else None,
        } if e2e_result else None,
    }

    output_path = output_dir / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Save detailed per-example log for error taxonomy analysis
    if e2e_result and e2e_result.get("results_log"):
        log_path = output_dir / "eval_detailed_log.jsonl"
        with open(log_path, "w") as f:
            for entry in e2e_result["results_log"]:
                f.write(json.dumps(entry) + "\n")
        print(f"Detailed log saved to {log_path}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Mode':<12} {'Recall@5':>10} {'MRR':>10}")
    print("-" * 34)
    for mode, result in retrieval_results.items():
        print(f"{mode:<12} {result['recall@k']:>9.1%} {result['mrr']:>10.3f}")

    if e2e_result:
        print(f"\nEnd-to-end accuracy: {e2e_result['accuracy']:.1%}")


if __name__ == "__main__":
    main()
