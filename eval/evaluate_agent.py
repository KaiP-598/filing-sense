"""Evaluate the LangGraph agentic RAG pipeline on FinQA.

Compares agent accuracy against the baseline single-pass RAG pipeline.

Usage:
    python -m eval.evaluate_agent \
        --model_path kaiwu598/filing-sense-grpo-qwen2.5-3b \
        --openai_api_key sk-... \
        --num_examples 200
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from datasets import load_dataset

from src.chunking import chunk_finqa_example
from src.indexing import build_index
from src.retrieval import load_reranker
from src.agent import build_agent
from eval.evaluate_rag import extract_answer, is_correct


def run_agent_eval(
    agent,
    test_examples: list,
    max_examples: int | None = None,
) -> dict:
    """Run the agent on test examples and measure accuracy.

    Logs per-example results including which node the agent succeeded/failed at,
    number of retries used, and whether the answer was correct.
    """
    examples = test_examples[:max_examples] if max_examples else test_examples
    correct = 0
    results_log = []

    for i, ex in enumerate(examples):
        query = ex["query"]
        gold = str(ex["exe_ans"])

        try:
            start = time.time()
            state = agent.invoke({"query": query})
            elapsed = time.time() - start

            raw_answer = state.get("answer", "")
            predicted = extract_answer(raw_answer)
            correct_flag = is_correct(predicted, gold)

            if correct_flag:
                correct += 1

            gen = state.get("generation_result")
            results_log.append({
                "idx": i,
                "query": query,
                "gold": gold,
                "predicted": predicted,
                "correct": correct_flag,
                "retries": state.get("retries", 0),
                "sub_queries": state.get("sub_queries", []),
                "relevant_chunks_found": len(state.get("relevant_chunks", [])),
                "total_chunks_retrieved": len(state.get("retrieved_chunks", [])),
                "model_output": raw_answer,
                "elapsed_sec": round(elapsed, 2),
                "num_chunks_used": gen.num_chunks_used if gen else 0,
            })

        except Exception as e:
            results_log.append({
                "idx": i,
                "query": query,
                "gold": gold,
                "predicted": None,
                "correct": False,
                "error": str(e),
            })

        if (i + 1) % 25 == 0:
            running_acc = correct / (i + 1) * 100
            retried = sum(1 for r in results_log if r.get("retries", 0) > 0)
            print(f"[{i+1}/{len(examples)}] accuracy={running_acc:.1f}%  retried={retried}")

    accuracy = correct / len(examples) * 100
    retried_count = sum(1 for r in results_log if r.get("retries", 0) > 0)
    avg_relevant = sum(r.get("relevant_chunks_found", 0) for r in results_log) / len(results_log)

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(examples),
        "retried_queries": retried_count,
        "avg_relevant_chunks": round(avg_relevant, 2),
        "results_log": results_log,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="kaiwu598/filing-sense-grpo-qwen2.5-3b")
    parser.add_argument("--openai_api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--num_examples", type=int, default=200)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--grading_model", default="gpt-4o-mini")
    args = parser.parse_args()

    print("Loading FinQA dataset...")
    ds = load_dataset("wandb/finqa-data-processed")
    train_examples = list(ds["train"])
    test_examples = list(ds["test"])

    print(f"Chunking {len(train_examples)} training examples...")
    chunks = [chunk_finqa_example(ex, i) for i, ex in enumerate(train_examples)]

    print("Building dual index (BM25 + FAISS)...")
    index = build_index(chunks)

    print("Loading reranker...")
    reranker = load_reranker()

    print(f"Loading model: {args.model_path}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    print("Building LangGraph agent...")
    agent = build_agent(
        index=index,
        reranker=reranker,
        model=model,
        tokenizer=tokenizer,
        openai_api_key=args.openai_api_key,
        model_name=args.model_path.split("/")[-1],
        grading_model=args.grading_model,
    )

    print(f"\nRunning agent eval on {args.num_examples} examples...\n")
    results = run_agent_eval(agent, test_examples, max_examples=args.num_examples)

    print(f"\n{'='*50}")
    print(f"Agent accuracy:      {results['accuracy']:.1f}%")
    print(f"Correct:             {results['correct']}/{results['total']}")
    print(f"Queries retried:     {results['retried_queries']}")
    print(f"Avg relevant chunks: {results['avg_relevant_chunks']}")
    print(f"{'='*50}\n")

    Path(args.output_dir).mkdir(exist_ok=True)

    summary_path = Path(args.output_dir) / "eval_results_agent.json"
    with open(summary_path, "w") as f:
        json.dump({k: v for k, v in results.items() if k != "results_log"}, f, indent=2)

    detail_path = Path(args.output_dir) / "eval_detailed_agent.jsonl"
    with open(detail_path, "w") as f:
        for r in results["results_log"]:
            f.write(json.dumps(r) + "\n")

    print(f"Results saved to {summary_path}")
    print(f"Detailed log saved to {detail_path}")


if __name__ == "__main__":
    main()
