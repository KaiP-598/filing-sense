"""GRPO training on FinQA — reinforcement learning from verifiable rewards.

Unlike SFT (which imitates gold answers), GRPO lets the model TRY to answer
questions, then reinforces correct attempts and penalizes wrong ones.
This is the DeepSeek-R1 algorithm adapted for financial reasoning.

How it works (one GRPO step):
    1. Sample N questions from FinQA training set
    2. For each question, generate G=8 candidate answers (via vLLM on GPU 1)
    3. Check each answer against ground truth (binary reward: 0 or 1)
    4. Compute group-normalized advantages:
       - Within each group of 8, correct answers get POSITIVE advantage
       - Wrong answers get NEGATIVE advantage
       - Model learns both "what works" and "what doesn't"
    5. Policy gradient update: increase probability of correct responses,
       decrease probability of wrong ones

Why GRPO after SFT?
    SFT teaches the FORMAT (Reasoning: ... Answer: ...) and basic skills.
    GRPO teaches the model to actually GET ANSWERS RIGHT by trial and error.
    The SFT model is the starting point — GRPO fine-tunes it further.

Requires 2 GPUs:
    - cuda:0: training model (forward/backward pass)
    - cuda:1: vLLM inference engine (generates candidate responses)

Usage (on 2-GPU machine):
    python scripts/train_grpo.py
    python scripts/train_grpo.py --model_path outputs/full-sft-qwen2.5-3b --n_steps 100

Interview notes:
    - GRPO vs PPO: GRPO doesn't need a critic/value network. Instead it normalizes
      rewards within each group of G responses. Simpler, fewer params, works great.
    - Why G=8: need variance within groups. If all 8 are wrong (or all right),
      advantage = 0 and no learning signal. G=8 is sweet spot for 16.5% accuracy model.
    - clip_eps=0.2: prevents the policy from changing too drastically in one step
      (same as PPO clipping). Without it, training can collapse.
    - Temperature=1.0 for generation: we WANT diversity in the 8 candidates.
      Greedy (temp=0) would give 8 identical responses = no learning signal.
    - Starting from SFT checkpoint: the model already knows the format, so GRPO
      can focus on improving REASONING rather than learning format from scratch.
"""

from __future__ import annotations

import argparse
import json
import random
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from unittest.mock import patch
from vllm import LLM, SamplingParams
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Reward function — adapted from eval/evaluate_rag.py
# ---------------------------------------------------------------------------

def extract_answer(model_output: str) -> str | None:
    """Extract the final answer from model output.

    Same logic as eval — tries 'Answer: X', then last number, then yes/no.
    """
    if not model_output:
        return None

    match = re.search(r'[Aa]nswer\s*:\s*([-\d.,]+%?)', model_output)
    if match:
        return match.group(1)

    numbers = re.findall(r'-?\d+\.?\d*', model_output)
    if numbers:
        return numbers[-1]

    lower = model_output.lower()
    if 'yes' in lower:
        return 'yes'
    if 'no' in lower:
        return 'no'

    return None


def normalize_number(s: str) -> float | None:
    """Parse string to float, handling %, commas."""
    try:
        return float(s.replace('%', '').replace(',', '').strip())
    except (ValueError, AttributeError):
        return None


def is_correct(predicted: str, ground_truth: str, tolerance: float = 0.01) -> bool:
    """Check if predicted matches ground truth within 1% tolerance."""
    if not predicted:
        return False

    pred_lower = predicted.lower().strip()
    gold_lower = ground_truth.lower().strip()

    if gold_lower in ('yes', 'no'):
        return pred_lower == gold_lower or gold_lower in pred_lower

    pred_num = normalize_number(predicted)
    gold_num = normalize_number(ground_truth)

    if pred_num is None or gold_num is None:
        return False

    # Handle percentage mismatch
    if abs(pred_num) > 1 and abs(gold_num) < 1 and gold_num != 0:
        pred_num = pred_num / 100
    if abs(pred_num) < 1 and abs(gold_num) > 1 and pred_num != 0:
        pred_num = pred_num * 100

    if gold_num == 0:
        return abs(pred_num) < 0.01
    return abs(pred_num - gold_num) / abs(gold_num) < tolerance


def finqa_reward(response: str, ground_truth: str) -> dict:
    """Binary reward: 1 if answer is correct, 0 otherwise.

    This is what GRPO optimizes. The model gets reward=1 for correct answers
    and reward=0 for wrong ones. Group normalization turns these into
    positive/negative advantages for the policy gradient update.
    """
    predicted = extract_answer(response)
    correct = is_correct(predicted, str(ground_truth))
    return {"reward": 1.0 if correct else 0.0}


# ---------------------------------------------------------------------------
# Group-normalized advantages (core GRPO insight)
# ---------------------------------------------------------------------------

def compute_advantages(
    responses: list[str],
    ground_truths: list[str],
    group_size: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Score all responses and compute group-normalized advantages.

    For each question, we generated G responses. Within each group:
        advantage = (reward - group_mean) / (group_std + eps)

    Example with G=8, suppose 2 of 8 are correct:
        rewards = [0, 1, 0, 0, 1, 0, 0, 0]
        mean = 0.25, std = 0.46
        correct answers get advantage = (1 - 0.25) / 0.46 = +1.63
        wrong answers get advantage = (0 - 0.25) / 0.46 = -0.54

    The model learns: "increase probability of those 2 correct responses,
    decrease probability of the 6 wrong ones."
    """
    n_total = len(responses)
    assert n_total % group_size == 0

    scores = []
    for resp, gt in zip(responses, ground_truths):
        r = finqa_reward(resp, gt)
        scores.append(r["reward"])

    raw_rewards = torch.tensor(scores, dtype=torch.float32)

    n_prompts = n_total // group_size
    grouped = raw_rewards.view(n_prompts, group_size)

    group_mean = grouped.mean(dim=1, keepdim=True)
    group_std = grouped.std(dim=1, keepdim=True)
    advantages = (grouped - group_mean) / (group_std + eps)

    stats = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward": raw_rewards.std().item(),
        "fraction_correct": (raw_rewards > 0).float().mean().item(),
    }

    return advantages.view(n_total), raw_rewards, stats


# ---------------------------------------------------------------------------
# Tokenization and log-prob utilities
# ---------------------------------------------------------------------------

def tokenize_pairs(prompts, responses, tokenizer):
    """Tokenize (prompt, response) pairs and build response mask.

    The response_mask marks which tokens are from the response (not prompt).
    We only compute loss on response tokens — the prompt is context, not
    something the model should be rewarded/penalized for.
    """
    all_ids, all_labels, all_masks = [], [], []
    max_len = 0

    for prompt, response in zip(prompts, responses):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        full_ids = prompt_ids + response_ids

        # Labels: same as input but shifted (for next-token prediction)
        labels = [-100] * len(prompt_ids) + response_ids
        # Response mask: 1 for response tokens, 0 for prompt
        mask = [0] * len(prompt_ids) + [1] * len(response_ids)

        all_ids.append(full_ids)
        all_labels.append(labels)
        all_masks.append(mask)
        max_len = max(max_len, len(full_ids))

    # Pad to uniform length
    pad_id = tokenizer.pad_token_id or 0
    for i in range(len(all_ids)):
        pad_len = max_len - len(all_ids[i])
        all_ids[i] = all_ids[i] + [pad_id] * pad_len
        all_labels[i] = all_labels[i] + [-100] * pad_len
        all_masks[i] = all_masks[i] + [0] * pad_len

    return {
        "input_ids": torch.tensor(all_ids, dtype=torch.long),
        "labels": torch.tensor(all_labels, dtype=torch.long),
        "response_mask": torch.tensor(all_masks, dtype=torch.float32),
    }


def get_log_probs(model, input_ids, labels):
    """Forward pass → extract per-token log probabilities.

    Returns log P(token_t | tokens_0..t-1) for each position.
    These are used to compute the policy gradient loss.
    """
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, :-1, :]  # shift: predict next token
    log_probs = torch.log_softmax(logits, dim=-1)

    # Gather log-prob of the actual token at each position
    target = labels[:, 1:]  # shifted labels
    target = target.clamp(min=0)  # replace -100 with 0 for gathering
    token_log_probs = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)

    # Mask out padding/prompt positions
    valid_mask = (labels[:, 1:] != -100).float()
    token_log_probs = token_log_probs * valid_mask

    return {"log_probs": token_log_probs}


# ---------------------------------------------------------------------------
# GRPO policy gradient step
# ---------------------------------------------------------------------------

def grpo_loss(log_probs, old_log_probs, response_mask, advantages, clip_eps=0.2):
    """Compute clipped policy gradient loss (PPO-style).

    ratio = exp(log_new - log_old)  → how much the policy changed
    loss = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)

    The clipping prevents the policy from changing too drastically.
    Without it, one lucky batch could push the model off a cliff.

    Think of it like iOS rate limiting: you allow gradual updates but
    reject any single update that's too large.
    """
    # Per-token importance ratio
    ratio = torch.exp(log_probs - old_log_probs)

    # Broadcast advantages from per-sequence to per-token
    seq_len = log_probs.shape[1]
    token_advantages = advantages.unsqueeze(1).expand(-1, seq_len)

    # Clipped surrogate objective
    surr1 = ratio * token_advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * token_advantages
    per_token_loss = -torch.min(surr1, surr2)

    # Average over response tokens only
    masked_loss = (per_token_loss * response_mask[:, 1:]).sum() / response_mask[:, 1:].sum().clamp(min=1)
    return masked_loss


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a financial analyst. Answer the question based ONLY on the provided context from SEC 10-K filings.

Rules:
- If the answer is a number, compute it step by step and give the final numerical answer.
- If the answer requires a percentage, show your calculation.
- If the context doesn't contain enough information, say "Insufficient context."
- Always show your reasoning before the final answer.

Format your response as:
Reasoning: <your step-by-step calculation>
Answer: <final answer>"""


def prepare_grpo_data(split: str = "train") -> list[dict]:
    """Load FinQA and prepare for GRPO.

    Each example has:
        - prompt: the full chat-formatted prompt (system + context + question)
        - ground_truth: the exact answer (number or yes/no)

    We use GOLD context (not retrieved) for training, same as SFT.
    The model learns REASONING skills, not how to handle bad retrieval.
    """
    from scripts.prepare_sft_data import _safe_join, _table_to_text

    ds = load_dataset("wandb/finqa-data-processed", split=split)
    examples = []

    for ex in ds:
        pre = _safe_join(ex.get("pre_text", ""))
        table = _table_to_text(ex.get("table", ""))
        post = _safe_join(ex.get("post_text", ""))
        context = f"{pre}\n\n{table}\n\n{post}".strip()
        query = ex.get("query", "")
        answer = ex.get("exe_ans", "")

        user_msg = f"{context}\n\n---\n\nQuestion: {query}"

        examples.append({
            "query": query,
            "context": user_msg,
            "ground_truth": str(answer),
        })

    return examples


def build_chat_prompt(context: str, tokenizer) -> str:
    """Build a chat-formatted prompt for vLLM generation.

    Uses the tokenizer's chat template so the model sees the same format
    it learned during SFT training.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# vLLM helpers (same two-GPU pattern as grpo-from-scratch)
# ---------------------------------------------------------------------------

def create_inference_engine(model_path: str, device: str, seed: int, mem_util: float = 0.6):
    """Initialize vLLM on a dedicated GPU for fast batched generation.

    Why a separate GPU? Training needs GPU memory for gradients and optimizer
    states. vLLM needs memory for KV cache. Putting both on one GPU = OOM.
    """
    from vllm.model_executor import set_random_seed as vllm_seed
    vllm_seed(seed)

    ws_patch = patch("torch.distributed.get_world_size", return_value=1)

    # The profiling assertion patch is version-specific — skip if not found
    try:
        from vllm.worker.worker import Worker
        has_profiling_assert = hasattr(Worker, "_assert_memory_footprint_increased_during_profiling")
    except ImportError:
        has_profiling_assert = False

    if has_profiling_assert:
        prof_patch = patch(
            "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
            return_value=None,
        )
    else:
        from contextlib import nullcontext
        prof_patch = nullcontext()

    with ws_patch, prof_patch:
        return LLM(
            model=model_path,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=mem_util,
            enforce_eager=True,
        )


def sync_weights(policy, llm):
    """Copy training model weights into vLLM's engine.

    After each GRPO step, the training model has new weights.
    We sync them to vLLM so the next generation uses the updated policy.
    """
    state = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state.items())


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GRPO training on FinQA")
    parser.add_argument("--model_path", type=str, default="outputs/full-sft-qwen2.5-3b",
                        help="Starting checkpoint (should be SFT-trained)")
    parser.add_argument("--output_dir", type=str, default="outputs/grpo-qwen2.5-3b")

    # GRPO hyperparameters
    parser.add_argument("--n_steps", type=int, default=100,
                        help="Number of GRPO rollout-train cycles")
    parser.add_argument("--group_size", type=int, default=8,
                        help="G: candidate responses per question")
    parser.add_argument("--n_prompts", type=int, default=16,
                        help="Questions per step (total rollouts = n_prompts * group_size)")
    parser.add_argument("--epochs_per_batch", type=int, default=1,
                        help="K: how many times to reuse each rollout batch")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--clip_eps", type=float, default=0.2,
                        help="PPO clipping epsilon")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Max tokens per generated response")
    parser.add_argument("--eval_every", type=int, default=10,
                        help="Evaluate every N steps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda:0"

    rollout_batch = args.n_prompts * args.group_size  # total rollouts per step

    print("=" * 60)
    print("GRPO Training on FinQA")
    print(f"  Model: {args.model_path}")
    print(f"  Steps: {args.n_steps}")
    print(f"  Group size (G): {args.group_size}")
    print(f"  Prompts per step: {args.n_prompts}")
    print(f"  Total rollouts per step: {rollout_batch}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Clip epsilon: {args.clip_eps}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load model (starting from SFT checkpoint)
    # ------------------------------------------------------------------
    print(f"\nLoading model: {args.model_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device)
        print("  Using flash_attention_2")
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device)
        print("  Using SDPA fallback")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")

    # ------------------------------------------------------------------
    # 2. vLLM inference engine on GPU 1
    # ------------------------------------------------------------------
    print("\nStarting vLLM inference engine on cuda:1...")
    engine = create_inference_engine(args.model_path, "cuda:1", args.seed)

    gen_params = SamplingParams(
        temperature=1.0,      # diversity is critical — we need varied candidates
        max_tokens=args.max_tokens,
        n=args.group_size,    # generate G responses per prompt in one call
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # 3. Load data + optimizer
    # ------------------------------------------------------------------
    print("\nLoading FinQA training data...")
    train_data = prepare_grpo_data("train")
    val_data = prepare_grpo_data("test")
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95),
    )

    micro_batch = max(1, rollout_batch // args.grad_accum)

    # ------------------------------------------------------------------
    # 4. GRPO loop
    # ------------------------------------------------------------------
    best_accuracy = 0.0

    for step in range(args.n_steps):
        print(f"\n{'=' * 60}")
        print(f"Step {step + 1}/{args.n_steps}")
        print(f"{'=' * 60}")

        # 4a. Sample questions
        questions = random.sample(train_data, min(args.n_prompts, len(train_data)))
        prompts = [build_chat_prompt(q["context"], tokenizer) for q in questions]
        truths = [q["ground_truth"] for q in questions]

        # 4b. Generate G responses per question via vLLM
        model.eval()
        sync_weights(model, engine)
        outputs = engine.generate(prompts, gen_params)

        # Flatten into (prompt, response, ground_truth) triples
        flat_prompts, flat_responses, flat_truths = [], [], []
        for prompt, output, gt in zip(prompts, outputs, truths):
            for completion in output.outputs:
                flat_prompts.append(prompt)
                flat_responses.append(completion.text)
                flat_truths.append(gt)

        # 4c. Compute group-normalized advantages
        advantages, raw_rewards, reward_stats = compute_advantages(
            responses=flat_responses,
            ground_truths=flat_truths,
            group_size=args.group_size,
        )
        print(f"  Reward: mean={reward_stats['mean_reward']:.4f}, "
              f"correct={reward_stats['fraction_correct']:.1%}")

        # Skip step if no learning signal (all correct or all wrong in every group)
        if advantages.abs().max() < 1e-8:
            print("  No learning signal (all groups uniform) — skipping step")
            continue

        # 4d. Tokenize all rollouts
        batch = tokenize_pairs(flat_prompts, flat_responses, tokenizer)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        response_mask = batch["response_mask"].to(device)
        advantages = advantages.to(device)

        # 4e. Snapshot reference log-probs (frozen, computed once)
        # These are the "old" probabilities used for importance ratio in PPO clipping
        ref_log_probs_chunks = []
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, input_ids.shape[0], micro_batch):
                j = min(i + micro_batch, input_ids.shape[0])
                result = get_log_probs(model, input_ids[i:j], labels[i:j])
                ref_log_probs_chunks.append(result["log_probs"])
            ref_log_probs = torch.cat(ref_log_probs_chunks, dim=0).detach()
            del ref_log_probs_chunks
            torch.cuda.empty_cache()

        # 4f. Train for K epochs on this rollout batch
        model.train()
        n_rollouts = len(flat_responses)

        for epoch in range(args.epochs_per_batch):
            indices = list(range(n_rollouts))
            random.shuffle(indices)
            mb_count = 0

            for mb_start in range(0, n_rollouts, micro_batch):
                mb_idx = indices[mb_start:mb_start + micro_batch]
                if len(mb_idx) == 0:
                    continue

                mb_ids = input_ids[mb_idx]
                mb_labels = labels[mb_idx]
                mb_mask = response_mask[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ref = ref_log_probs[mb_idx]

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    result = get_log_probs(model, mb_ids, mb_labels)

                loss = grpo_loss(
                    log_probs=result["log_probs"],
                    old_log_probs=mb_ref,
                    response_mask=mb_mask,
                    advantages=mb_adv,
                    clip_eps=args.clip_eps,
                )

                (loss / args.grad_accum).backward()
                mb_count += 1

                if mb_count % args.grad_accum == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad()

                    print(f"    loss={loss.item():.4f}, grad_norm={grad_norm.item():.4f}")

        # Clean up
        del input_ids, labels, response_mask, advantages, ref_log_probs
        torch.cuda.empty_cache()

        # 4g. Periodic evaluation
        if (step + 1) % args.eval_every == 0:
            print(f"\n  Evaluating on {min(100, len(val_data))} val examples...")
            model.eval()
            sync_weights(model, engine)

            eval_examples = random.sample(val_data, min(100, len(val_data)))
            eval_prompts = [build_chat_prompt(ex["context"], tokenizer) for ex in eval_examples]

            eval_params = SamplingParams(
                temperature=0.0,  # greedy for eval — deterministic
                max_tokens=args.max_tokens,
            )
            eval_outputs = engine.generate(eval_prompts, eval_params)

            correct = 0
            for ex, output in zip(eval_examples, eval_outputs):
                response = output.outputs[0].text
                predicted = extract_answer(response)
                if is_correct(predicted, ex["ground_truth"]):
                    correct += 1

            accuracy = correct / len(eval_examples)
            print(f"  Val accuracy: {accuracy:.1%} ({correct}/{len(eval_examples)})")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                print(f"  New best! Saving checkpoint...")
                os.makedirs(args.output_dir, exist_ok=True)
                model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)

    # ------------------------------------------------------------------
    # 5. Final save
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training config
    config = {
        "model_path": args.model_path,
        "method": "grpo",
        "n_steps": args.n_steps,
        "group_size": args.group_size,
        "n_prompts": args.n_prompts,
        "epochs_per_batch": args.epochs_per_batch,
        "lr": args.lr,
        "clip_eps": args.clip_eps,
        "best_val_accuracy": best_accuracy,
        "total_params": total_params,
    }
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"DONE! Best val accuracy: {best_accuracy:.1%}")
    print(f"Model saved to {args.output_dir}")
    print(f"Next: python -m eval.evaluate_rag --model_path {args.output_dir} --num_examples 200 --use_reranker --output_dir results")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
