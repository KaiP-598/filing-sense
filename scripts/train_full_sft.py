"""Full SFT training on FinQA — all parameters unfrozen.

Unlike LoRA (which trains 0.43% of parameters via low-rank adapters),
Full SFT trains ALL 1.7B parameters. This gives the model more capacity
to learn financial reasoning patterns, but requires more VRAM (~36 GB)
and risks overfitting on small datasets.

Usage (on GPU machine with 80GB+ VRAM):
    python scripts/train_full_sft.py
    python scripts/train_full_sft.py --epochs 3 --lr 2e-5

Interview notes:
    - Full SFT vs LoRA: Full SFT modifies every weight including FFN, embeddings,
      layer norms. LoRA only modifies attention projections via low-rank matrices.
    - LR is 10x lower than LoRA (2e-5 vs 2e-4) — with all params unfrozen,
      large LR causes catastrophic forgetting (model forgets pre-training knowledge)
    - Gradient checkpointing is critical — without it, OOM on 80GB GPU
    - bf16 mixed precision: forward pass in bf16, gradients accumulated in fp32
    - Output is a full model (not an adapter) — larger to store but no merging needed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


def load_sft_data(data_path: str) -> list[dict]:
    """Load chat-format JSONL data."""
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(description="Full SFT on FinQA")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--data_path", type=str, default="data/sft_train.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/full-sft-qwen2.5-3b")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-device batch size (small — full model is memory-hungry)")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="Gradient accumulation (effective batch = batch_size × grad_accum)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (10x lower than LoRA to avoid catastrophic forgetting)")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()

    effective_batch = args.batch_size * args.grad_accum
    print("=" * 60)
    print("Full SFT Training")
    print(f"  Model: {args.model_name}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Effective batch size: {effective_batch}")
    print(f"  Learning rate: {args.lr}")
    print(f"  NOTE: All parameters unfrozen — this is NOT LoRA")
    print("=" * 60)

    # --- Load model (no quantization, no adapters — full precision) ---
    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,  # bf16 for memory efficiency
        device_map="auto",
    )

    # Enable gradient checkpointing — trades compute for memory
    # Without this: OOM on 80GB GPU. With this: ~36 GB VRAM.
    model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,} ({trainable_params/total_params:.2%})")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load data ---
    print(f"\nLoading SFT data from {args.data_path}...")
    raw_data = load_sft_data(args.data_path)
    print(f"  {len(raw_data)} training examples")

    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(format_chat, remove_columns=["messages", "id"])

    # --- Training config ---
    output_dir = Path(args.output_dir)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,        # more warmup than LoRA — full model needs gentler start
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_strategy="no",      # don't save checkpoints — disk space is tight on rentals
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # --- Train ---
    total_steps = len(dataset) * args.epochs // effective_batch
    print(f"\nStarting training...")
    print(f"  Total steps: ~{total_steps}")
    print(f"  Estimated time: {total_steps * 3 / 60:.0f}-{total_steps * 5 / 60:.0f} min on H100")
    trainer.train()

    # --- Save full model ---
    print(f"\nSaving full model to {output_dir}...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)

    config = {
        "model_name": args.model_name,
        "method": "full_sft",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": effective_batch,
        "lr": args.lr,
        "max_seq_length": args.max_seq_length,
        "total_params": total_params,
        "training_examples": len(dataset),
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print("DONE! Next steps:")
    print(f"  1. Run eval: python -m eval.evaluate_rag --model_path {output_dir} --num_examples 200")
    print(f"  2. Full model saved at: {output_dir}")
    print(f"  3. Upload: huggingface-cli upload kaiwu598/filing-sense-full-sft-qwen2.5-3b {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
