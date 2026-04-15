"""LoRA SFT training on FinQA using Unsloth.

Unsloth provides 2x faster QLoRA training by optimizing attention kernels.
QLoRA = 4-bit quantized base model + LoRA adapters (only adapters are trained).

Usage (on GPU machine):
    python scripts/train_lora.py
    python scripts/train_lora.py --model_name Qwen/Qwen2.5-3B --epochs 3
    python scripts/train_lora.py --model_name google/gemma-2-2b  # dual-model comparison

Interview notes:
    - QLoRA: base model quantized to 4-bit (NF4), LoRA adapters in fp16/bf16
    - Only ~25M params trained (rank 16 × all attention projections) vs 3B total (<1%)
    - Memory: ~8 GB VRAM vs ~36 GB for full fine-tune
    - Why rank 16: weight update matrices are empirically low-rank; 16 directions
      capture 95%+ of the useful gradient signal (SVD insight)
    - Target modules: q,k,v,o projections in all transformer blocks
      (NOT FFN — diminishing returns, NOT just last layer — reasoning needs changes throughout)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


def load_sft_data(data_path: str) -> list[dict]:
    """Load chat-format JSONL data."""
    data = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data


def main():
    parser = argparse.ArgumentParser(description="LoRA SFT on FinQA")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B",
                        help="Base model from HuggingFace")
    parser.add_argument("--data_path", type=str, default="data/sft_train.jsonl",
                        help="Path to SFT training data (JSONL)")
    parser.add_argument("--output_dir", type=str, default="outputs/lora-qwen2.5-3b",
                        help="Directory to save LoRA adapter")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate (2e-4 is standard for LoRA)")
    parser.add_argument("--rank", type=int, default=16,
                        help="LoRA rank (16 = good balance of quality vs params)")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Max sequence length (context + response)")
    parser.add_argument("--no_unsloth", action="store_true",
                        help="Use standard PEFT instead of Unsloth (fallback)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"LoRA SFT Training")
    print(f"  Model: {args.model_name}")
    print(f"  Rank: {args.rank}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"  Learning rate: {args.lr}")
    print("=" * 60)

    # --- Load model with LoRA ---
    if not args.no_unsloth:
        try:
            from unsloth import FastLanguageModel
            print("\nLoading model with Unsloth (2x faster)...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model_name,
                max_seq_length=args.max_seq_length,
                load_in_4bit=True,  # QLoRA: 4-bit quantized base
                dtype=None,         # auto-detect bf16/fp16
            )

            model = FastLanguageModel.get_peft_model(
                model,
                r=args.rank,
                lora_alpha=args.rank * 2,   # alpha = 2 * rank is standard
                lora_dropout=0,              # Unsloth optimizes for 0 dropout
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    # Skip gate_proj, up_proj, down_proj (FFN) — diminishing returns
                ],
                bias="none",
                use_gradient_checkpointing="unsloth",  # saves 30% memory
            )
            print("  Unsloth LoRA model ready")
        except ImportError:
            print("  Unsloth not available, falling back to PEFT...")
            args.no_unsloth = True

    if args.no_unsloth:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        print("\nLoading model with PEFT + bitsandbytes...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
        )

        # Required for 4-bit: enables gradient checkpointing + fixes input gradients
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        print("  PEFT LoRA model ready (gradient checkpointing ON)")

    # Print trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Trainable params: {trainable:,} / {total:,} ({trainable/total:.2%})")

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load data ---
    print(f"\nLoading SFT data from {args.data_path}...")
    raw_data = load_sft_data(args.data_path)
    print(f"  {len(raw_data)} training examples")

    # Convert to HF Dataset for SFTTrainer
    from datasets import Dataset

    def format_chat(example):
        """Apply tokenizer's chat template to messages."""
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
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        seed=42,
        report_to="none",  # no W&B needed
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # --- Train ---
    print(f"\nStarting training...")
    print(f"  Total steps: ~{len(dataset) * args.epochs // (args.batch_size * args.grad_accum)}")
    trainer.train()

    # --- Save adapter ---
    print(f"\nSaving LoRA adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Also save training config for reproducibility
    config = {
        "model_name": args.model_name,
        "rank": args.rank,
        "alpha": args.rank * 2,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "max_seq_length": args.max_seq_length,
        "trainable_params": trainable,
        "total_params": total,
        "training_examples": len(dataset),
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print("DONE! Next steps:")
    print(f"  1. Run eval: python -m eval.evaluate_rag --model_path {args.model_name} "
          f"--adapter_path {output_dir} --num_examples 200")
    print(f"  2. Adapter saved at: {output_dir}")
    print(f"  3. Upload to HF Hub: huggingface-cli upload {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
