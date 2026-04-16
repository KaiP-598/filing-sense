#!/bin/bash
# =============================================================
# LoRA SFT training + eval (any CUDA GPU machine)
#
# Usage: SSH into a GPU instance, then:
#   bash scripts/run_lora.sh              # train + eval (default)
#   bash scripts/run_lora.sh train        # training only
#   bash scripts/run_lora.sh eval         # eval only (adapter must exist)
#
# Expect: ~1-2 hours training + ~30 min eval on RTX 3090/4090
# Requirements: NVIDIA GPU with 24GB+ VRAM, CUDA 12.4+
# =============================================================

set -e

MODE="${1:-all}"

echo "============================================"
echo "FilingSense — LoRA SFT (mode: $MODE)"
echo "============================================"
echo ""

# --- Step 1: Force-install exact pinned dependencies ---
# Always install, never trust pre-installed packages (Vast.ai images have broken combos)
echo "[1/4] Installing pinned dependencies..."
echo "  Removing conflicting packages..."
pip uninstall -y -q transformers trl sentence-transformers xformers \
    torchvision torchaudio torchcodec torchao unsloth 2>/dev/null || true
echo "  Installing torch 2.5.1 + transformers 4.44.2 (verified combo)..."
pip install -q --no-cache-dir \
    'torch==2.5.1+cu124' \
    --index-url https://download.pytorch.org/whl/cu124
pip install -q --no-cache-dir \
    'transformers==4.44.2' 'sentence-transformers==3.1.1' 'trl==0.11.4' \
    'peft>=0.13,<0.14' datasets faiss-cpu rank-bm25 accelerate bitsandbytes rich tqdm numpy
echo "  Verifying imports..."
python3 -c "
from trl import SFTTrainer
from peft import LoraConfig
from sentence_transformers import SentenceTransformer
import torch, transformers
print(f'  torch={torch.__version__}, transformers={transformers.__version__}')
print('  All imports OK')
"

# --- Step 2: Pre-download models + data ---
echo "[2/4] Pre-downloading models + dataset..."
python3 -c "
from datasets import load_dataset
print('Downloading dataset...')
load_dataset('wandb/finqa-data-processed')
print('Downloading embedding model...')
from sentence_transformers import SentenceTransformer
SentenceTransformer('BAAI/bge-small-en-v1.5')
"

# --- Step 3: Prepare SFT data ---
echo "[3/4] Preparing SFT training data..."
python3 scripts/prepare_sft_data.py --output data/sft_train.jsonl

# --- Step 4: Train and/or eval ---
echo "[4/4] Running..."

if [ "$MODE" = "train" ] || [ "$MODE" = "all" ]; then
    # Auto-detect GPU and set batch size accordingly
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
    echo "  Detected GPU: $GPU_NAME"

    # H100/A100 can handle larger batches → faster training
    if echo "$GPU_NAME" | grep -qi "H100\|A100"; then
        BATCH_SIZE=16
        GRAD_ACCUM=1
        echo "  Using large batch (H100/A100 mode): batch=$BATCH_SIZE, grad_accum=$GRAD_ACCUM"
    else
        BATCH_SIZE=4
        GRAD_ACCUM=4
        echo "  Using standard batch: batch=$BATCH_SIZE, grad_accum=$GRAD_ACCUM"
    fi

    echo ""
    echo ">>> LoRA Training (Qwen2.5-3B)..."
    echo "    Rank: 16, Epochs: 3, LR: 2e-4"
    echo ""
    python3 scripts/train_lora.py \
        --model_name Qwen/Qwen2.5-3B \
        --data_path data/sft_train.jsonl \
        --output_dir outputs/lora-qwen2.5-3b \
        --epochs 3 \
        --batch_size $BATCH_SIZE \
        --grad_accum $GRAD_ACCUM \
        --lr 2e-4 \
        --rank 16 \
        --no_unsloth
fi

if [ "$MODE" = "eval" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo ">>> End-to-end eval (LoRA model)..."
    echo ""
    # Use local adapter if it exists (just trained), otherwise pull from HuggingFace Hub
    if [ -d "outputs/lora-qwen2.5-3b" ]; then
        ADAPTER_PATH="outputs/lora-qwen2.5-3b"
    else
        ADAPTER_PATH="kaiwu598/filing-sense-lora-qwen2.5-3b"
    fi
    echo "  Adapter: $ADAPTER_PATH"

    python3 -m eval.evaluate_rag \
        --model_path Qwen/Qwen2.5-3B \
        --adapter_path "$ADAPTER_PATH" \
        --num_examples 200 \
        --use_reranker \
        --output_dir results
fi

echo ""
echo "============================================"
echo "DONE! Results:"
echo "============================================"
cat results/eval_results.json
echo ""
echo ""
echo "Adapter saved at: outputs/lora-qwen2.5-3b/"
echo "To download: tar czf lora-adapter.tar.gz outputs/lora-qwen2.5-3b/"
