#!/bin/bash
# =============================================================
# Full SFT training + eval (needs 80GB+ VRAM — H100/A100)
#
# Usage: SSH into a GPU instance, then:
#   bash scripts/run_full_sft.sh              # train + eval
#   bash scripts/run_full_sft.sh train        # training only
#   bash scripts/run_full_sft.sh eval         # eval only (model must exist)
#
# Expect: ~2-3 hr training + ~30 min eval on H100
# For LoRA eval, use: bash scripts/run_lora.sh eval
# =============================================================

set -e

MODE="${1:-all}"

echo "============================================"
echo "FilingSense — Full SFT (mode: $MODE)"
echo "============================================"
echo ""

# --- Step 1: Install dependencies ---
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

# Auto-detect GPU
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
GPU_MEM=$(python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_mem / 1e9:.0f}GB')" 2>/dev/null || echo "unknown")
echo "  Detected GPU: $GPU_NAME ($GPU_MEM)"

if [ "$MODE" = "train" ] || [ "$MODE" = "all" ]; then
    # Full SFT needs 80GB+ — check before wasting time
    GPU_MEM_GB=$(python3 -c "import torch; print(int(torch.cuda.get_device_properties(0).total_mem / 1e9))" 2>/dev/null || echo "0")
    if [ "$GPU_MEM_GB" -lt 40 ]; then
        echo "ERROR: Full SFT needs 80GB+ VRAM (detected: ${GPU_MEM_GB}GB)"
        echo "  Use an H100 or A100 80GB instance."
        echo "  For smaller GPUs, use LoRA instead: bash scripts/run_lora.sh"
        exit 1
    fi

    # Adjust batch size for GPU
    if echo "$GPU_NAME" | grep -qi "H100\|H200\|B200"; then
        BATCH_SIZE=4
        GRAD_ACCUM=4
    elif echo "$GPU_NAME" | grep -qi "A100"; then
        BATCH_SIZE=2
        GRAD_ACCUM=8
    else
        BATCH_SIZE=1
        GRAD_ACCUM=16
    fi
    echo "  Batch config: batch=$BATCH_SIZE, grad_accum=$GRAD_ACCUM (effective=$(($BATCH_SIZE * $GRAD_ACCUM)))"

    echo ""
    echo ">>> Full SFT Training (Qwen2.5-3B)..."
    echo "    All 1.7B params unfrozen, Epochs: 3, LR: 2e-5"
    echo ""
    python3 scripts/train_full_sft.py \
        --model_name Qwen/Qwen2.5-3B \
        --data_path data/sft_train.jsonl \
        --output_dir outputs/full-sft-qwen2.5-3b \
        --epochs 3 \
        --batch_size $BATCH_SIZE \
        --grad_accum $GRAD_ACCUM \
        --lr 2e-5
fi

if [ "$MODE" = "eval" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo ">>> End-to-end eval (Full SFT model)..."
    echo ""
    python3 -m eval.evaluate_rag \
        --model_path outputs/full-sft-qwen2.5-3b \
        --num_examples 200 \
        --use_reranker \
        --output_dir results

    # Save with descriptive name
    cp results/eval_results.json results/eval_full_sft_qwen2.5_3b.json
    cp results/eval_detailed_log.jsonl results/eval_detailed_full_sft.jsonl
    echo "  Saved: results/eval_full_sft_qwen2.5_3b.json"
    echo "  Saved: results/eval_detailed_full_sft.jsonl"
fi

echo ""
echo "============================================"
echo "DONE! Results:"
echo "============================================"
echo ""

# Print all result files
for f in results/eval_*_qwen2.5_3b.json; do
    if [ -f "$f" ]; then
        echo "--- $(basename $f) ---"
        cat "$f" | python3 -c "import sys,json; d=json.load(sys.stdin); e2e=d.get('end_to_end',{}); print(f'  Accuracy: {e2e.get(\"accuracy\",0):.1%}')" 2>/dev/null || cat "$f"
        echo ""
    fi
done

echo "Detailed logs for error taxonomy:"
ls -la results/eval_detailed_*.jsonl 2>/dev/null || echo "  (none)"
echo ""
echo "To pull results to your Mac:"
echo "  scp -P PORT root@HOST:/app/results/eval_*.json /path/to/filing-sense/results/"
echo "  scp -P PORT root@HOST:/app/results/eval_detailed_*.jsonl /path/to/filing-sense/results/"
