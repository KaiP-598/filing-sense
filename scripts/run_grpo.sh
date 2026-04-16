#!/bin/bash
# =============================================================
# GRPO training + eval (needs 2 GPUs — 80GB+ each)
#
# GRPO requires TWO GPUs:
#   GPU 0: training model (forward/backward pass)
#   GPU 1: vLLM inference engine (generates candidate responses)
#
# Usage: SSH into a 2-GPU instance, then:
#   bash scripts/run_grpo.sh              # train + eval
#   bash scripts/run_grpo.sh train        # training only
#   bash scripts/run_grpo.sh eval         # eval only (model must exist)
#
# Prerequisite: Full SFT model must exist at outputs/full-sft-qwen2.5-3b/
#   OR download from HuggingFace:
#   huggingface-cli download kaiwu598/filing-sense-full-sft-qwen2.5-3b \
#       --local-dir outputs/full-sft-qwen2.5-3b
# =============================================================

set -e

MODE="${1:-all}"

echo "============================================"
echo "FilingSense — GRPO Training (mode: $MODE)"
echo "============================================"
echo ""

# --- Step 1: Install dependencies ---
echo "[1/3] Installing dependencies..."
pip uninstall -y -q transformers trl sentence-transformers xformers \
    torchvision torchaudio torchcodec torchao unsloth 2>/dev/null || true

echo "  Installing torch 2.5.1 + vLLM..."
pip install -q --no-cache-dir \
    'torch==2.5.1+cu124' \
    --index-url https://download.pytorch.org/whl/cu124
pip install -q --no-cache-dir \
    'transformers==4.44.2' 'sentence-transformers==3.1.1' 'trl==0.11.4' \
    'peft>=0.13,<0.14' datasets faiss-cpu rank-bm25 accelerate bitsandbytes \
    rich tqdm numpy vllm

echo "  Verifying imports..."
python3 -c "
import torch, transformers, vllm
print(f'  torch={torch.__version__}, transformers={transformers.__version__}')
print(f'  vllm={vllm.__version__}')
print(f'  GPUs available: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
print('  All imports OK')
"

# Check GPU count
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
if [ "$GPU_COUNT" -lt 2 ]; then
    echo "ERROR: GRPO requires 2 GPUs (found $GPU_COUNT)"
    echo "  GPU 0 = training, GPU 1 = vLLM inference"
    echo "  Rent a 2-GPU instance (e.g. 2x H100, 2x A100)"
    exit 1
fi

# --- Step 2: Get SFT model ---
echo "[2/3] Checking for SFT model..."
if [ ! -d "outputs/full-sft-qwen2.5-3b" ]; then
    echo "  SFT model not found locally. Downloading from HuggingFace..."
    pip install -q huggingface_hub
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('kaiwu598/filing-sense-full-sft-qwen2.5-3b',
                  local_dir='outputs/full-sft-qwen2.5-3b')
print('  Downloaded SFT model')
"
else
    echo "  Found local SFT model at outputs/full-sft-qwen2.5-3b"
fi

# --- Step 3: Train and/or eval ---
echo "[3/3] Running..."

if [ "$MODE" = "train" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo ">>> GRPO Training (starting from Full SFT model)..."
    echo "    100 GRPO steps, G=8, 16 prompts/step"
    echo ""
    python3 scripts/train_grpo.py \
        --model_path outputs/full-sft-qwen2.5-3b \
        --output_dir outputs/grpo-qwen2.5-3b \
        --n_steps 100 \
        --group_size 8 \
        --n_prompts 16 \
        --lr 2e-5 \
        --eval_every 10
fi

if [ "$MODE" = "eval" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo ">>> End-to-end eval (GRPO model)..."
    echo ""
    python3 -m eval.evaluate_rag \
        --model_path outputs/grpo-qwen2.5-3b \
        --num_examples 200 \
        --use_reranker \
        --output_dir results

    # Save with descriptive name
    cp results/eval_results.json results/eval_grpo_qwen2.5_3b.json
    cp results/eval_detailed_log.jsonl results/eval_detailed_grpo.jsonl
    echo "  Saved: results/eval_grpo_qwen2.5_3b.json"
    echo "  Saved: results/eval_detailed_grpo.jsonl"
fi

echo ""
echo "============================================"
echo "DONE! Results:"
echo "============================================"
echo ""

for f in results/eval_*_qwen2.5_3b.json; do
    if [ -f "$f" ]; then
        echo "--- $(basename $f) ---"
        cat "$f" | python3 -c "import sys,json; d=json.load(sys.stdin); e2e=d.get('end_to_end',{}); print(f'  Accuracy: {e2e.get(\"accuracy\",0):.1%}')" 2>/dev/null || cat "$f"
        echo ""
    fi
done

echo "To upload GRPO model:"
echo "  hf upload kaiwu598/filing-sense-grpo-qwen2.5-3b outputs/grpo-qwen2.5-3b"
echo ""
echo "To pull results to your Mac:"
echo "  scp -P PORT root@HOST:/app/results/eval_*grpo*.json /path/to/filing-sense/results/"
echo "  scp -P PORT root@HOST:/app/results/eval_detailed_grpo.jsonl /path/to/filing-sense/results/"
