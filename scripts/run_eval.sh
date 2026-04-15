#!/bin/bash
# =============================================================
# Vast.ai setup + eval script for FilingSense
#
# Usage: SSH into your Vast.ai instance, then:
#   bash scripts/run_eval.sh [retrieval|e2e|all]
#
# Examples:
#   bash scripts/run_eval.sh retrieval   # retrieval only (already done)
#   bash scripts/run_eval.sh e2e         # end-to-end only (needs GPU)
#   bash scripts/run_eval.sh all         # both
#   bash scripts/run_eval.sh             # defaults to e2e
# =============================================================

set -e

MODE="${1:-e2e}"

echo "============================================"
echo "FilingSense — Vast.ai Eval (mode: $MODE)"
echo "============================================"

# --- Step 1: Install deps (pin versions to avoid torchcodec issue) ---
echo "[1/3] Installing dependencies..."
pip install -q 'sentence-transformers<4' 'transformers<5' datasets faiss-cpu rank-bm25 accelerate tqdm numpy

# --- Step 2: Download everything upfront ---
echo "[2/3] Pre-downloading models + dataset..."
python3 -c "
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
print('Downloading dataset...')
load_dataset('wandb/finqa-data-processed')
print('Downloading embedding model...')
SentenceTransformer('BAAI/bge-small-en-v1.5')
"

if [ "$MODE" = "e2e" ] || [ "$MODE" = "all" ]; then
    python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
print('Downloading Qwen2.5-3B...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B', torch_dtype=torch.float16)
print('Model downloaded!')
"
fi

# --- Step 3: Run eval ---
echo "[3/3] Running evaluation..."

if [ "$MODE" = "retrieval" ] || [ "$MODE" = "all" ]; then
    echo ">>> Retrieval eval..."
    python3 -m eval.evaluate_rag --num_examples 200 --output_dir results
fi

if [ "$MODE" = "e2e" ] || [ "$MODE" = "all" ]; then
    echo ">>> End-to-end eval (with reranker)..."
    python3 -m eval.evaluate_rag --model_path Qwen/Qwen2.5-3B --num_examples 200 --use_reranker --output_dir results
fi

echo ""
echo "============================================"
echo "DONE! Results:"
echo "============================================"
cat results/eval_results.json
