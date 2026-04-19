# FilingSense

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

AI system for answering financial questions from SEC 10-K filings — powered by hybrid retrieval (BM25 + FAISS), cross-encoder reranking, and GRPO-trained reasoning.

Benchmarked on [FinQA](https://arxiv.org/abs/2109.00122): 8,281 Q&A pairs from S&P 500 10-K filings with ground-truth numerical answers.

## What This Does

```
User:  "What was the pension cost growth rate from 2012 to 2013?"
       → retrieves the right table from Union Pacific's 10-K
       → calculates: (111 - 89) / 89 = 24.7%
```

## Architecture

```
Query → Hybrid Retrieval (BM25 + FAISS + reranker) → LLM → Answer
         │                                              │
         ├─ BM25: exact keyword matching (44.5%)        ├─ Base Qwen2.5-3B
         ├─ FAISS: bge-small-en-v1.5 (28.0%)           ├─ + LoRA SFT
         ├─ RRF: merge both rankings (46.5%)            ├─ + Full SFT
         └─ Cross-encoder: ms-marco-MiniLM-L-6-v2      └─ + GRPO (68% gold)
```

## Results (200 FinQA test examples)

| Stage | E2E Accuracy |
|---|---|
| Base Qwen2.5-3B + RAG | 11.5% |
| + LoRA SFT | 16.5% |
| + Full SFT | 16.5% |
| + GRPO | 17.0% |
| + GRPO + LangGraph Agent | **20.5%** |

With gold context (bypassing retrieval), GRPO achieves **68%** — the 68% vs 17% gap shows retrieval is the bottleneck, not model quality.

For detailed analysis, error taxonomy, and production roadmap, see [ANALYSIS.md](ANALYSIS.md).

## Models

All checkpoints on HuggingFace:

- [`kaiwu598/filing-sense-lora-qwen2.5-3b`](https://huggingface.co/kaiwu598/filing-sense-lora-qwen2.5-3b) — LoRA adapter
- [`kaiwu598/filing-sense-full-sft-qwen2.5-3b`](https://huggingface.co/kaiwu598/filing-sense-full-sft-qwen2.5-3b) — Full SFT
- [`kaiwu598/filing-sense-grpo-qwen2.5-3b`](https://huggingface.co/kaiwu598/filing-sense-grpo-qwen2.5-3b) — GRPO (best checkpoint)

## Project Structure

```
src/
  chunking.py      # Section-aware chunking for 10-K filings
  indexing.py       # FAISS (dense) + BM25 (sparse) dual index
  retrieval.py      # Hybrid search + RRF + cross-encoder reranker
  generation.py     # Prompt building + LLM answer generation
eval/
  evaluate_rag.py   # End-to-end RAG evaluation on FinQA
  error_taxonomy.py # Error classification across all models
data/
  download_finqa.py # Download FinQA dataset from HuggingFace
scripts/
  run_lora_sft.sh   # LoRA fine-tuning script
  run_full_sft.sh   # Full SFT script
  run_grpo.sh       # GRPO training script
```

## Quick Start

```bash
pip install -r requirements.txt
python data/download_finqa.py
python -m eval.evaluate_rag --model_path Qwen/Qwen2.5-3B --num_examples 200
```

## Related Work

- [Fin-R1](https://arxiv.org/abs/2503.16252) — showed GRPO beats PPO/DPO on FinQA, achieving 76.0% with a 7B model
- [FinLoRA](https://arxiv.org/abs/2505.19819) — benchmarked LoRA methods on financial datasets
- [FinQA](https://arxiv.org/abs/2109.00122) (Chen et al., EMNLP 2021) — original dataset
- [GRPO from Scratch](https://github.com/KaiP-598/grpo-from-scratch) — my from-scratch implementation with 10 ablation experiments

## License

MIT
