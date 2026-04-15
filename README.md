# FilingSense

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

AI analyst for SEC filings. Ask financial questions about any public company — powered by RAG, agentic workflows, and custom fine-tuning with GRPO.

> **Work in progress.** Core RAG pipeline is functional. Fine-tuning, agent, and live EDGAR mode coming soon.

## What This Does

FilingSense answers multi-step financial questions by retrieving relevant sections from SEC 10-K filings and reasoning over them:

```
User:  "What was the pension cost growth rate from 2012 to 2013?"
       → retrieves the right table from Union Pacific's 10-K
       → calculates: (111 - 89) / 89 = 24.7%
```

## Architecture

```
Query → Hybrid Retrieval (BM25 + FAISS + reranker) → LLM → Answer
         │                                              │
         ├─ BM25: exact keyword matching                ├─ Base model (Qwen2.5-3B)
         ├─ FAISS: semantic similarity                  ├─ + LoRA SFT (coming)
         ├─ RRF: merge both rankings                    ├─ + Full SFT (coming)
         └─ Cross-encoder: precision rerank             └─ + GRPO (coming)
```

## Pipeline Progression (benchmarked on FinQA)

| Method | Accuracy | Status |
|---|---|---|
| Base model | ~25% | Coming |
| + RAG | ~50% | **In progress** |
| + RAG + LoRA SFT | ~63% | Coming |
| + RAG + Full SFT | ~66% | Coming |
| + RAG + Full SFT + GRPO | ~73% | Coming |

## Project Structure

```
src/
  chunking.py      # Section-aware chunking for 10-K filings
  indexing.py       # FAISS (dense) + BM25 (sparse) dual index
  retrieval.py      # Hybrid search + RRF + cross-encoder reranker
  generation.py     # Prompt building + LLM answer generation
data/
  download_finqa.py # Download FinQA dataset from HuggingFace
eval/
  (coming)          # Evaluation scripts and metrics
tests/
  (coming)          # Unit tests
```

## Dataset

[FinQA](https://arxiv.org/abs/2109.00122) (Chen et al., EMNLP 2021): 8,281 Q&A pairs grounded in SEC 10-K annual reports from S&P 500 companies, 1999-2019. Each example includes the source table, surrounding text, question, step-by-step calculation program, and ground-truth answer.

## Quick Start

```bash
pip install -r requirements.txt
python data/download_finqa.py
```

## Related Work

- [Fin-R1](https://arxiv.org/abs/2503.16252) — showed GRPO beats PPO/DPO on FinQA
- [FinLoRA](https://arxiv.org/abs/2505.19819) — benchmarked LoRA methods on financial datasets
- [GRPO from Scratch](https://github.com/KaiP-598/grpo-from-scratch) — my from-scratch implementation of the GRPO algorithm

This project differs by: (a) comparing two model families to isolate cost-efficiency tradeoffs, (b) shipping the full production stack (RAG + agent + SFT + GRPO + live demo), and (c) building GRPO from scratch rather than using TRL.

## License

MIT
