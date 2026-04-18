# FilingSense: Technical Analysis

A deep dive into building an AI system for answering financial questions from SEC filings — from retrieval to GRPO, with error analysis showing exactly where the pipeline fails.

## The Problem

SEC 10-K filings are dense, table-heavy documents. Answering questions like *"What was the change in cash provided by operating activities from 2010 to 2011?"* requires:

1. **Finding** the right table across thousands of pages
2. **Extracting** the correct numbers from the right rows/columns
3. **Reasoning** — performing the right math operation (subtraction, division, percentage change)

We benchmark on [FinQA](https://arxiv.org/abs/2109.00122) (Chen et al., EMNLP 2021): 8,281 Q&A pairs from S&P 500 10-K filings with ground-truth numerical answers.

## Architecture

```
                              FilingSense Pipeline
┌─────────┐    ┌──────────────────────────┐    ┌──────────────────┐    ┌────────┐
│  Query  │───▶│   Hybrid Retrieval       │───▶│   Reranker       │───▶│  LLM   │
│         │    │  BM25 + FAISS + RRF      │    │  cross-encoder   │    │  GRPO  │
└─────────┘    └──────────────────────────┘    └──────────────────┘    └────────┘
                 │                                                        │
                 ├─ BM25: keyword matching (44.5% recall)                 ├─ Base Qwen2.5-3B
                 ├─ FAISS: bge-small-en-v1.5 embeddings (28.0%)          ├─ + LoRA SFT
                 ├─ Reciprocal Rank Fusion (46.5%)                       ├─ + Full SFT
                 └─ Cross-encoder: ms-marco-MiniLM-L-6-v2                └─ + GRPO (68% gold)
```

## Results

### End-to-End Accuracy (200 FinQA test examples)

| Stage | E2E Accuracy | Training Cost | GPU Hours |
|---|---|---|---|
| Base Qwen2.5-3B + RAG | 11.5% | $0 | — |
| + LoRA SFT | 16.5% | ~$3 | 1 hr (1× H100) |
| + Full SFT | 16.5% | ~$15 | 3 hr (1× H100) |
| + GRPO | 17.0% | ~$25 | 5 hr (2× H100 SXM) |

### Retrieval (hybrid search over 6,624 chunks)

| Method | Recall@5 | MRR |
|---|---|---|
| BM25 only | 44.5% | 0.369 |
| FAISS only | 28.0% | 0.221 |
| Hybrid (BM25 + FAISS + RRF) | 46.5% | 0.374 |

### GRPO with Gold Context

When the model receives the *correct* context directly (bypassing retrieval):

| Model | Accuracy |
|---|---|
| Full SFT | ~40% |
| GRPO (best, step 80) | **68%** |
| GRPO (step 100, overfitted) | 60% |

The 68% vs 17% gap tells the whole story: **the model can reason, but retrieval caps the system.**

## Key Findings

### 1. BM25 Beats FAISS on Financial Data

BM25 (keyword matching) achieves 44.5% recall vs FAISS (semantic search) at 28.0%. Financial documents are terminology-heavy — exact terms like "operating income", "total debt", "fiscal year 2011" matter more than semantic similarity. Hybrid adds only 2% over BM25 alone.

**Takeaway:** For terminology-heavy domains, always benchmark BM25 before investing in embeddings — exact lexical matches often outperform semantic similarity.

### 2. LoRA = Full SFT on This Task

Both achieve 16.5% end-to-end accuracy. This isn't surprising — the accuracy ceiling is set by retrieval (46.5% recall), not model capacity. LoRA with 0.3% of parameters matches full fine-tuning because the bottleneck is upstream.

**Takeaway:** When retrieval caps end-to-end accuracy, LoRA (0.3% of parameters) matches full fine-tuning at 5× lower cost. Identify the bottleneck before scaling the model.

### 3. GRPO Learns Reasoning, Not Just Format

SFT teaches the model to *format* answers like the training data. GRPO teaches it to *reason correctly* — verified by the reward signal (exact-match on the numerical answer).

Evidence: GRPO jumps from ~40% to 68% on gold context, while SFT variants plateau. The model learned multi-step financial math (percentage change, ratios, growth rates) that SFT alone couldn't teach.

### 4. GRPO Overfits After 80 Steps

Training peaked at step 80 (68%) and degraded to 60% by step 100. This is classic RL overfitting — the model memorizes training-distribution patterns that don't generalize. Early stopping is essential.

### 5. Retrieval Is the Bottleneck (Error Taxonomy)

We classified every wrong answer across all three models:

| Error Category | LoRA SFT | Full SFT | GRPO |
|---|---|---|---|
| Correct | 33 (16.5%) | 33 (16.5%) | 34 (17.0%) |
| Retrieval miss | 106 (53.0%) | 104 (52.0%) | 104 (52.0%) |
| Wrong extraction | 49 (24.5%) | 51 (25.5%) | 49 (24.5%) |
| Wrong operation | 7 (3.5%) | 5 (2.5%) | 6 (3.0%) |
| Reasoning loop | 1 (0.5%) | 3 (1.5%) | 3 (1.5%) |
| Format/sign error | 4 (2.0%) | 4 (2.0%) | 4 (2.0%) |

**All three models have nearly identical error distributions.** The training method doesn't matter when retrieval fails — you can't reason over missing context.

Error definitions:
- **Retrieval miss**: correct chunk not in top-5 retrieved results
- **Wrong extraction**: right document retrieved, but model pulls numbers from wrong row/column
- **Wrong operation**: right numbers, wrong math (e.g., subtracted when should have divided)
- **Reasoning loop**: model generates excessive steps, often repeating calculations
- **Format/sign error**: correct computation but wrong sign, scale (% vs decimal), or rounding

## Production Roadmap

Ordered by expected accuracy impact:

### 1. Retrieval (52% of errors)
- **Table-aware chunking**: current chunking splits tables mid-row. Preserve table boundaries.
- **Domain embeddings**: replace bge-small with a financial-domain model (e.g., FinBERT-based).
- **Query expansion**: decompose "What was the change in X from 2010 to 2011?" into sub-queries for each year.
- **Increase top-k with better reranking**: retrieve 20, rerank to 5 with a stronger cross-encoder.

### 2. Table Understanding (24.5% of errors)
- **Structured table extraction**: parse tables into row/column format before feeding to the model, so it doesn't confuse adjacent cells.
- **Row/column highlighting**: include header mappings in the prompt (e.g., "Column 3 = fiscal year 2011").

### 3. Training (3% of errors)
- **More diverse training data**: FinQA has ~6,600 training examples. Adding ConvFinQA or TAT-QA would improve generalization.
- **Longer GRPO training with curriculum**: start with simple questions, increase difficulty.

## Training Details

### GRPO Configuration
- Base: Qwen2.5-3B (full SFT checkpoint)
- Generation: vLLM on GPU 1, training on GPU 0
- Group size: 8 responses per prompt, 16 prompts per step
- Reward: exact-match on extracted numerical answer
- Clipping: PPO-style with epsilon=0.2
- Learning rate: 5e-7 with cosine schedule
- Total: 80 steps (best checkpoint), ~5 hours on 2× H100 SXM

### LoRA SFT Configuration
- Rank: 16, alpha: 32, dropout: 0.05
- Target modules: all linear layers
- Training: 3 epochs on FinQA train set
- Total: ~1 hour on 1× H100

## Cost Analysis

| Component | Cost |
|---|---|
| LoRA SFT training | ~$3 |
| Full SFT training | ~$15 |
| GRPO training (2× H100) | ~$25 |
| Eval runs (3 models × RTX 4090) | ~$5 |
| **Total training + eval** | **~$48** |

Inference cost per query (self-hosted on RTX 4090):
- Model loading: ~6 GB VRAM
- Generation: ~5-10 sec/query
- At scale with vLLM batching: ~$0.001/query

## Models

All models are publicly available on HuggingFace:

- [`kaiwu598/filing-sense-lora-qwen2.5-3b`](https://huggingface.co/kaiwu598/filing-sense-lora-qwen2.5-3b) — LoRA adapter
- [`kaiwu598/filing-sense-full-sft-qwen2.5-3b`](https://huggingface.co/kaiwu598/filing-sense-full-sft-qwen2.5-3b) — Full SFT
- [`kaiwu598/filing-sense-grpo-qwen2.5-3b`](https://huggingface.co/kaiwu598/filing-sense-grpo-qwen2.5-3b) — GRPO (best checkpoint, 68% gold-context accuracy)

## Related Work

- **Fin-R1** (arXiv 2503.16252) — showed GRPO beats PPO/DPO on FinQA, achieving 76.0% with a 7B model
- **FinLoRA** (arXiv 2505.19819) — benchmarked LoRA methods on financial datasets
- **FinQA** (Chen et al., EMNLP 2021) — original dataset
- **GRPO from Scratch** — [my implementation](https://github.com/KaiP-598/grpo-from-scratch) with 10 ablation experiments

This project differs by providing a full production stack (RAG + SFT + GRPO + error taxonomy + deployment) rather than just training numbers, and by demonstrating that retrieval — not model quality — is the primary bottleneck in financial QA systems.
