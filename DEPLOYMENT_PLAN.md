# FilingSense Deployment Plan

Full-stack deployment: Next.js frontend + FastAPI backend + pre-built indexes for top 30 tech companies.

## Architecture

```
┌─────────────────────────────────┐
│  Vercel (free)                  │
│  Next.js + TypeScript frontend  │
│  - Company dropdown (30 tickers)│
│  - Question input               │
│  - Answer + reasoning display   │
│  - Sources accordion            │
└──────────────┬──────────────────┘
               │ POST /api/answer
               │ {ticker, question}
┌──────────────▼──────────────────┐
│  HuggingFace Spaces ($9/mo)     │
│  FastAPI backend                │
│  - Loads model + indexes once   │
│  - /health, /tickers, /answer   │
└─────────────────────────────────┘
               │
               │ loaded at startup
┌──────────────▼──────────────────┐
│  HuggingFace Datasets (free)    │
│  kaiwu598/filing-sense-chunks   │
│  - Pre-chunked 10-Ks (30 cos)   │
│  - ~5MB total                   │
└─────────────────────────────────┘
```

## Repo Structure

Everything lives in the same `filing-sense` repo:

```
filing-sense/
├── src/                        # existing ML pipeline (unchanged)
├── eval/                       # existing evaluation scripts
├── scripts/
│   └── build_live_chunks.py    # NEW: fetch + chunk 30 tickers, upload to HF
├── demo/
│   ├── backend/
│   │   ├── main.py             # NEW: FastAPI app
│   │   └── requirements.txt    # NEW: backend deps
│   └── frontend/               # NEW: Next.js app
│       ├── app/
│       │   └── page.tsx        # main page
│       ├── components/
│       │   ├── CompanySelector.tsx
│       │   ├── QuestionInput.tsx
│       │   └── AnswerDisplay.tsx
│       └── package.json
```

## Top 30 Tech Companies

All US-listed, all file 10-K with SEC EDGAR.

| # | Ticker | Company | Sector |
|---|---|---|---|
| 1 | NVDA | NVIDIA | Semiconductors |
| 2 | AAPL | Apple | Hardware/Software |
| 3 | MSFT | Microsoft | Cloud/Software |
| 4 | GOOG | Alphabet | Search/Cloud/AI |
| 5 | AMZN | Amazon | Cloud/E-commerce |
| 6 | META | Meta | Social/AI |
| 7 | AVGO | Broadcom | Semiconductors |
| 8 | TSLA | Tesla | EV/AI |
| 9 | ORCL | Oracle | Enterprise Software |
| 10 | MU | Micron | Memory/Semiconductors |
| 11 | AMD | AMD | Semiconductors |
| 12 | NFLX | Netflix | Streaming |
| 13 | PLTR | Palantir | AI/Defense |
| 14 | CSCO | Cisco | Networking |
| 15 | INTC | Intel | Semiconductors |
| 16 | AMAT | Applied Materials | Semiconductor Equipment |
| 17 | IBM | IBM | Enterprise/Cloud |
| 18 | CRM | Salesforce | CRM/Cloud |
| 19 | ADBE | Adobe | Creative Software |
| 20 | NOW | ServiceNow | Enterprise Software |
| 21 | CRWD | CrowdStrike | Cybersecurity |
| 22 | TXN | Texas Instruments | Semiconductors |
| 23 | ANET | Arista Networks | Networking |
| 24 | LRCX | Lam Research | Semiconductor Equipment |
| 25 | KLAC | KLA | Semiconductor Equipment |
| 26 | ADI | Analog Devices | Semiconductors |
| 27 | PYPL | PayPal | Fintech |
| 28 | COIN | Coinbase | Fintech/Crypto |
| 29 | SNOW | Snowflake | Cloud/Data |
| 30 | UBER | Uber | Tech/Mobility |

## Step 1 — Data Pipeline (run once on Vast.ai)

**Script:** `scripts/build_live_chunks.py`

What it does:
1. Fetches latest 10-K for each of the 30 tickers from SEC EDGAR
2. Strips XBRL metadata, extracts plain text
3. Chunks each filing into ~2K char chunks with 200 char overlap
4. Uploads all chunks as JSONL to `kaiwu598/filing-sense-chunks` on HF Datasets

EDGAR User-Agent: set via `EDGAR_USER_AGENT` env var locally (use `filingsense@gmail.com` for the build run — this script runs once, not in production).

**Output:** ~115 chunks per company × 30 companies = ~3,450 chunks total, ~5MB.

**Time estimate:** ~5 min on Vast.ai (mostly network I/O, 30 EDGAR fetches).

## Step 2 — FastAPI Backend

**File:** `demo/backend/main.py`

Endpoints:
```
GET  /health    → {"status": "ok", "tickers_loaded": 30}
GET  /tickers   → [{"ticker": "AAPL", "name": "Apple", "sector": "..."}, ...]
POST /answer    → {"ticker": "AAPL", "question": "..."} 
                → {"answer": "...", "reasoning": "...", "sources": [...], "filing_date": "..."}
```

Startup sequence (one-time, ~2-3 min when Space boots):
1. Download chunks from `kaiwu598/filing-sense-chunks` HF Dataset
2. Build BM25 + FAISS index per ticker (30 separate indexes, all in memory)
3. Load GRPO model (`kaiwu598/filing-sense-grpo-qwen2.5-3b`)
4. Ready to serve

**Deploy:** HuggingFace Spaces, persistent tier ($9/mo, no sleep). The 2-3 min startup happens once on first deploy — after that the process stays alive indefinitely. Every user request hits an already-warm model (~6-12 sec response time).

## Step 3 — Next.js Frontend

**Directory:** `demo/frontend/`

Stack: Next.js 14, TypeScript, Tailwind CSS. Dark theme (Linear/Vercel style).

Three components:
- `CompanySelector` — searchable dropdown, shows ticker + company name + sector
- `QuestionInput` — text input + submit, 4 example questions that populate on click
- `AnswerDisplay` — loading skeleton → Reasoning block → Answer block → Sources accordion

One environment variable:
```
NEXT_PUBLIC_API_URL=https://kaiwu598-filing-sense-backend.hf.space
```

**Deploy:** Vercel, free tier, auto-deploys from GitHub on push.

## Step 4 — Deploy

1. Run `scripts/build_live_chunks.py` on Vast.ai → uploads to HF Datasets
2. Push `demo/backend/` to HuggingFace Spaces → enable persistent ($9/mo)
3. Push `demo/frontend/` → connect to Vercel → set `NEXT_PUBLIC_API_URL` env var
4. Test end-to-end

## Cost Summary

| Service | Cost |
|---|---|
| HF Datasets (chunks) | Free |
| HF Spaces persistent (backend) | $9/mo |
| Vercel (frontend) | Free |
| Domain (optional) | ~$12/yr |
| **Total** | **~$9/mo** |

## What We're NOT Building (yet)

- User auth / query history
- Multiple filings per company (just latest 10-K)
- Mobile optimization
- Streaming responses (answer appears word by word)
- Landing page / blog post
