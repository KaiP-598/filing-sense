"""Fetch and chunk 10-K filings for the top 30 tech companies.

Downloads each company's latest 10-K from SEC EDGAR, chunks it into
~2K char segments, and uploads to HuggingFace Datasets.

Run once on Vast.ai (mostly network I/O, ~5-10 min):

    export EDGAR_USER_AGENT="FilingSense filingsense@gmail.com"
    export HF_TOKEN="hf_..."
    python scripts/build_live_chunks.py

Output: huggingface.co/datasets/kaiwu598/filing-sense-chunks
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from src.edgar import fetch_latest_10k
from src.chunking import chunk_raw_text


TICKERS = [
    # Mega-cap tech
    ("NVDA",  "NVIDIA",             "Semiconductors"),
    ("AAPL",  "Apple",              "Hardware/Software"),
    ("MSFT",  "Microsoft",          "Cloud/Software"),
    ("GOOG",  "Alphabet",           "Search/Cloud/AI"),
    ("AMZN",  "Amazon",             "Cloud/E-commerce"),
    ("META",  "Meta",               "Social/AI"),
    ("AVGO",  "Broadcom",           "Semiconductors"),
    ("TSLA",  "Tesla",              "EV/AI"),
    # Enterprise software
    ("ORCL",  "Oracle",             "Enterprise Software"),
    ("CRM",   "Salesforce",         "CRM/Cloud"),
    ("ADBE",  "Adobe",              "Creative Software"),
    ("NOW",   "ServiceNow",         "Enterprise Software"),
    ("IBM",   "IBM",                "Enterprise/Cloud"),
    # Semiconductors
    ("AMD",   "AMD",                "Semiconductors"),
    ("MU",    "Micron",             "Memory/Semiconductors"),
    ("INTC",  "Intel",              "Semiconductors"),
    ("TXN",   "Texas Instruments",  "Semiconductors"),
    ("AMAT",  "Applied Materials",  "Semiconductor Equipment"),
    ("LRCX",  "Lam Research",       "Semiconductor Equipment"),
    ("KLAC",  "KLA",                "Semiconductor Equipment"),
    ("ADI",   "Analog Devices",     "Semiconductors"),
    # Networking/Security
    ("CSCO",  "Cisco",              "Networking"),
    ("ANET",  "Arista Networks",    "Networking"),
    ("CRWD",  "CrowdStrike",        "Cybersecurity"),
    # Streaming/Media
    ("NFLX",  "Netflix",            "Streaming"),
    # AI/Data
    ("PLTR",  "Palantir",           "AI/Defense"),
    ("SNOW",  "Snowflake",          "Cloud/Data"),
    # Fintech/Mobility
    ("PYPL",  "PayPal",             "Fintech"),
    ("COIN",  "Coinbase",           "Fintech/Crypto"),
    ("UBER",  "Uber",               "Tech/Mobility"),
]


def fetch_and_chunk(ticker: str, company: str, sector: str) -> list[dict]:
    """Fetch 10-K for one ticker and return chunks as plain dicts."""
    text, metadata = fetch_latest_10k(ticker)
    chunks = chunk_raw_text(text, ticker=ticker, date=metadata["date"])

    records = []
    for chunk in chunks:
        records.append({
            "chunk_id":   chunk.chunk_id,
            "ticker":     ticker,
            "company":    company,
            "sector":     sector,
            "date":       metadata["date"],
            "url":        metadata["url"],
            "chunk_idx":  chunk.metadata["chunk_idx"],
            "text":       chunk.text,
        })

    return records


def save_locally(all_records: list[dict], output_dir: str) -> Path:
    """Save all chunks to a local JSONL file."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "chunks.jsonl"
    with open(path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(all_records)} chunks to {path}")
    return path


def upload_to_hf(jsonl_path: Path, repo_id: str, hf_token: str):
    """Upload the chunks JSONL to a HuggingFace Dataset repo."""
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)

    # Create repo if it doesn't exist
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    api.upload_file(
        path_or_fileobj=str(jsonl_path),
        path_in_repo="chunks.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded to https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/live_chunks")
    parser.add_argument("--repo_id", default="kaiwu598/filing-sense-chunks")
    parser.add_argument("--tickers", nargs="+", help="subset of tickers to process (default: all 30)")
    parser.add_argument("--skip_upload", action="store_true", help="save locally only, don't upload to HF")
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args()

    # Filter tickers if subset requested
    tickers_to_run = TICKERS
    if args.tickers:
        subset = {t.upper() for t in args.tickers}
        tickers_to_run = [(t, c, s) for t, c, s in TICKERS if t in subset]
        print(f"Running subset: {[t for t, _, _ in tickers_to_run]}")

    all_records = []
    failed = []

    for i, (ticker, company, sector) in enumerate(tickers_to_run):
        print(f"\n[{i+1}/{len(tickers_to_run)}] {ticker} — {company}")
        try:
            records = fetch_and_chunk(ticker, company, sector)
            all_records.extend(records)
            print(f"  → {len(records)} chunks")
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append((ticker, str(e)))

        # Be polite to EDGAR — already sleeping inside fetch_latest_10k
        # but add a small extra gap between companies
        if i < len(tickers_to_run) - 1:
            time.sleep(0.5)

    print(f"\n{'='*50}")
    print(f"Total chunks: {len(all_records)}")
    print(f"Failed: {len(failed)}")
    for ticker, err in failed:
        print(f"  {ticker}: {err}")
    print(f"{'='*50}\n")

    if not all_records:
        print("No chunks generated, exiting.")
        return

    jsonl_path = save_locally(all_records, args.output_dir)

    if not args.skip_upload:
        if not args.hf_token:
            print("No HF_TOKEN set — skipping upload. Use --skip_upload to suppress this warning.")
        else:
            upload_to_hf(jsonl_path, args.repo_id, args.hf_token)


if __name__ == "__main__":
    main()
