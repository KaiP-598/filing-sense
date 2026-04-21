"""Configuration and constants for the FilingSense backend."""

from __future__ import annotations

import os

# Model
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "kaiwu598/filing-sense-grpo-qwen2.5-3b",
)

# HuggingFace Dataset
HF_CHUNKS_REPO = os.environ.get(
    "HF_CHUNKS_REPO",
    "kaiwu598/filing-sense-chunks",
)

# Retrieval
RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "5"))
USE_RERANKER = os.environ.get("USE_RERANKER", "true").lower() == "true"

# Supported tickers — must match what's in the HF dataset
TICKERS: list[dict] = [
    {"ticker": "NVDA",  "name": "NVIDIA",              "sector": "Semiconductors"},
    {"ticker": "AAPL",  "name": "Apple",                "sector": "Hardware/Software"},
    {"ticker": "MSFT",  "name": "Microsoft",            "sector": "Cloud/Software"},
    {"ticker": "GOOG",  "name": "Alphabet",             "sector": "Search/Cloud/AI"},
    {"ticker": "AMZN",  "name": "Amazon",               "sector": "Cloud/E-commerce"},
    {"ticker": "META",  "name": "Meta",                 "sector": "Social/AI"},
    {"ticker": "AVGO",  "name": "Broadcom",             "sector": "Semiconductors"},
    {"ticker": "TSLA",  "name": "Tesla",                "sector": "EV/AI"},
    {"ticker": "ORCL",  "name": "Oracle",               "sector": "Enterprise Software"},
    {"ticker": "CRM",   "name": "Salesforce",           "sector": "CRM/Cloud"},
    {"ticker": "ADBE",  "name": "Adobe",                "sector": "Creative Software"},
    {"ticker": "NOW",   "name": "ServiceNow",           "sector": "Enterprise Software"},
    {"ticker": "IBM",   "name": "IBM",                  "sector": "Enterprise/Cloud"},
    {"ticker": "AMD",   "name": "AMD",                  "sector": "Semiconductors"},
    {"ticker": "MU",    "name": "Micron",               "sector": "Memory/Semiconductors"},
    {"ticker": "INTC",  "name": "Intel",                "sector": "Semiconductors"},
    {"ticker": "TXN",   "name": "Texas Instruments",    "sector": "Semiconductors"},
    {"ticker": "AMAT",  "name": "Applied Materials",    "sector": "Semiconductor Equipment"},
    {"ticker": "LRCX",  "name": "Lam Research",         "sector": "Semiconductor Equipment"},
    {"ticker": "KLAC",  "name": "KLA",                  "sector": "Semiconductor Equipment"},
    {"ticker": "ADI",   "name": "Analog Devices",       "sector": "Semiconductors"},
    {"ticker": "CSCO",  "name": "Cisco",                "sector": "Networking"},
    {"ticker": "ANET",  "name": "Arista Networks",      "sector": "Networking"},
    {"ticker": "CRWD",  "name": "CrowdStrike",          "sector": "Cybersecurity"},
    {"ticker": "NFLX",  "name": "Netflix",              "sector": "Streaming"},
    {"ticker": "PLTR",  "name": "Palantir",             "sector": "AI/Defense"},
    {"ticker": "SNOW",  "name": "Snowflake",            "sector": "Cloud/Data"},
    {"ticker": "PYPL",  "name": "PayPal",               "sector": "Fintech"},
    {"ticker": "COIN",  "name": "Coinbase",             "sector": "Fintech/Crypto"},
    {"ticker": "UBER",  "name": "Uber",                 "sector": "Tech/Mobility"},
]

TICKER_SET = {t["ticker"] for t in TICKERS}
