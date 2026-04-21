"""EDGAR API client for fetching live SEC 10-K filings.

Three API calls to get a filing:
    1. Ticker → CIK  (SEC company tickers endpoint)
    2. CIK → accession number + primary document  (submissions endpoint)
    3. Accession + document → raw HTML → parsed text  (EDGAR archives)

All three endpoints are free, no auth required. SEC rate limit is 10 req/s.
We sleep 0.11s between calls to stay safely under.

Usage:
    from src.edgar import fetch_latest_10k
    text, metadata = fetch_latest_10k("AAPL")
    # text: ~200K chars of plain text from the filing
    # metadata: {"ticker": "AAPL", "cik": 320193, "date": "2024-11-01", ...}
"""

from __future__ import annotations

import time

import requests
from bs4 import BeautifulSoup


import os
USER_AGENT = os.environ.get("EDGAR_USER_AGENT", "FilingSense/1.0 (github.com/KaiP-598/filing-sense)")
HEADERS = {"User-Agent": USER_AGENT}

TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik_padded}.json"
FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"
FILING_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{acc_dashed}-index.json"

_TICKER_CACHE: dict[str, int] | None = None  # ticker -> cik, loaded once


def _get_ticker_map() -> dict[str, int]:
    """Fetch and cache the SEC ticker → CIK mapping."""
    global _TICKER_CACHE
    if _TICKER_CACHE is not None:
        return _TICKER_CACHE

    resp = requests.get(TICKERS_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    fields = data["fields"]
    ticker_idx = fields.index("ticker")
    cik_idx = fields.index("cik")

    _TICKER_CACHE = {
        row[ticker_idx].upper(): row[cik_idx]
        for row in data["data"]
    }
    return _TICKER_CACHE


def ticker_to_cik(ticker: str) -> int:
    """Convert a stock ticker to SEC CIK number.

    Args:
        ticker: e.g. "AAPL", "MSFT", "TSLA"

    Returns:
        CIK as integer, e.g. 320193 for Apple

    Raises:
        ValueError: if ticker not found in SEC database
    """
    mapping = _get_ticker_map()
    upper = ticker.upper()
    if upper not in mapping:
        raise ValueError(f"Ticker '{ticker}' not found in SEC database")
    return mapping[upper]


def _get_readable_doc(cik: int, acc_nodash: str, acc_dashed: str) -> str:
    """Find the human-readable .htm document from the filing index.

    The primaryDocument field in submissions.json often points to an XBRL
    instance document (full of schema URLs, not text). We hit the filing
    index JSON to find the actual 10-K report document.

    Priority order:
        1. Any .htm/.html file with "10-k" or the accession number in its name
        2. The largest .htm/.html file (usually the main report)
        3. Fall back to primaryDocument
    """
    index_url = FILING_INDEX_URL.format(
        cik=cik, acc_nodash=acc_nodash, acc_dashed=acc_dashed
    )
    time.sleep(0.11)
    resp = requests.get(index_url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        return None  # caller will fall back to primaryDocument

    data = resp.json()
    files = data.get("directory", {}).get("item", [])

    # Filter to .htm/.html files only
    htm_files = [f for f in files if f["name"].lower().endswith((".htm", ".html"))]
    if not htm_files:
        return None

    # Prefer files that look like the main report (contain ticker or "10k" in name)
    def score(f):
        name = f["name"].lower()
        s = 0
        if "10-k" in name or "10k" in name:
            s += 10
        if "xbrl" in name or "r1." in name or "r2." in name:
            s -= 5  # avoid XBRL viewer fragments
        s += int(f.get("size", 0)) // 100000  # larger = more likely main doc
        return s

    htm_files.sort(key=score, reverse=True)
    return htm_files[0]["name"]


def get_latest_10k_info(cik: int) -> dict:
    """Fetch metadata for the most recent 10-K filing.

    Args:
        cik: SEC CIK number (integer)

    Returns:
        dict with keys: accession, date, doc, url
    """
    cik_padded = str(cik).zfill(10)
    url = SUBMISSIONS_URL.format(cik_padded=cik_padded)

    time.sleep(0.11)  # stay under 10 req/s
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    recent = data["filings"]["recent"]
    forms = recent["form"]
    accessions = recent["accessionNumber"]
    dates = recent["filingDate"]
    docs = recent["primaryDocument"]

    for form, acc, dt, primary_doc in zip(forms, accessions, dates, docs):
        if form == "10-K":
            acc_nodash = acc.replace("-", "")

            # Try to find the readable report doc from the filing index
            readable_doc = _get_readable_doc(cik, acc_nodash, acc)
            doc = readable_doc or primary_doc

            filing_url = FILING_URL.format(cik=cik, acc_nodash=acc_nodash, doc=doc)
            return {
                "accession": acc,
                "acc_nodash": acc_nodash,
                "date": dt,
                "doc": doc,
                "url": filing_url,
            }

    raise ValueError(f"No 10-K filing found for CIK {cik}")


def _parse_html_to_text(html_bytes: bytes) -> str:
    """Extract plain text from 10-K HTML using BeautifulSoup.

    Modern SEC filings use iXBRL — XBRL data is embedded inline inside an
    <ix:header> block at the top of the HTML. We remove it before extracting
    text, otherwise the first ~20K chars are XBRL schema URLs, not prose.
    """
    import re

    soup = BeautifulSoup(html_bytes, "html.parser")

    # Remove XBRL inline header (iXBRL) — this is the structured data block,
    # not human-readable text
    for tag in soup.find_all(["ix:header", "ix:hidden"]):
        tag.decompose()

    # Remove other noise tags
    for tag in soup(["script", "style", "head", "meta"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)

    # Collapse repeated whitespace
    text = re.sub(r" {3,}", "  ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text


def fetch_latest_10k(ticker: str) -> tuple[str, dict]:
    """Download and parse the latest 10-K for a given ticker.

    This is the main entry point for live mode. Makes 3 API calls:
        1. GET company_tickers_exchange.json (cached after first call)
        2. GET submissions/CIK{cik}.json
        3. GET Archives/.../primarydocument.htm

    Args:
        ticker: stock ticker, e.g. "AAPL"

    Returns:
        (text, metadata) where:
            text: full plain text of the 10-K (~100-300K chars)
            metadata: {"ticker", "cik", "date", "url", "char_count"}
    """
    print(f"[EDGAR] Looking up ticker: {ticker}")
    cik = ticker_to_cik(ticker)

    print(f"[EDGAR] CIK={cik}, fetching 10-K info...")
    filing_info = get_latest_10k_info(cik)
    print(f"[EDGAR] Found 10-K filed {filing_info['date']}: {filing_info['url']}")

    time.sleep(0.11)
    print(f"[EDGAR] Downloading filing...")
    resp = requests.get(filing_info["url"], headers=HEADERS, timeout=60)
    resp.raise_for_status()

    print(f"[EDGAR] Parsing HTML ({len(resp.content):,} bytes)...")
    text = _parse_html_to_text(resp.content)

    metadata = {
        "ticker": ticker.upper(),
        "cik": cik,
        "date": filing_info["date"],
        "url": filing_info["url"],
        "accession": filing_info["accession"],
        "char_count": len(text),
    }

    print(f"[EDGAR] Done. {len(text):,} chars extracted.")
    return text, metadata
