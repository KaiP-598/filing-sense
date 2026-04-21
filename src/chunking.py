"""Section-aware chunking for SEC 10-K filings.

FinQA examples are already structured as (pre_text, table, post_text) triples.
Each triple is one "chunk" — a self-contained context window from a 10-K filing.

For the EDGAR live mode (Day 13), we'll add raw 10-K parsing here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single retrievable unit from a 10-K filing."""
    chunk_id: str
    text: str            # flattened text for embedding/BM25
    pre_text: str        # paragraphs before the table
    table_text: str      # table rendered as text
    post_text: str       # paragraphs after the table
    metadata: dict = field(default_factory=dict)  # company, year, etc.


def _safe_parse(raw) -> list[str]:
    """Parse a stringified list, or return as-is if already a list."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        return [raw] if raw else []
    return [str(raw)] if raw else []


def _table_to_text(raw_table) -> str:
    """Convert a FinQA table to readable text. Handles strings and lists."""
    if not raw_table:
        return ""
    # wandb mirror stores tables as strings — just use them directly
    if isinstance(raw_table, str):
        return raw_table
    if isinstance(raw_table, list):
        lines = []
        for row in raw_table:
            if isinstance(row, list):
                lines.append(" | ".join(str(cell) for cell in row))
            else:
                lines.append(str(row))
        return "\n".join(lines)
    return str(raw_table)


def chunk_finqa_example(example: dict, idx: int = 0) -> Chunk:
    """Convert one FinQA example into a Chunk.

    Each FinQA example is already a natural chunk: the context surrounding
    one financial table in a 10-K filing.
    """
    pre_parts = _safe_parse(example.get("pre_text", ""))
    post_parts = _safe_parse(example.get("post_text", ""))
    table_text = _table_to_text(example.get("table", ""))

    pre_text = " ".join(pre_parts)
    post_text = " ".join(post_parts)

    # Full text for embedding: pre + table + post
    full_text = f"{pre_text}\n\n{table_text}\n\n{post_text}".strip()

    # Deterministic ID from content
    chunk_id = example.get("id", hashlib.md5(full_text.encode()).hexdigest()[:12])

    return Chunk(
        chunk_id=str(chunk_id) if chunk_id else f"chunk_{idx}",
        text=full_text,
        pre_text=pre_text,
        table_text=table_text,
        post_text=post_text,
        metadata={
            "query": example.get("query", ""),
            "answer": example.get("exe_ans", ""),
            "program": example.get("program", ""),
        },
    )


def chunk_dataset(dataset) -> list[Chunk]:
    """Convert an entire FinQA split into chunks."""
    chunks = []
    for i, ex in enumerate(dataset):
        chunks.append(chunk_finqa_example(ex, idx=i))
    return chunks


def chunk_raw_text(
    text: str,
    ticker: str = "unknown",
    date: str = "",
    chunk_size: int = 2000,
    overlap: int = 200,
) -> list[Chunk]:
    """Split raw 10-K text into overlapping chunks for indexing.

    Unlike FinQA (which has pre-structured pre/table/post), raw EDGAR text
    is one continuous string. We use a sliding window:
        - chunk_size: ~2000 chars per chunk (~500 tokens)
        - overlap: 200 chars shared between adjacent chunks, so a table
          that straddles a boundary appears in both chunks

    We split on whitespace boundaries (never mid-word) to keep text readable.

    Args:
        text: full plain text of the 10-K
        ticker: company ticker for metadata
        date: filing date for metadata
        chunk_size: target size per chunk in characters
        overlap: chars to repeat between adjacent chunks

    Returns:
        list of Chunk objects ready for build_index()
    """
    words = text.split()
    chunks: list[Chunk] = []

    # Build char-indexed word boundaries for clean splits
    # Instead of splitting on exact char count, accumulate words until we hit chunk_size
    current_words: list[str] = []
    current_len = 0
    chunk_idx = 0
    word_idx = 0

    while word_idx < len(words):
        word = words[word_idx]
        current_words.append(word)
        current_len += len(word) + 1  # +1 for space

        if current_len >= chunk_size:
            chunk_text = " ".join(current_words)
            chunk_id = f"{ticker}_{date}_{chunk_idx}"

            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                pre_text=chunk_text,
                table_text="",
                post_text="",
                metadata={"ticker": ticker, "date": date, "chunk_idx": chunk_idx},
            ))

            chunk_idx += 1

            # Move back `overlap` chars worth of words for the next chunk
            overlap_words = []
            overlap_len = 0
            for w in reversed(current_words):
                overlap_len += len(w) + 1
                if overlap_len > overlap:
                    break
                overlap_words.insert(0, w)

            current_words = overlap_words
            current_len = sum(len(w) + 1 for w in current_words)

        word_idx += 1

    # Last partial chunk
    if current_words:
        chunk_text = " ".join(current_words)
        chunks.append(Chunk(
            chunk_id=f"{ticker}_{date}_{chunk_idx}",
            text=chunk_text,
            pre_text=chunk_text,
            table_text="",
            post_text="",
            metadata={"ticker": ticker, "date": date, "chunk_idx": chunk_idx},
        ))

    return chunks
