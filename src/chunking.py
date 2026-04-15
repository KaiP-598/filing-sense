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
