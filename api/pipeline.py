"""FilingSensePipeline — loads indexes and model, serves answer requests.

Startup (one-time, ~2-3 min on HF Spaces persistent):
    1. Download chunks from HuggingFace Datasets
    2. Build BM25 + FAISS index per ticker (30 indexes, all in memory)
    3. Load GRPO model

Per request (~6-12 sec):
    1. Look up ticker's index
    2. Hybrid retrieval (BM25 + FAISS + reranker)
    3. Generate answer with GRPO model
    4. Parse reasoning / answer from output
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from api.config import (
    MODEL_PATH,
    HF_CHUNKS_REPO,
    RETRIEVAL_TOP_K,
    USE_RERANKER,
    TICKERS,
)
from api.models import AnswerResponse, SourceChunk
from src.chunking import Chunk
from src.indexing import build_index, DualIndex
from src.retrieval import retrieve, load_reranker
from src.generation import generate_answer


@dataclass
class TickerIndex:
    index: DualIndex
    filing_date: str
    filing_url: str


def _parse_reasoning_and_answer(raw: str) -> tuple[str, str]:
    """Split model output into (reasoning, answer).

    The GRPO model is trained to output:
        Reasoning: <step-by-step>
        Answer: <final answer>

    Falls back gracefully if the format isn't followed.
    """
    reasoning_match = re.search(r"Reasoning:\s*(.*?)(?=Answer:|$)", raw, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r"Answer:\s*(.*?)$", raw, re.DOTALL | re.IGNORECASE)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    answer = answer_match.group(1).strip() if answer_match else raw.strip()

    return reasoning, answer


class FilingSensePipeline:
    """Manages indexes and model for all 30 tickers.

    Designed to be instantiated once at server startup and reused
    across all requests.
    """

    def __init__(self):
        self._indexes: dict[str, TickerIndex] = {}
        self._model = None
        self._tokenizer = None
        self._reranker = None
        self._model_loaded = False

    def load(self):
        """Full startup sequence — call once before serving requests."""
        self._load_indexes()
        self._load_model()

    def _load_indexes(self):
        """Download chunks from HF Datasets and build indexes for all tickers."""
        print(f"[Pipeline] Downloading chunks from {HF_CHUNKS_REPO}...")
        from datasets import load_dataset
        ds = load_dataset(HF_CHUNKS_REPO, data_files="chunks.jsonl", split="train")

        # Group records by ticker
        ticker_records: dict[str, list[dict]] = {}
        for record in ds:
            t = record["ticker"]
            ticker_records.setdefault(t, []).append(record)

        # Load embedding model once, reuse across all tickers
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Pipeline] Loading embedding model (device={device})...")
        embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

        print(f"[Pipeline] Building indexes for {len(ticker_records)} tickers...")
        for ticker, records in ticker_records.items():
            chunks = [
                Chunk(
                    chunk_id=r["chunk_id"],
                    text=r["text"],
                    pre_text=r["text"],
                    table_text="",
                    post_text="",
                    metadata={"ticker": r["ticker"], "date": r["date"]},
                )
                for r in records
            ]
            index = build_index(chunks, embed_model=embed_model, show_progress=False)
            self._indexes[ticker] = TickerIndex(
                index=index,
                filing_date=records[0]["date"],
                filing_url=records[0]["url"],
            )
            print(f"  [{ticker}] {len(chunks)} chunks indexed")

        print(f"[Pipeline] All indexes ready.")

    def _load_model(self):
        """Load the GRPO model and tokenizer."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[Pipeline] Loading model: {MODEL_PATH}")
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self._model.eval()

        if USE_RERANKER:
            print("[Pipeline] Loading reranker...")
            self._reranker = load_reranker()

        self._model_loaded = True
        print("[Pipeline] Model ready.")

    @property
    def tickers_loaded(self) -> int:
        return len(self._indexes)

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    def answer(self, ticker: str, question: str) -> AnswerResponse:
        """Run the full RAG pipeline for a question about a ticker.

        Args:
            ticker: uppercase ticker, e.g. "AAPL"
            question: natural language question

        Returns:
            AnswerResponse with answer, reasoning, and source chunks
        """
        ticker = ticker.upper()
        if ticker not in self._indexes:
            raise ValueError(f"Ticker '{ticker}' not supported. Supported: {sorted(self._indexes.keys())}")

        ticker_index = self._indexes[ticker]

        results = retrieve(
            ticker_index.index,
            question,
            top_k_per_source=20,
            top_n_final=RETRIEVAL_TOP_K,
            reranker=self._reranker,
            use_reranker=self._reranker is not None,
        )

        gen_result = generate_answer(
            query=question,
            results=results,
            model=self._model,
            tokenizer=self._tokenizer,
            model_name=MODEL_PATH.split("/")[-1],
        )

        reasoning, answer = _parse_reasoning_and_answer(gen_result.answer)

        sources = [
            SourceChunk(
                chunk_id=r.chunk.chunk_id,
                text=r.chunk.text[:400],
                score=round(r.score, 4),
            )
            for r in results
        ]

        return AnswerResponse(
            ticker=ticker,
            question=question,
            reasoning=reasoning,
            answer=answer,
            sources=sources,
            filing_date=ticker_index.filing_date,
            filing_url=ticker_index.filing_url,
        )
