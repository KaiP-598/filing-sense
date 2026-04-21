"""Live mode: answer questions about any company's latest 10-K filing.

Fetches the filing from EDGAR, chunks it, builds an in-memory index,
and uses the GRPO model to answer. No pre-built index needed.

Usage:
    from src.live import LivePipeline

    pipeline = LivePipeline(model_path="kaiwu598/filing-sense-grpo-qwen2.5-3b")
    answer = pipeline.answer("AAPL", "What was the revenue growth rate in 2024?")
    print(answer)

Or with the agent (query decomposition + grading):
    answer = pipeline.answer("AAPL", "...", use_agent=True, openai_api_key="sk-...")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from src.edgar import fetch_latest_10k
from src.chunking import chunk_raw_text
from src.indexing import build_index, DualIndex
from src.retrieval import retrieve, load_reranker
from src.generation import generate_answer, generate_answer_api, GenerationResult


@dataclass
class LiveResult:
    """Result from a live mode query."""
    answer: str
    ticker: str
    query: str
    filing_date: str
    filing_url: str
    num_chunks_retrieved: int
    generation_result: GenerationResult | None = None
    chunks_preview: list[str] = field(default_factory=list)  # first 200 chars of each chunk used


class LivePipeline:
    """End-to-end pipeline for answering questions about live SEC filings.

    Loads the model once, caches the EDGAR index per ticker so repeated
    questions about the same company don't re-download the filing.

    Args:
        model_path: HuggingFace model ID or local path
        use_reranker: whether to apply cross-encoder reranking
        top_k: number of chunks to retrieve
    """

    def __init__(
        self,
        model_path: str = "kaiwu598/filing-sense-grpo-qwen2.5-3b",
        use_reranker: bool = True,
        top_k: int = 5,
    ):
        self.model_path = model_path
        self.top_k = top_k
        self._index_cache: dict[str, tuple[DualIndex, dict]] = {}  # ticker -> (index, metadata)

        print(f"Loading model: {model_path}")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()
        print("Model loaded.")

        self.reranker = load_reranker() if use_reranker else None

    def _get_index(self, ticker: str) -> tuple[DualIndex, dict]:
        """Return (index, metadata) for ticker, downloading+indexing if needed."""
        key = ticker.upper()
        if key not in self._index_cache:
            text, metadata = fetch_latest_10k(ticker)
            chunks = chunk_raw_text(
                text,
                ticker=metadata["ticker"],
                date=metadata["date"],
            )
            print(f"[Live] {len(chunks)} chunks from {metadata['date']} 10-K")
            index = build_index(chunks, show_progress=True)
            self._index_cache[key] = (index, metadata)
        return self._index_cache[key]

    def answer(
        self,
        ticker: str,
        query: str,
        use_agent: bool = False,
        openai_api_key: str | None = None,
    ) -> LiveResult:
        """Answer a question about a company's latest 10-K.

        Args:
            ticker: stock ticker, e.g. "AAPL"
            query: natural language question
            use_agent: if True, use LangGraph agent with query decomposition
            openai_api_key: required when use_agent=True

        Returns:
            LiveResult with the answer and provenance metadata
        """
        index, metadata = self._get_index(ticker)

        if use_agent:
            api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
            from src.agent import build_agent
            agent = build_agent(
                index=index,
                reranker=self.reranker,
                model=self.model,
                tokenizer=self.tokenizer,
                openai_api_key=api_key,
                model_name=self.model_path.split("/")[-1],
            )
            state = agent.invoke({"query": query})
            raw_answer = state.get("answer", "Insufficient context.")
            chunks_used = state.get("relevant_chunks", []) or state.get("retrieved_chunks", [])
            gen_result = state.get("generation_result")
            num_retrieved = len(state.get("retrieved_chunks", []))
        else:
            results = retrieve(
                index,
                query,
                top_k_per_source=20,
                top_n_final=self.top_k,
                reranker=self.reranker,
                use_reranker=self.reranker is not None,
            )
            gen_result = generate_answer(
                query=query,
                results=results,
                model=self.model,
                tokenizer=self.tokenizer,
                model_name=self.model_path.split("/")[-1],
            )
            raw_answer = gen_result.answer
            chunks_used = results
            num_retrieved = len(results)

        chunks_preview = [r.chunk.text[:200] for r in chunks_used[:5]]

        return LiveResult(
            answer=raw_answer,
            ticker=ticker.upper(),
            query=query,
            filing_date=metadata["date"],
            filing_url=metadata["url"],
            num_chunks_retrieved=num_retrieved,
            generation_result=gen_result,
            chunks_preview=chunks_preview,
        )
