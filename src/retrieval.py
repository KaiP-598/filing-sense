"""Hybrid retrieval: BM25 + FAISS + reciprocal rank fusion + cross-encoder reranker.

Pipeline:
    query → BM25 top-K → }
    query → FAISS top-K → } → RRF merge → cross-encoder rerank → top-N final
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import CrossEncoder

from src.chunking import Chunk
from src.indexing import DualIndex, _tokenize_simple


DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RetrievalResult:
    """A single retrieved chunk with its score."""
    chunk: Chunk
    score: float
    rank: int


def _search_faiss(index: DualIndex, query: str, top_k: int = 20) -> list[tuple[int, float]]:
    """Dense retrieval via FAISS. Returns (chunk_idx, score) pairs."""
    query_emb = index.embed_model.encode(
        [query], normalize_embeddings=True
    ).astype(np.float32)
    scores, indices = index.faiss_index.search(query_emb, top_k)
    return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]


def _search_bm25(index: DualIndex, query: str, top_k: int = 20) -> list[tuple[int, float]]:
    """Sparse retrieval via BM25. Returns (chunk_idx, score) pairs."""
    tokens = _tokenize_simple(query)
    scores = index.bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


def _reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF score for document d = sum over rankings of 1 / (k + rank_in_ranking).
    k=60 is the standard default from the original RRF paper.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _) in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items


def retrieve(
    index: DualIndex,
    query: str,
    top_k_per_source: int = 20,
    top_n_final: int = 5,
    reranker: CrossEncoder | None = None,
    use_reranker: bool = True,
) -> list[RetrievalResult]:
    """Full hybrid retrieval pipeline.

    1. BM25 top-K
    2. FAISS top-K
    3. RRF merge
    4. Cross-encoder rerank (optional)
    5. Return top-N
    """
    # Step 1-2: parallel retrieval
    bm25_results = _search_bm25(index, query, top_k=top_k_per_source)
    faiss_results = _search_faiss(index, query, top_k=top_k_per_source)

    # Step 3: RRF merge
    fused = _reciprocal_rank_fusion([bm25_results, faiss_results])

    # Take top candidates for reranking
    candidates = fused[:top_k_per_source]

    # Step 4: cross-encoder rerank
    if use_reranker and reranker is not None and candidates:
        pairs = [(query, index.chunks[idx].text) for idx, _ in candidates]
        rerank_scores = reranker.predict(pairs)
        # Re-sort by reranker score
        scored = list(zip([idx for idx, _ in candidates], rerank_scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        candidates = [(idx, float(score)) for idx, score in scored]

    # Step 5: top-N results
    results = []
    for rank, (idx, score) in enumerate(candidates[:top_n_final]):
        results.append(RetrievalResult(
            chunk=index.chunks[idx],
            score=score,
            rank=rank,
        ))

    return results


def load_reranker(model_name: str = DEFAULT_RERANKER) -> CrossEncoder:
    """Load the cross-encoder reranker model."""
    print(f"Loading reranker: {model_name}")
    return CrossEncoder(model_name)
