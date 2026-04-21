"""Build dense (FAISS) and sparse (BM25) indexes over chunks.

Dense: sentence-transformers embeddings → FAISS flat index
Sparse: BM25Okapi over tokenized chunk text
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.chunking import Chunk


# Small but effective model — free, 384-dim embeddings
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def _default_device() -> str:
    """Use GPU if available, else CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class DualIndex:
    """Holds both dense and sparse indexes plus the chunk store."""
    chunks: list[Chunk]
    faiss_index: faiss.IndexFlatIP
    bm25: BM25Okapi
    embed_model: SentenceTransformer


def _tokenize_simple(text: str) -> list[str]:
    """Whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


def build_index(
    chunks: list[Chunk],
    embed_model_name: str = DEFAULT_EMBED_MODEL,
    embed_model: SentenceTransformer | None = None,
    batch_size: int = 64,
    show_progress: bool = True,
) -> DualIndex:
    """Build both FAISS and BM25 indexes from chunks.

    Args:
        chunks: list of Chunk objects to index
        embed_model_name: sentence-transformer model ID (used if embed_model not provided)
        embed_model: pre-loaded SentenceTransformer (pass to avoid reloading across tickers)
        batch_size: encoding batch size
        show_progress: show progress bar during encoding
    """
    texts = [c.text for c in chunks]

    # --- Dense index (FAISS) ---
    if embed_model is None:
        print(f"Loading embedding model: {embed_model_name}")
        device = _default_device()
        embed_model = SentenceTransformer(embed_model_name, device=device)
        print(f"Using device: {device}")

    print(f"Encoding {len(texts)} chunks...")
    embeddings = embed_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # for cosine similarity via inner product
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)  # inner product = cosine (normalized)
    faiss_index.add(embeddings)
    print(f"FAISS index: {faiss_index.ntotal} vectors, {dim} dims")

    # --- Sparse index (BM25) ---
    tokenized = [_tokenize_simple(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    print(f"BM25 index: {len(tokenized)} documents")

    return DualIndex(
        chunks=chunks,
        faiss_index=faiss_index,
        bm25=bm25,
        embed_model=embed_model,
    )
