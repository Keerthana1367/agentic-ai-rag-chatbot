"""Vector store — FAISS index with JSON metadata for persistence.

FAISS runs fully in-process with no external services (no Docker, no cloud).
The index and metadata are saved to disk so data survives process restarts.
"""

import json
import logging
from typing import Any

import faiss
import numpy as np

from src.config import (
    EMBEDDING_DIMENSION,
    FAISS_DIR,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
)

logger = logging.getLogger(__name__)


def load_index() -> tuple[faiss.IndexFlatIP, list[dict[str, Any]]]:
    """Load the FAISS index and chunk metadata from disk.

    If no saved index exists, returns an empty index and empty metadata list.

    Returns:
        A tuple of (faiss_index, metadata_list).
    """
    if FAISS_INDEX_PATH.exists() and FAISS_METADATA_PATH.exists():
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(FAISS_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        logger.info("Loaded FAISS index with %d vectors", index.ntotal)
        return index, metadata

    # Inner-product index on normalised vectors = cosine similarity.
    # We normalise before inserting, so IP scores are in [0, 1].
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    logger.info("Created new FAISS index (dim=%d)", EMBEDDING_DIMENSION)
    return index, []


def save_index(index: faiss.IndexFlatIP, metadata: list[dict[str, Any]]) -> None:
    """Persist the FAISS index and chunk metadata to disk.

    Args:
        index: The FAISS index to save.
        metadata: List of per-vector metadata dicts.
    """
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(FAISS_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
    logger.info("Saved FAISS index (%d vectors) to %s", index.ntotal, FAISS_DIR)


def add_chunks(
    index: faiss.IndexFlatIP,
    metadata: list[dict[str, Any]],
    embeddings: np.ndarray,
    chunks: list[dict[str, Any]],
) -> None:
    """Add chunk embeddings and metadata to the index.

    Vectors are L2-normalised before insertion so that inner-product
    scores equal cosine similarity.

    Args:
        index: FAISS index to add vectors to.
        metadata: Existing metadata list (modified in place).
        embeddings: Numpy array of shape (n, dim).
        chunks: List of chunk dicts with keys: chunk_id, text, page, source.
    """
    # Normalise so inner product = cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    for chunk in chunks:
        metadata.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "page": chunk["page"],
            "source": chunk["source"],
        })

    logger.info("Added %d vectors to index (total: %d)", len(chunks), index.ntotal)


def query_chunks(
    index: faiss.IndexFlatIP,
    metadata: list[dict[str, Any]],
    query_embedding: np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    """Run a nearest-neighbor search and return the top-k chunks.

    Args:
        index: FAISS index to search.
        metadata: Chunk metadata list (aligned by position with index vectors).
        query_embedding: The embedded query as a 1-D numpy array.
        top_k: Number of results to return.

    Returns:
        List of dicts, each with keys: text, page, score (0-1 similarity).
    """
    # Reshape to (1, dim) and normalise for cosine similarity
    query_vec = query_embedding.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query_vec)

    # FAISS returns (distances, indices) arrays of shape (1, top_k)
    scores, indices = index.search(query_vec, top_k)

    results: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            # FAISS returns -1 when fewer than top_k results exist
            continue
        meta = metadata[idx]
        results.append({
            "text": meta["text"],
            "page": meta["page"],
            "score": round(float(score), 4),
        })

    return results
