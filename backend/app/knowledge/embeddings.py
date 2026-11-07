"""
Embedding Generation Service for IncidentX Knowledge Base — Stage 3.

Wraps sentence-transformers/all-MiniLM-L6-v2 (or any configured model) with:
  - Singleton model caching (no reload per request).
  - Batch embedding for document chunks.
  - Dedicated query embedding method.
  - Validation: dimension check, NaN/Inf guard, empty-input guard.

This module deliberately does NOT touch FAISS or any retrieval logic.
"""

import logging
import math
from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.schemas.knowledge import ChunkEmbedding, DocumentChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model caching — the SentenceTransformer is loaded once per process.
# lru_cache on the factory function keyed on model_name guarantees that
# a second call with the same name returns the already-loaded instance.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _load_model(model_name: str) -> SentenceTransformer:
    """Load and cache a SentenceTransformer model by name.

    The @lru_cache decorator ensures the model is instantiated only once
    per unique model_name within a process lifetime.
    """
    logger.info("Loading embedding model '%s' (this happens once).", model_name)
    model = SentenceTransformer(model_name)
    logger.info("Embedding model '%s' loaded successfully.", model_name)
    return model


class EmbeddingService:
    """Stateless service that generates text embeddings using sentence-transformers.

    Usage
    -----
    >>> svc = EmbeddingService()
    >>> vector = svc.embed_text("Database CPU spiked to 100%")
    >>> len(vector)
    384
    """

    def __init__(self, model_name: str | None = None):
        settings = get_settings()
        self.model_name: str = model_name or settings.embedding_model
        self.expected_dimension: int = settings.embedding_dimension
        self.batch_size: int = settings.embedding_batch_size
        # Trigger model load eagerly so the first real request is fast.
        self._model = _load_model(self.model_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_vector(self, vector: np.ndarray, context: str = "") -> None:
        """Raise ValueError if the vector is malformed."""
        if vector.ndim != 1:
            raise ValueError(f"Expected 1-D embedding, got shape {vector.shape}. {context}")
        if vector.shape[0] != self.expected_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.expected_dimension}, "
                f"got {vector.shape[0]}. {context}"
            )
        if np.isnan(vector).any():
            raise ValueError(f"Embedding contains NaN values. {context}")
        if np.isinf(vector).any():
            raise ValueError(f"Embedding contains infinite values. {context}")

    @staticmethod
    def _require_non_empty_text(text: str, label: str = "text") -> str:
        """Strip and reject blank strings early to avoid silent zero-vectors."""
        stripped = text.strip()
        if not stripped:
            raise ValueError(f"Cannot embed empty {label}.")
        return stripped

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string and return a validated float list.

        Parameters
        ----------
        text : str
            Any non-empty string.

        Returns
        -------
        List[float]
            384-dimensional (or model-specific) unit vector.
        """
        clean = self._require_non_empty_text(text)
        try:
            vector: np.ndarray = self._model.encode(
                clean,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            vector = np.asarray(vector, dtype=np.float32)
        except Exception as exc:
            logger.error("Model inference failed for single text: %s", exc)
            raise RuntimeError(f"Embedding inference failed: {exc}") from exc

        self._validate_vector(vector, context="embed_text")
        return vector.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a natural-language query using the same model as documents.

        Uses the same encode path as embed_text so query and document
        vectors are always comparable in the same vector space.
        """
        clean = self._require_non_empty_text(query, label="query")
        try:
            vector: np.ndarray = self._model.encode(
                clean,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            vector = np.asarray(vector, dtype=np.float32)
        except Exception as exc:
            logger.error("Model inference failed for query: %s", exc)
            raise RuntimeError(f"Query embedding inference failed: {exc}") from exc

        self._validate_vector(vector, context="embed_query")
        return vector.tolist()

    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[ChunkEmbedding]:
        """Batch-embed a list of DocumentChunks and return ChunkEmbedding objects.

        Processes chunks in batches of `self.batch_size` to control memory
        usage. Metadata (chunk_id, document_id, chunk_index) is preserved
        verbatim from each DocumentChunk.

        Parameters
        ----------
        chunks : List[DocumentChunk]
            Chunks produced by Stage 2 IntelligentChunker.

        Returns
        -------
        List[ChunkEmbedding]
            One ChunkEmbedding per input chunk, in the same order.
        """
        if not chunks:
            return []

        # Filter out chunks with blank text before sending to the model.
        valid_chunks = [c for c in chunks if c.text.strip()]
        if not valid_chunks:
            raise ValueError("All provided chunks have empty text — nothing to embed.")

        texts = [c.text.strip() for c in valid_chunks]

        logger.info(
            "Generating embeddings for %d chunks (batch_size=%d, model=%s).",
            len(texts),
            self.batch_size,
            self.model_name,
        )

        try:
            # encode returns (N, D) ndarray
            vectors: np.ndarray = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            vectors = np.asarray(vectors, dtype=np.float32)
        except Exception as exc:
            logger.error("Batch embedding inference failed: %s", exc)
            raise RuntimeError(f"Batch embedding inference failed: {exc}") from exc

        result: List[ChunkEmbedding] = []
        for i, (chunk, vector) in enumerate(zip(valid_chunks, vectors)):
            self._validate_vector(vector, context=f"chunk {chunk.chunk_id}")
            result.append(
                ChunkEmbedding(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    embedding=vector.tolist(),
                    dimension=int(vector.shape[0]),
                    embedding_model=self.model_name,
                )
            )

        logger.info("Successfully generated %d embeddings.", len(result))
        return result

    def compute_vector_norm(self, vector: List[float]) -> float:
        """Return the L2 norm of an embedding vector."""
        arr = np.asarray(vector, dtype=np.float64)
        return float(math.sqrt(float(np.dot(arr, arr))))


# ---------------------------------------------------------------------------
# Module-level singleton — imported by the service and API layers.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Return the shared EmbeddingService singleton (cached)."""
    return EmbeddingService()

