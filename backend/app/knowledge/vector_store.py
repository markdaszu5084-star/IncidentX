"""
FAISS Vector Database Integration for IncidentX Knowledge Base — Stage 4.

Provides vector storage, metadata association, and safe local persistence for
chunk embeddings generated in Stage 3 using FAISS IndexIDMap2 and IndexFlatIP.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np

from app.config import get_settings
from app.knowledge.embeddings import EmbeddingService, get_embedding_service
from app.knowledge.exceptions import (
    CorruptedIndexException,
    DimensionMismatchException,
    DocumentNotFoundException,
    DuplicateDocumentException,
    VectorStoreException,
)
from app.schemas.knowledge import (
    ChunkEmbedding,
    DocumentChunk,
    DocumentIndexInfo,
    IndexDocumentResponse,
    VectorMetadata,
    VectorStoreActionResponse,
    VectorStoreStatsResponse,
    VectorStoreStatusResponse,
)

logger = logging.getLogger(__name__)


class FaissVectorStore:
    """Manages an in-memory and disk-persisted FAISS vector index with rich chunk metadata."""

    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
        dimension: Optional[int] = None,
        embedding_service: Optional[EmbeddingService] = None,
        auto_load: bool = True,
    ):
        settings = get_settings()
        self.embedding_service = embedding_service or get_embedding_service()
        self.dimension: int = dimension or self.embedding_service.expected_dimension
        self.embedding_model: str = self.embedding_service.model_name

        # Resolve storage directory
        if storage_dir is not None:
            self.index_dir = Path(storage_dir).resolve()
        else:
            self.index_dir = Path(settings.faiss_index_dir).resolve()

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "index.faiss"
        self.metadata_path = self.index_dir / "metadata.json"

        # Internal state
        self._index: Optional[faiss.IndexIDMap2] = None
        self._next_vector_id: int = 1
        self._vector_metadata: Dict[int, VectorMetadata] = {}
        self._indexed_documents: Dict[str, DocumentIndexInfo] = {}
        self._created_at: str = datetime.now(timezone.utc).isoformat()
        self._updated_at: str = self._created_at

        # Initialize or reload
        if auto_load and self.is_persisted():
            try:
                self.load_index()
            except Exception as e:
                logger.warning(
                    "Failed to load existing FAISS index from '%s': %s. Re-initializing empty index.",
                    self.index_dir,
                    e,
                )
                self._init_empty_index()
        else:
            self._init_empty_index()

    def _init_empty_index(self) -> None:
        """Create a new empty IndexIDMap2 wrapping IndexFlatIP."""
        sub_index = faiss.IndexFlatIP(self.dimension)
        self._index = faiss.IndexIDMap2(sub_index)
        self._next_vector_id = 1
        self._vector_metadata = {}
        self._indexed_documents = {}
        self._created_at = datetime.now(timezone.utc).isoformat()
        self._updated_at = self._created_at
        logger.info(
            "Initialized empty FAISS IndexIDMap2(IndexFlatIP(%d)) for model '%s'.",
            self.dimension,
            self.embedding_model,
        )

    # ------------------------------------------------------------------
    # Properties & Status
    # ------------------------------------------------------------------

    @property
    def total_vectors(self) -> int:
        """Return the count of active vectors currently in the FAISS index."""
        return self._index.ntotal if self._index is not None else 0

    @property
    def total_documents(self) -> int:
        """Return the count of distinct documents currently indexed."""
        return len(self._indexed_documents)

    def is_persisted(self) -> bool:
        """Check whether both index.faiss and metadata.json exist on disk."""
        return self.index_path.exists() and self.metadata_path.exists()

    def is_document_indexed(self, document_id: str) -> bool:
        """Return True if a document is already present in the index."""
        return document_id in self._indexed_documents

    def get_document_info(self, document_id: str) -> Optional[DocumentIndexInfo]:
        """Return index info for a specific document, or None if not indexed."""
        return self._indexed_documents.get(document_id)

    def get_status(self) -> VectorStoreStatusResponse:
        """Return live vector store status."""
        return VectorStoreStatusResponse(
            status="ready",
            initialized=self._index is not None,
            total_vectors=self.total_vectors,
            total_documents=self.total_documents,
            dimension=self.dimension,
            embedding_model=self.embedding_model,
            similarity_metric="inner_product",
            index_file_exists=self.index_path.exists(),
            metadata_file_exists=self.metadata_path.exists(),
        )

    def get_stats(self) -> VectorStoreStatsResponse:
        """Return detailed statistics and inventory of the indexed documents."""
        docs_list = list(self._indexed_documents.values())
        return VectorStoreStatsResponse(
            status="success",
            total_vectors=self.total_vectors,
            total_documents=self.total_documents,
            dimension=self.dimension,
            embedding_model=self.embedding_model,
            similarity_metric="inner_product",
            is_persisted=self.is_persisted(),
            index_file_exists=self.index_path.exists(),
            metadata_file_exists=self.metadata_path.exists(),
            indexed_documents=docs_list,
        )

    # ------------------------------------------------------------------
    # Vector & Document Ingestion
    # ------------------------------------------------------------------

    def add_document_vectors(
        self,
        document_id: str,
        chunks: List[DocumentChunk],
        embeddings: List[ChunkEmbedding],
        overwrite: bool = False,
        auto_save: bool = True,
    ) -> IndexDocumentResponse:
        """Add embedding vectors and chunk metadata for a document into FAISS."""
        if not chunks:
            raise VectorStoreException(f"No chunks provided for document '{document_id}'.")
        if not embeddings:
            raise VectorStoreException(f"No embeddings provided for document '{document_id}'.")

        # Duplicate check
        if self.is_document_indexed(document_id):
            if not overwrite:
                raise DuplicateDocumentException(
                    f"Document '{document_id}' is already indexed in the vector store. Set overwrite=True to re-index."
                )
            logger.info("Overwriting existing vectors for document '%s'.", document_id)
            self.remove_document_vectors(document_id, auto_save=False)

        # Match chunks and embeddings
        emb_map = {e.chunk_id: e for e in embeddings}
        matched_pairs: List[Tuple[DocumentChunk, ChunkEmbedding]] = []
        for chunk in chunks:
            if chunk.chunk_id in emb_map:
                matched_pairs.append((chunk, emb_map[chunk.chunk_id]))
            else:
                logger.warning("Chunk '%s' missing corresponding embedding, skipping.", chunk.chunk_id)

        if not matched_pairs:
            raise VectorStoreException(
                f"No matching chunk-embedding pairs found for document '{document_id}'."
            )

        # Dimension and validity verification
        raw_vectors: List[List[float]] = []
        for chunk, emb in matched_pairs:
            vec = emb.embedding
            if len(vec) != self.dimension:
                raise DimensionMismatchException(
                    f"Vector dimension {len(vec)} does not match index dimension {self.dimension} for chunk '{chunk.chunk_id}'."
                )
            raw_vectors.append(vec)

        vectors_np = np.asarray(raw_vectors, dtype=np.float32)

        # Check for NaN / Inf
        if np.isnan(vectors_np).any():
            raise VectorStoreException("Embedding vectors contain NaN values.")
        if np.isinf(vectors_np).any():
            raise VectorStoreException("Embedding vectors contain infinite values.")

        # Ensure unit normalization for inner product == cosine similarity
        norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
        # Avoid zero division
        norms[norms == 0] = 1.0
        vectors_np = vectors_np / norms

        # Allocate integer IDs
        assigned_ids: List[int] = []
        new_metadata_entries: Dict[int, VectorMetadata] = {}

        for chunk, emb in matched_pairs:
            vid = self._next_vector_id
            self._next_vector_id += 1
            assigned_ids.append(vid)

            meta = VectorMetadata(
                vector_id=vid,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                text=chunk.text,
                character_count=chunk.character_count,
                word_count=chunk.word_count,
                metadata=chunk.metadata,
            )
            new_metadata_entries[vid] = meta

        # Add to FAISS index
        ids_np = np.asarray(assigned_ids, dtype=np.int64)
        try:
            self._index.add_with_ids(vectors_np, ids_np)
        except Exception as e:
            logger.error("Failed to add vectors to FAISS index: %s", e)
            raise VectorStoreException(f"FAISS add_with_ids failure: {e}") from e

        # Update in-memory metadata
        self._vector_metadata.update(new_metadata_entries)
        now_iso = datetime.now(timezone.utc).isoformat()
        first_chunk = matched_pairs[0][0]
        self._indexed_documents[document_id] = DocumentIndexInfo(
            document_id=document_id,
            filename=first_chunk.filename,
            total_chunks=len(assigned_ids),
            vector_ids=assigned_ids,
            indexed_at=now_iso,
        )
        self._updated_at = now_iso

        logger.info(
            "Successfully indexed %d vectors for document '%s' (total vectors: %d).",
            len(assigned_ids),
            document_id,
            self.total_vectors,
        )

        if auto_save:
            self.save_index()

        return IndexDocumentResponse(
            status="success",
            message=f"Document '{document_id}' successfully indexed with {len(assigned_ids)} vectors.",
            document_id=document_id,
            vectors_added=len(assigned_ids),
            total_vectors_in_index=self.total_vectors,
        )

    # ------------------------------------------------------------------
    # Vector & Document Removal
    # ------------------------------------------------------------------

    def remove_document_vectors(
        self,
        document_id: str,
        auto_save: bool = True,
    ) -> int:
        """Remove all vectors and metadata associated with a document from the FAISS index."""
        if not self.is_document_indexed(document_id):
            raise DocumentNotFoundException(
                f"Document '{document_id}' is not present in the vector store.",
                details={"document_id": document_id},
            )

        doc_info = self._indexed_documents[document_id]
        vector_ids = doc_info.vector_ids

        if vector_ids and self._index is not None:
            ids_np = np.asarray(vector_ids, dtype=np.int64)
            try:
                removed_count = self._index.remove_ids(ids_np)
                logger.info(
                    "Removed %d vector IDs from FAISS index for document '%s'.",
                    removed_count,
                    document_id,
                )
            except Exception as e:
                logger.error("FAISS remove_ids failed for document '%s': %s", document_id, e)
                raise VectorStoreException(f"Failed to remove vectors from FAISS: {e}") from e

        # Cleanup metadata
        for vid in vector_ids:
            self._vector_metadata.pop(vid, None)

        del self._indexed_documents[document_id]
        self._updated_at = datetime.now(timezone.utc).isoformat()

        if auto_save:
            self.save_index()

        return len(vector_ids)

    # ------------------------------------------------------------------
    # Stage 5 — Semantic Search & Vector Retrieval
    # ------------------------------------------------------------------

    def search_vectors(
        self,
        query_vector: Union[List[float], np.ndarray],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        document_id_filter: Optional[str] = None,
    ) -> List[Tuple[VectorMetadata, float]]:
        """Search the FAISS index for the most similar chunks to a query vector.

        Parameters
        ----------
        query_vector : List[float] or np.ndarray
            384-dimensional query embedding vector.
        top_k : int, default=5
            Maximum number of top results to return.
        score_threshold : float, optional
            Minimum similarity score threshold. Chunks with score < threshold are filtered out.
        document_id_filter : str, optional
            If provided, filters results to only include chunks from this document.

        Returns
        -------
        List[Tuple[VectorMetadata, float]]
            List of (VectorMetadata, score) tuples ranked in descending similarity order.
        """
        if self._index is None or self.total_vectors == 0:
            return []

        # Validate and normalize query vector
        query_np = np.asarray(query_vector, dtype=np.float32)
        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)
        elif query_np.ndim != 2 or query_np.shape[0] != 1:
            raise VectorStoreException(f"Expected 1D query vector, got shape {query_np.shape}.")

        if query_np.shape[1] != self.dimension:
            raise DimensionMismatchException(
                f"Query vector dimension {query_np.shape[1]} does not match index dimension {self.dimension}."
            )

        if np.isnan(query_np).any():
            raise VectorStoreException("Query vector contains NaN values.")
        if np.isinf(query_np).any():
            raise VectorStoreException("Query vector contains infinite values.")

        # Ensure unit normalization for inner product == cosine similarity
        norm = float(np.linalg.norm(query_np))
        if norm > 0:
            query_np = query_np / norm

        # Determine search window
        if document_id_filter:
            k = min(max(top_k * 5, 50), self.total_vectors)
        else:
            k = min(top_k, self.total_vectors)

        try:
            distances, indices = self._index.search(query_np, k)
        except Exception as e:
            logger.error("FAISS search failure: %s", e)
            raise VectorStoreException(f"FAISS search execution error: {e}") from e

        results: List[Tuple[VectorMetadata, float]] = []
        for score, vid in zip(distances[0], indices[0]):
            vid = int(vid)
            if vid == -1:  # FAISS padding sentinel for insufficient results
                continue
            if vid not in self._vector_metadata:
                continue

            meta = self._vector_metadata[vid]
            if document_id_filter and meta.document_id != document_id_filter:
                continue

            score_val = float(score)
            if score_threshold is not None and score_val < score_threshold:
                continue

            results.append((meta, score_val))
            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------------------------
    # Persistence (Save & Load)
    # ------------------------------------------------------------------

    def save_index(self) -> None:
        """Persist the current FAISS index and companion metadata safely to disk."""
        if self._index is None:
            raise VectorStoreException("Cannot save uninitialized FAISS index.")

        self.index_dir.mkdir(parents=True, exist_ok=True)

        temp_index_path = self.index_dir / "index.faiss.tmp"
        temp_meta_path = self.index_dir / "metadata.json.tmp"

        try:
            # 1. Write binary FAISS index to temp file
            faiss.write_index(self._index, str(temp_index_path))

            # 2. Serialize metadata dictionary safely (JSON, no pickle)
            meta_payload = {
                "version": "1.0",
                "dimension": self.dimension,
                "embedding_model": self.embedding_model,
                "similarity_metric": "inner_product",
                "total_vectors": self.total_vectors,
                "next_vector_id": self._next_vector_id,
                "created_at": self._created_at,
                "updated_at": self._updated_at,
                "indexed_documents": {
                    k: v.model_dump(mode="json") for k, v in self._indexed_documents.items()
                },
                "vector_metadata": {
                    str(k): v.model_dump(mode="json") for k, v in self._vector_metadata.items()
                },
            }

            with open(temp_meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_payload, f, indent=2)

            # 3. Safely replace final files
            if temp_index_path.exists():
                if self.index_path.exists():
                    self.index_path.unlink()
                temp_index_path.rename(self.index_path)

            if temp_meta_path.exists():
                if self.metadata_path.exists():
                    self.metadata_path.unlink()
                temp_meta_path.rename(self.metadata_path)

            logger.info(
                "FAISS index and metadata successfully saved to '%s' (%d vectors, %d documents).",
                self.index_dir,
                self.total_vectors,
                self.total_documents,
            )

        except Exception as e:
            # Clean up temporary files on failure
            if temp_index_path.exists():
                temp_index_path.unlink()
            if temp_meta_path.exists():
                temp_meta_path.unlink()
            logger.error("Failed to persist FAISS index: %s", e)
            raise VectorStoreException(f"Index persistence failure: {e}") from e

    def load_index(self) -> bool:
        """Load the FAISS index and companion metadata from disk."""
        if not self.index_path.exists() or not self.metadata_path.exists():
            logger.warning(
                "Index files not found at '%s'. Cannot load index.",
                self.index_dir,
            )
            return False

        try:
            # 1. Read metadata JSON first
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                meta_payload = json.load(f)

            loaded_dim = meta_payload.get("dimension")
            if loaded_dim != self.dimension:
                raise DimensionMismatchException(
                    f"Saved index dimension ({loaded_dim}) does not match current model dimension ({self.dimension})."
                )

            # 2. Read FAISS index
            loaded_index = faiss.read_index(str(self.index_path))

            # 3. Validate loaded index structure
            if loaded_index.d != self.dimension:
                raise DimensionMismatchException(
                    f"FAISS index binary dimension {loaded_index.d} != configured dimension {self.dimension}."
                )

            # Ensure loaded index is wrapped as IndexIDMap2 if needed
            if isinstance(loaded_index, faiss.IndexIDMap2):
                self._index = loaded_index
            else:
                self._index = faiss.IndexIDMap2(loaded_index)

            # 4. Reconstruct in-memory metadata
            self._next_vector_id = meta_payload.get("next_vector_id", 1)
            self._created_at = meta_payload.get("created_at", datetime.now(timezone.utc).isoformat())
            self._updated_at = meta_payload.get("updated_at", self._created_at)

            # Reconstruct indexed documents
            raw_docs = meta_payload.get("indexed_documents", {})
            self._indexed_documents = {
                k: DocumentIndexInfo.model_validate(v) for k, v in raw_docs.items()
            }

            # Reconstruct vector metadata
            raw_vmeta = meta_payload.get("vector_metadata", {})
            self._vector_metadata = {
                int(k): VectorMetadata.model_validate(v) for k, v in raw_vmeta.items()
            }

            # Verify integrity
            if self._index.ntotal != len(self._vector_metadata):
                logger.warning(
                    "Integrity warning: FAISS index count (%d) differs from metadata count (%d).",
                    self._index.ntotal,
                    len(self._vector_metadata),
                )

            logger.info(
                "FAISS index and metadata loaded successfully from '%s' (%d vectors, %d documents).",
                self.index_dir,
                self.total_vectors,
                self.total_documents,
            )
            return True

        except DimensionMismatchException:
            raise
        except Exception as e:
            logger.error("Failed to load FAISS index from '%s': %s", self.index_dir, e)
            raise CorruptedIndexException(
                f"Failed to load or parse FAISS index: {e}",
                details={"index_dir": str(self.index_dir), "error": str(e)},
            ) from e

    def clear(self) -> VectorStoreActionResponse:
        """Reset the index and delete persisted index files."""
        self._init_empty_index()

        if self.index_path.exists():
            self.index_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()

        logger.info("Cleared FAISS index and removed files from '%s'.", self.index_dir)
        return VectorStoreActionResponse(
            status="success",
            message="Vector store index cleared and files removed.",
            total_vectors=0,
            total_documents=0,
        )

