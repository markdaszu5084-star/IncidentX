"""Secure document storage and metadata persistence for the Knowledge Base."""

import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional
from app.config import get_settings
from app.knowledge.exceptions import DocumentNotFoundException
from app.schemas.knowledge import ChunkEmbedding, DocumentChunk, DocumentMetadata


class DocumentStorageManager:
    """Manages disk persistence of raw documents, extracted text, chunks, metadata, and embeddings."""

    def __init__(self, base_dir: Optional[str] = None):
        settings = get_settings()
        self.base_dir = Path(base_dir or settings.knowledge_base_dir).resolve()
        self.raw_dir = self.base_dir / "raw"
        self.metadata_dir = self.base_dir / "metadata"
        self.text_dir = self.base_dir / "text"
        self.chunks_dir = self.base_dir / "chunks"
        self.embeddings_dir = self.base_dir / "embeddings"
        self.faiss_dir = self.base_dir / "faiss"
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create necessary subdirectories safely."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_sha256(content: bytes) -> str:
        """Compute cryptographic SHA-256 hash of document bytes."""
        return hashlib.sha256(content).hexdigest()

    def store_document(
        self,
        document_id: str,
        filename: str,
        content: bytes,
        metadata: DocumentMetadata,
        extracted_text: str,
    ) -> Path:
        """
        Securely persist the raw file, extracted text, and metadata JSON to disk.
        """
        # 1. Save raw binary
        raw_target = self.raw_dir / f"{document_id}_{filename}"
        with open(raw_target, "wb") as f:
            f.write(content)

        # 2. Save extracted text
        text_target = self.text_dir / f"{document_id}.txt"
        with open(text_target, "w", encoding="utf-8") as f:
            f.write(extracted_text)

        # 3. Save metadata JSON
        meta_target = self.metadata_dir / f"{document_id}.json"
        with open(meta_target, "w", encoding="utf-8") as f:
            f.write(metadata.model_dump_json(indent=2))

        return raw_target

    def store_chunks(
        self,
        document_id: str,
        chunks: List[DocumentChunk],
    ) -> Path:
        """
        Persist processed text chunks to an organized JSON file.
        Preserves document-to-chunk relationships and chunk metadata.
        """
        chunks_target = self.chunks_dir / f"{document_id}_chunks.json"
        serialized = [c.model_dump(mode="json") for c in chunks]

        container = {
            "document_id": document_id,
            "total_chunks": len(chunks),
            "chunks": serialized,
        }

        with open(chunks_target, "w", encoding="utf-8") as f:
            json.dump(container, f, indent=2)

        return chunks_target

    def get_metadata(self, document_id: str) -> DocumentMetadata:
        """Retrieve stored metadata for a given document ID."""
        meta_target = self.metadata_dir / f"{document_id}.json"
        if not meta_target.exists():
            raise DocumentNotFoundException(
                f"Document with ID '{document_id}' was not found.",
                details={"document_id": document_id},
            )

        with open(meta_target, "r", encoding="utf-8") as f:
            data = json.load(f)
        return DocumentMetadata.model_validate(data)

    def save_metadata(self, metadata: DocumentMetadata) -> None:
        """Update and overwrite existing document metadata."""
        meta_target = self.metadata_dir / f"{metadata.document_id}.json"
        with open(meta_target, "w", encoding="utf-8") as f:
            f.write(metadata.model_dump_json(indent=2))

    def get_text(self, document_id: str) -> str:
        """Retrieve stored extracted text for a given document ID."""
        text_target = self.text_dir / f"{document_id}.txt"
        if not text_target.exists():
            raise DocumentNotFoundException(
                f"Extracted text for document '{document_id}' not found.",
                details={"document_id": document_id},
            )

        with open(text_target, "r", encoding="utf-8") as f:
            return f.read()

    def get_chunks(self, document_id: str) -> List[DocumentChunk]:
        """Retrieve stored chunks for a given document ID."""
        # Check document existence first
        self.get_metadata(document_id)

        chunks_target = self.chunks_dir / f"{document_id}_chunks.json"
        if not chunks_target.exists():
            return []

        with open(chunks_target, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunk_list = data.get("chunks", [])
        return [DocumentChunk.model_validate(c) for c in chunk_list]

    def get_chunk(self, document_id: str, chunk_index: int) -> DocumentChunk:
        """Retrieve a specific chunk by zero-based index."""
        chunks = self.get_chunks(document_id)
        if chunk_index < 0 or chunk_index >= len(chunks):
            raise DocumentNotFoundException(
                f"Chunk index {chunk_index} out of range for document '{document_id}' (total: {len(chunks)}).",
                details={"document_id": document_id, "chunk_index": chunk_index, "total_chunks": len(chunks)},
            )
        return chunks[chunk_index]

    def list_documents(self) -> List[DocumentMetadata]:
        """List all ingested documents sorted by creation date descending."""
        documents: List[DocumentMetadata] = []
        if not self.metadata_dir.exists():
            return documents

        for json_file in self.metadata_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                documents.append(DocumentMetadata.model_validate(data))
            except Exception:
                continue

        # Sort by creation timestamp descending (newest first)
        documents.sort(key=lambda d: d.created_at, reverse=True)
        return documents

    def delete_document(self, document_id: str) -> bool:
        """Remove all stored artifacts (raw, text, chunks, metadata) for a document ID."""
        deleted_any = False

        # Metadata
        meta_target = self.metadata_dir / f"{document_id}.json"
        if meta_target.exists():
            meta_target.unlink()
            deleted_any = True

        # Text
        text_target = self.text_dir / f"{document_id}.txt"
        if text_target.exists():
            text_target.unlink()
            deleted_any = True

        # Chunks
        chunks_target = self.chunks_dir / f"{document_id}_chunks.json"
        if chunks_target.exists():
            chunks_target.unlink()
            deleted_any = True

        # Raw file
        for raw_file in self.raw_dir.glob(f"{document_id}_*"):
            if raw_file.is_file():
                raw_file.unlink()
                deleted_any = True

        # Embeddings (may not exist — do not count towards deleted_any)
        embeddings_target = self.embeddings_dir / f"{document_id}_embeddings.json"
        if embeddings_target.exists():
            embeddings_target.unlink()

        if not deleted_any:
            raise DocumentNotFoundException(f"Document with ID '{document_id}' not found.")

        return True


    # ------------------------------------------------------------------
    # Stage 3 — Embedding persistence
    # ------------------------------------------------------------------

    def store_embeddings(self, document_id: str, embeddings: list) -> Path:
        """Persist chunk embeddings to a JSON file under knowledge_base/embeddings/.

        Each record stores chunk_id, document_id, chunk_index, dimension,
        embedding_model, and the full embedding vector as a list of floats.
        """
        from app.schemas.knowledge import ChunkEmbedding  # local import avoids circular refs

        target = self.embeddings_dir / f"{document_id}_embeddings.json"
        serialized = [e.model_dump(mode="json") for e in embeddings]
        container = {
            "document_id": document_id,
            "total_embeddings": len(embeddings),
            "embeddings": serialized,
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(container, f, indent=2)
        return target

    def get_embeddings(self, document_id: str) -> list:
        """Load persisted chunk embeddings for a document.

        Returns an empty list if no embeddings file exists yet.
        """
        from app.schemas.knowledge import ChunkEmbedding  # local import

        # Confirm document exists first
        self.get_metadata(document_id)

        target = self.embeddings_dir / f"{document_id}_embeddings.json"
        if not target.exists():
            return []

        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [ChunkEmbedding.model_validate(e) for e in data.get("embeddings", [])]

    def embeddings_exist(self, document_id: str) -> bool:
        """Return True if an embeddings file exists for the given document ID."""
        target = self.embeddings_dir / f"{document_id}_embeddings.json"
        return target.exists()

    def delete_embeddings(self, document_id: str) -> bool:
        """Remove stored embeddings for a document if present."""
        target = self.embeddings_dir / f"{document_id}_embeddings.json"
        if target.exists():
            target.unlink()
            return True
        return False
