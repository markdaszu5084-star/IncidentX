"""High-level Knowledge Base ingestion service orchestrating validation, extraction, chunking, storage, and embedding."""

import uuid
from typing import Any, List, Optional
from app.config import get_settings
from app.knowledge.chunker import IntelligentChunker
from app.knowledge.extractor import extract_document_content
from app.knowledge.storage import DocumentStorageManager
from app.knowledge.validator import validate_document_upload
from app.schemas.knowledge import (
    ChunkListResponse,
    DocumentChunk,
    DocumentDetail,
    DocumentEmbeddingResponse,
    DocumentListResponse,
    DocumentMetadata,
    DocumentUploadResponse,
    EmbeddingStatusResponse,
    IndexDocumentResponse,
    QueryEmbeddingResponse,
    VectorStoreActionResponse,
    VectorStoreStatsResponse,
    VectorStoreStatusResponse,
)
from app.schemas.retrieval import RetrievalRequest, RetrievalResponse


class KnowledgeDocumentService:
    """Service handling document upload lifecycle from raw byte validation to intelligent chunking and persistence."""

    def __init__(
        self,
        storage_manager: Optional[DocumentStorageManager] = None,
        vector_store: Optional[Any] = None,
        embedding_service: Optional[Any] = None,
    ):
        self.storage = storage_manager or DocumentStorageManager()
        self.settings = get_settings()
        from app.knowledge.embeddings import get_embedding_service
        self.embedding_service = embedding_service or get_embedding_service()
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            from app.knowledge.vector_store import FaissVectorStore
            self.vector_store = FaissVectorStore(
                storage_dir=self.storage.faiss_dir,
                embedding_service=self.embedding_service,
            )
        from app.knowledge.retrieval import SemanticRetrievalEngine
        self.retrieval_engine = SemanticRetrievalEngine(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
        )

    def process_and_store_document(
        self,
        filename: str,
        content: bytes,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        custom_metadata: Optional[dict] = None,
    ) -> DocumentUploadResponse:
        """
        Validate, extract text, intelligently chunk, and store a document.

        Steps:
        1. Security, extension, and magic byte validation.
        2. Cryptographic checksum calculation.
        3. Text extraction and normalization.
        4. Intelligent text chunking with RecursiveCharacterTextSplitter.
        5. Unique document ID generation.
        6. Disk persistence of raw file, extracted text, metadata, and chunks.
        7. Response formatting with preview and chunk metrics.
        """
        # 1. Validation
        clean_filename, file_type = validate_document_upload(
            filename=filename,
            content=content,
            max_size_bytes=self.settings.max_doc_file_size_bytes,
        )

        # 2. Checksum
        sha256 = self.storage.compute_sha256(content)

        # 3. Extraction & Cleaning
        extracted = extract_document_content(content, file_type)

        # 4. Generate unique document ID
        doc_id = f"doc_{uuid.uuid4().hex[:12]}"

        # 5. Intelligent Chunking
        chunker = IntelligentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk_extracted_content(
            document_id=doc_id,
            filename=clean_filename,
            content=extracted,
            custom_metadata=custom_metadata,
        )

        # 6. Metadata creation
        metadata = DocumentMetadata(
            document_id=doc_id,
            filename=clean_filename,
            file_type=file_type,
            file_size_bytes=len(content),
            character_count=extracted.character_count,
            word_count=extracted.word_count,
            page_count=extracted.page_count,
            total_chunks=len(chunks),
            chunk_size=chunker.chunk_size,
            chunk_overlap=chunker.chunk_overlap,
            sha256_hash=sha256,
            storage_path=f"raw/{doc_id}_{clean_filename}",
            custom_metadata=custom_metadata or {},
        )

        # 7. Disk persistence
        self.storage.store_document(
            document_id=doc_id,
            filename=clean_filename,
            content=content,
            metadata=metadata,
            extracted_text=extracted.text,
        )
        self.storage.store_chunks(
            document_id=doc_id,
            chunks=chunks,
        )

        # 8. Format preview
        preview_length = 300
        preview = (
            extracted.text[:preview_length] + ("..." if len(extracted.text) > preview_length else "")
            if extracted.text
            else "[No text extracted from document]"
        )

        return DocumentUploadResponse(
            status="success",
            message=f"Document '{clean_filename}' ({file_type.upper()}) successfully ingested, extracted, and split into {len(chunks)} chunks.",
            document=metadata,
            total_chunks=len(chunks),
            chunk_size=chunker.chunk_size,
            chunk_overlap=chunker.chunk_overlap,
            extracted_text_preview=preview,
        )

    def get_document_detail(self, document_id: str) -> DocumentDetail:
        """Fetch full document metadata and complete extracted text."""
        metadata = self.storage.get_metadata(document_id)
        text = self.storage.get_text(document_id)
        return DocumentDetail(metadata=metadata, extracted_text=text)

    def list_all_documents(self) -> DocumentListResponse:
        """List all ingested knowledge documents."""
        docs = self.storage.list_documents()
        return DocumentListResponse(
            total_documents=len(docs),
            documents=docs,
        )

    def get_document_chunks(self, document_id: str) -> ChunkListResponse:
        """Fetch all structured chunks for a document."""
        metadata = self.storage.get_metadata(document_id)
        chunks = self.storage.get_chunks(document_id)
        return ChunkListResponse(
            document_id=document_id,
            filename=metadata.filename,
            total_chunks=len(chunks),
            chunk_size=metadata.chunk_size or self.settings.default_chunk_size,
            chunk_overlap=metadata.chunk_overlap or self.settings.default_chunk_overlap,
            chunks=chunks,
        )

    def get_document_chunk(self, document_id: str, chunk_index: int) -> DocumentChunk:
        """Fetch an individual chunk by index."""
        return self.storage.get_chunk(document_id, chunk_index)

    def rechunk_document(
        self,
        document_id: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> ChunkListResponse:
        """
        Re-chunk an existing ingested document with modified chunk_size or chunk_overlap.
        Updates stored chunks and document metadata.
        """
        metadata = self.storage.get_metadata(document_id)
        text = self.storage.get_text(document_id)

        chunker = IntelligentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        new_chunks = chunker.chunk_text(
            text=text,
            document_id=document_id,
            filename=metadata.filename,
            page_number=None,
            start_index=0,
            custom_metadata=metadata.custom_metadata,
        )

        # Store updated chunks
        self.storage.store_chunks(document_id, new_chunks)

        # Update metadata
        metadata.total_chunks = len(new_chunks)
        metadata.chunk_size = chunker.chunk_size
        metadata.chunk_overlap = chunker.chunk_overlap
        self.storage.save_metadata(metadata)

        return ChunkListResponse(
            document_id=document_id,
            filename=metadata.filename,
            total_chunks=len(new_chunks),
            chunk_size=chunker.chunk_size,
            chunk_overlap=chunker.chunk_overlap,
            chunks=new_chunks,
        )

    def delete_document(self, document_id: str) -> bool:
        """Delete an ingested document, its stored artifacts, and its vectors from FAISS."""
        if hasattr(self, "vector_store") and self.vector_store.is_document_indexed(document_id):
            try:
                self.vector_store.remove_document_vectors(document_id, auto_save=True)
            except Exception:
                pass
        return self.storage.delete_document(document_id)

    # ------------------------------------------------------------------
    # Stage 3 — Embedding methods
    # ------------------------------------------------------------------

    def generate_and_store_document_embeddings(
        self,
        document_id: str,
    ) -> "DocumentEmbeddingResponse":
        """Generate embeddings for all chunks of a document and persist them.

        Steps
        -----
        1. Load stored chunks from Stage 2.
        2. Run batch embedding via EmbeddingService.
        3. Persist ChunkEmbedding objects to knowledge_base/embeddings/.
        4. Return a summary response (no raw embedding arrays exposed).
        """
        from app.knowledge.embeddings import get_embedding_service

        # Confirm document exists (raises DocumentNotFoundException if not)
        self.storage.get_metadata(document_id)

        chunks = self.storage.get_chunks(document_id)
        if not chunks:
            return DocumentEmbeddingResponse(
                status="warning",
                message=f"Document '{document_id}' has no chunks to embed.",
                document_id=document_id,
                total_embeddings_generated=0,
                embedding_model=self.settings.embedding_model,
                dimension=self.settings.embedding_dimension,
            )

        svc = get_embedding_service()
        embeddings = svc.embed_chunks(chunks)
        self.storage.store_embeddings(document_id, embeddings)

        return DocumentEmbeddingResponse(
            status="success",
            message=(
                f"Generated and stored {len(embeddings)} embeddings for document '{document_id}'."
            ),
            document_id=document_id,
            total_embeddings_generated=len(embeddings),
            embedding_model=svc.model_name,
            dimension=svc.expected_dimension,
        )

    def get_embedding_status(self, document_id: str) -> "EmbeddingStatusResponse":
        """Check whether embeddings have been generated for a document."""
        # Confirm document exists
        self.storage.get_metadata(document_id)

        if not self.storage.embeddings_exist(document_id):
            return EmbeddingStatusResponse(
                document_id=document_id,
                embeddings_generated=False,
                total_embeddings=0,
            )

        stored = self.storage.get_embeddings(document_id)
        first_model = stored[0].embedding_model if stored else None
        first_dim = stored[0].dimension if stored else None

        return EmbeddingStatusResponse(
            document_id=document_id,
            embeddings_generated=True,
            total_embeddings=len(stored),
            embedding_model=first_model,
            dimension=first_dim,
        )

    def generate_query_embedding(self, query: str) -> "QueryEmbeddingResponse":
        """Embed a natural-language query; return dimension info and a preview.

        Only the first 5 dimensions and the L2 norm are returned to avoid
        transmitting the full 384-float array over the wire unnecessarily.
        """
        from app.knowledge.embeddings import get_embedding_service

        svc = get_embedding_service()
        vector = svc.embed_query(query)
        norm = svc.compute_vector_norm(vector)

        return QueryEmbeddingResponse(
            query=query,
            embedding_model=svc.model_name,
            dimension=len(vector),
            embedding_preview=vector[:5],
            vector_norm=round(norm, 6),
        )


    # ------------------------------------------------------------------
    # Stage 4 — FAISS Vector Database methods
    # ------------------------------------------------------------------

    def index_document(
        self,
        document_id: str,
        overwrite: bool = False,
    ) -> "IndexDocumentResponse":
        """Index a document's chunks and embedding vectors into the FAISS vector database.

        If embeddings were already generated (Stage 3), reuses them.
        Otherwise, generates and persists them first.
        """
        from app.knowledge.exceptions import VectorStoreException

        # Confirm document exists (raises DocumentNotFoundException if not)
        self.storage.get_metadata(document_id)
        chunks = self.storage.get_chunks(document_id)
        if not chunks:
            raise VectorStoreException(f"Document '{document_id}' has no chunks to index.")

        # Re-use existing embeddings or generate on the fly
        embeddings = self.storage.get_embeddings(document_id)
        if not embeddings:
            from app.knowledge.embeddings import get_embedding_service
            svc = get_embedding_service()
            embeddings = svc.embed_chunks(chunks)
            self.storage.store_embeddings(document_id, embeddings)

        return self.vector_store.add_document_vectors(
            document_id=document_id,
            chunks=chunks,
            embeddings=embeddings,
            overwrite=overwrite,
            auto_save=True,
        )

    def remove_document_from_index(self, document_id: str) -> "VectorStoreActionResponse":
        """Remove a document from the FAISS vector index without deleting its raw files."""
        removed_count = self.vector_store.remove_document_vectors(document_id, auto_save=True)
        return VectorStoreActionResponse(
            status="success",
            message=f"Removed {removed_count} vectors for document '{document_id}' from the vector store.",
            total_vectors=self.vector_store.total_vectors,
            total_documents=self.vector_store.total_documents,
            details={"document_id": document_id, "vectors_removed": removed_count},
        )

    def get_vector_store_status(self) -> "VectorStoreStatusResponse":
        """Get live health and status information for the FAISS vector store."""
        return self.vector_store.get_status()

    def get_vector_store_stats(self) -> "VectorStoreStatsResponse":
        """Get comprehensive statistics and inventory for the FAISS vector store."""
        return self.vector_store.get_stats()

    def save_vector_store(self) -> "VectorStoreActionResponse":
        """Persist the current FAISS index and metadata to disk."""
        self.vector_store.save_index()
        return VectorStoreActionResponse(
            status="success",
            message="FAISS vector index and metadata successfully persisted to disk.",
            total_vectors=self.vector_store.total_vectors,
            total_documents=self.vector_store.total_documents,
        )

    def load_vector_store(self) -> "VectorStoreActionResponse":
        """Reload the FAISS vector index and metadata from disk."""
        loaded = self.vector_store.load_index()
        if not loaded:
            return VectorStoreActionResponse(
                status="warning",
                message="No existing index files found on disk to load.",
                total_vectors=self.vector_store.total_vectors,
                total_documents=self.vector_store.total_documents,
            )
        return VectorStoreActionResponse(
            status="success",
            message="FAISS vector index and metadata successfully loaded from disk.",
            total_vectors=self.vector_store.total_vectors,
            total_documents=self.vector_store.total_documents,
        )


    # ------------------------------------------------------------------
    # Stage 5 — Semantic Retrieval methods
    # ------------------------------------------------------------------

    def search(self, request: "RetrievalRequest") -> "RetrievalResponse":
        """Execute semantic search against the FAISS index and prepare LLM context."""
        return self.retrieval_engine.search(request)
