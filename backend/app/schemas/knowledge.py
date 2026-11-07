"""Pydantic schemas for Knowledge Base documents, chunks, and API responses."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class DocumentMetadata(BaseModel):
    """Metadata for an ingested knowledge base document."""

    document_id: str = Field(description="Unique deterministic or generated document identifier")
    filename: str = Field(description="Original uploaded filename (sanitized)")
    file_type: str = Field(description="Document format: 'pdf' or 'txt'")
    file_size_bytes: int = Field(description="File size in bytes")
    character_count: int = Field(description="Total extracted character count")
    word_count: int = Field(description="Total extracted word count")
    page_count: Optional[int] = Field(default=None, description="Total pages if PDF document")
    total_chunks: int = Field(default=0, description="Total number of generated text chunks")
    chunk_size: Optional[int] = Field(default=None, description="Character chunk size used for splitting")
    chunk_overlap: Optional[int] = Field(default=None, description="Character overlap between consecutive chunks")
    sha256_hash: str = Field(description="SHA-256 content checksum for deduplication and integrity")
    storage_path: str = Field(description="Relative path where original file is securely archived")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of document ingestion",
    )
    custom_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary user or system metadata attributes"
    )


class DocumentChunk(BaseModel):
    """Structured representation of an individual text chunk for RAG indexing."""

    chunk_id: str = Field(description="Unique deterministic chunk identifier (e.g. doc_123_c0000)")
    document_id: str = Field(description="Parent document identifier")
    filename: str = Field(description="Original document filename")
    chunk_index: int = Field(description="Sequential zero-based index of the chunk")
    text: str = Field(description="Cleaned, normalized text content of the chunk")
    character_count: int = Field(description="Character count of the chunk text")
    word_count: int = Field(description="Word count of the chunk text")
    page_number: Optional[int] = Field(default=None, description="Source page number for PDF documents")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provenance attributes (chunk size, overlap, source, timestamps)",
    )


class ChunkingConfig(BaseModel):
    """Configuration options for intelligent text chunking."""

    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Maximum character length of each chunk (100 - 10,000)",
    )
    chunk_overlap: int = Field(
        default=150,
        ge=0,
        le=2000,
        description="Character overlap between consecutive chunks (0 - 2,000)",
    )

    @model_validator(mode="after")
    def validate_overlap_less_than_size(self) -> "ChunkingConfig":
        """Ensure chunk overlap is strictly less than chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})."
            )
        return self


class DocumentUploadResponse(BaseModel):
    """API response model for successful document upload and chunking."""

    status: str = Field(default="success", description="Operation status indicator")
    message: str = Field(description="Human-readable success message")
    document: DocumentMetadata = Field(description="Detailed metadata of the uploaded document")
    total_chunks: int = Field(default=0, description="Total number of chunks produced")
    chunk_size: int = Field(default=1000, description="Chunk size applied")
    chunk_overlap: int = Field(default=150, description="Chunk overlap applied")
    extracted_text_preview: str = Field(
        description="Truncated preview of the extracted plain text"
    )


class DocumentDetail(BaseModel):
    """Detailed view of an ingested document."""

    metadata: DocumentMetadata
    extracted_text: str = Field(description="Complete extracted plain text content")


class DocumentListResponse(BaseModel):
    """API response model listing ingested knowledge base documents."""

    total_documents: int = Field(description="Total number of documents in knowledge base")
    documents: List[DocumentMetadata] = Field(description="List of document metadata items")


class ChunkListResponse(BaseModel):
    """API response model returning the list of chunks for a document."""

    document_id: str = Field(description="Parent document identifier")
    filename: str = Field(description="Parent document filename")
    total_chunks: int = Field(description="Total number of chunks")
    chunk_size: int = Field(description="Chunk size parameter used")
    chunk_overlap: int = Field(description="Chunk overlap parameter used")
    chunks: List[DocumentChunk] = Field(description="List of structured text chunks")


# ---------------------------------------------------------------------------
# Stage 3 — Embedding schemas
# ---------------------------------------------------------------------------


class ChunkEmbedding(BaseModel):
    """Stores a single chunk's embedding vector alongside its provenance metadata."""

    chunk_id: str = Field(description="Deterministic chunk identifier (e.g. doc_abc_c0001)")
    document_id: str = Field(description="Parent document identifier")
    chunk_index: int = Field(description="Zero-based position of this chunk within the document")
    embedding: List[float] = Field(description="Dense embedding vector produced by the embedding model")
    dimension: int = Field(description="Number of dimensions in the embedding vector (e.g. 384)")
    embedding_model: str = Field(description="Name/path of the model used to generate this embedding")


class DocumentEmbeddingResponse(BaseModel):
    """API response after generating and storing embeddings for a document."""

    status: str = Field(default="success", description="Operation status indicator")
    message: str = Field(description="Human-readable success or error message")
    document_id: str = Field(description="Target document identifier")
    total_embeddings_generated: int = Field(description="Number of chunk embeddings generated and persisted")
    embedding_model: str = Field(description="Model used for embedding generation")
    dimension: int = Field(description="Embedding vector dimensionality")


class EmbeddingStatusResponse(BaseModel):
    """Reports whether embeddings have been generated for a document."""

    document_id: str = Field(description="Target document identifier")
    embeddings_generated: bool = Field(description="True if embeddings file exists on disk")
    total_embeddings: int = Field(default=0, description="Number of stored embeddings (0 if none)")
    embedding_model: Optional[str] = Field(default=None, description="Model used when embeddings were created")
    dimension: Optional[int] = Field(default=None, description="Vector dimensionality of stored embeddings")


class QueryEmbeddingRequest(BaseModel):
    """Request body for embedding a natural-language query."""

    query: str = Field(description="Query string to embed for semantic similarity search")


class QueryEmbeddingResponse(BaseModel):
    """API response for a query embedding request."""

    query: str = Field(description="Original query string")
    embedding_model: str = Field(description="Model used to embed the query")
    dimension: int = Field(description="Embedding vector dimensionality")
    # Only the first 5 values — avoids sending 384 floats over the wire.
    embedding_preview: List[float] = Field(description="First 5 dimensions of the embedding (preview only)")
    vector_norm: float = Field(description="L2 norm of the full embedding vector")


# ---------------------------------------------------------------------------
# Stage 4 — FAISS Vector Database schemas
# ---------------------------------------------------------------------------


class VectorMetadata(BaseModel):
    """Metadata associated with an individual embedding vector in FAISS."""

    vector_id: int = Field(description="Integer ID corresponding to the FAISS index internal ID")
    chunk_id: str = Field(description="Deterministic chunk identifier (e.g. doc_abc_c0001)")
    document_id: str = Field(description="Parent document identifier")
    filename: str = Field(description="Original document filename")
    chunk_index: int = Field(description="Sequential zero-based index of the chunk")
    page_number: Optional[int] = Field(default=None, description="Source page number for PDF documents")
    text: str = Field(description="Cleaned, normalized text content of the chunk")
    character_count: int = Field(description="Character count of the chunk text")
    word_count: int = Field(description="Word count of the chunk text")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary source provenance metadata attributes",
    )


class DocumentIndexInfo(BaseModel):
    """Information regarding a document currently stored in the FAISS index."""

    document_id: str = Field(description="Parent document identifier")
    filename: str = Field(description="Original uploaded document filename")
    total_chunks: int = Field(description="Number of chunks/vectors indexed for this document")
    vector_ids: List[int] = Field(description="List of integer vector IDs assigned in FAISS")
    indexed_at: str = Field(description="ISO 8601 UTC timestamp of indexing")


class IndexDocumentResponse(BaseModel):
    """API response after indexing a document's embedding vectors into FAISS."""

    status: str = Field(default="success", description="Operation status indicator")
    message: str = Field(description="Human-readable status message")
    document_id: str = Field(description="Identifier of the indexed document")
    vectors_added: int = Field(description="Number of vectors newly inserted into the index")
    total_vectors_in_index: int = Field(description="Total vector count across all documents in index")


class VectorStoreStatusResponse(BaseModel):
    """API response reporting the live status of the FAISS vector database."""

    status: str = Field(default="ready", description="Vector store health status")
    initialized: bool = Field(default=True, description="Whether the FAISS index is loaded in memory")
    total_vectors: int = Field(description="Total number of active vectors in the index")
    total_documents: int = Field(description="Total number of distinct documents indexed")
    dimension: int = Field(description="Embedding vector dimensionality (e.g. 384)")
    embedding_model: str = Field(description="Configured embedding model")
    similarity_metric: str = Field(default="inner_product", description="Similarity metric (inner_product = cosine for unit vectors)")
    index_file_exists: bool = Field(description="Whether the persisted index.faiss exists on disk")
    metadata_file_exists: bool = Field(description="Whether the companion metadata.json exists on disk")


class VectorStoreStatsResponse(BaseModel):
    """Detailed statistics and inventory of the FAISS vector database."""

    status: str = Field(default="success", description="Operation status indicator")
    total_vectors: int = Field(description="Total number of vectors in index")
    total_documents: int = Field(description="Total number of distinct documents indexed")
    dimension: int = Field(description="Embedding dimensionality")
    embedding_model: str = Field(description="Model used for embeddings")
    similarity_metric: str = Field(default="inner_product", description="Similarity metric")
    is_persisted: bool = Field(description="True if in-memory index is in sync with disk files")
    index_file_exists: bool = Field(description="Whether index.faiss exists on disk")
    metadata_file_exists: bool = Field(description="Whether metadata.json exists on disk")
    indexed_documents: List[DocumentIndexInfo] = Field(
        default_factory=list,
        description="Detailed list of documents currently indexed",
    )


class VectorStoreActionResponse(BaseModel):
    """API response for index maintenance actions (save, load, clear, remove)."""

    status: str = Field(default="success", description="Operation status indicator")
    message: str = Field(description="Human-readable description of the completed action")
    total_vectors: int = Field(default=0, description="Resulting total vector count")
    total_documents: int = Field(default=0, description="Resulting total document count")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional action-specific details",
    )
