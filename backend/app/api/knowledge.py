"""FastAPI router for Knowledge Base document upload, intelligent chunking, inspection, retrieval, and embedding."""

import logging
from typing import Dict, Optional
from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile, status
from fastapi.responses import PlainTextResponse
from app.knowledge.exceptions import (
    CorruptedDocumentException,
    CorruptedIndexException,
    DimensionMismatchException,
    DocumentNotFoundException,
    DuplicateDocumentException,
    FileSizeExceededException,
    InvalidFileTypeException,
    VectorStoreException,
)
from app.knowledge.service import KnowledgeDocumentService
from app.schemas.knowledge import (
    ChunkListResponse,
    DocumentChunk,
    DocumentDetail,
    DocumentEmbeddingResponse,
    DocumentListResponse,
    DocumentUploadResponse,
    EmbeddingStatusResponse,
    IndexDocumentResponse,
    QueryEmbeddingRequest,
    QueryEmbeddingResponse,
    VectorStoreActionResponse,
    VectorStoreStatsResponse,
    VectorStoreStatusResponse,
)
from app.schemas.retrieval import RetrievalRequest, RetrievalResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/knowledge", tags=["Knowledge Base"])

# Instantiate service singleton
service = KnowledgeDocumentService()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload, ingest, and chunk a document",
    description=(
        "Upload a PDF (.pdf) or Plain Text (.txt) document. "
        "Validates magic bytes, extracts text, cleans formatting, "
        "generates intelligent overlapping chunks using LangChain RecursiveCharacterTextSplitter, "
        "and securely archives raw files, text, metadata, and chunk artifacts."
    ),
    responses={
        201: {"description": "Document successfully uploaded, parsed, chunked, and stored."},
        400: {"description": "Unsupported file format, invalid parameters, or bad file structure."},
        413: {"description": "File size exceeds maximum permitted limit."},
        422: {"description": "Corrupted or unreadable document content."},
        500: {"description": "Internal server processing error."},
    },
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload (.pdf or .txt)"),
    chunk_size: Optional[int] = Query(
        default=None,
        ge=100,
        le=10000,
        description="Optional chunk size in characters (default: 1000)",
    ),
    chunk_overlap: Optional[int] = Query(
        default=None,
        ge=0,
        le=2000,
        description="Optional chunk overlap in characters (default: 150)",
    ),
) -> DocumentUploadResponse:
    """Handle asynchronous document upload, text extraction, and intelligent chunking."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing filename in upload payload.",
        )

    # Validate overlap relationship if both provided
    if chunk_size is not None and chunk_overlap is not None:
        if chunk_overlap >= chunk_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size}).",
            )

    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"Error reading uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file stream: {str(e)}",
        )

    try:
        response = service.process_and_store_document(
            filename=file.filename,
            content=content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return response

    except InvalidFileTypeException as e:
        logger.warning(f"File validation rejected '{file.filename}': {e.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message,
        )
    except FileSizeExceededException as e:
        logger.warning(f"File size exceeded for '{file.filename}': {e.message}")
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=e.message,
        )
    except CorruptedDocumentException as e:
        logger.warning(f"Corrupted document '{file.filename}': {e.message}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.message,
        )
    except ValueError as e:
        logger.warning(f"Validation error for '{file.filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.exception(f"Unexpected error processing document '{file.filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while processing the document: {str(e)}",
        )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all ingested documents",
    description="Returns metadata for all documents currently ingested in the knowledge base.",
)
async def list_documents() -> DocumentListResponse:
    """List all ingested documents with metadata."""
    try:
        return service.list_all_documents()
    except Exception as e:
        logger.exception(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve documents: {str(e)}",
        )


@router.get(
    "/documents/{document_id}",
    response_model=DocumentDetail,
    status_code=status.HTTP_200_OK,
    summary="Get document details by ID",
    description="Retrieves metadata and full extracted text for a specific document.",
    responses={
        200: {"description": "Document found and returned."},
        404: {"description": "Document ID not found."},
    },
)
async def get_document(document_id: str) -> DocumentDetail:
    """Retrieve details and full text for a single document."""
    try:
        return service.get_document_detail(document_id)
    except DocumentNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message,
        )
    except Exception as e:
        logger.exception(f"Error fetching document '{document_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not fetch document: {str(e)}",
        )


@router.get(
    "/documents/{document_id}/text",
    response_class=PlainTextResponse,
    status_code=status.HTTP_200_OK,
    summary="Get raw extracted document text",
    description="Returns the clean plain text of the document directly as text/plain.",
    responses={
        200: {"description": "Extracted text returned."},
        404: {"description": "Document ID not found."},
    },
)
async def get_document_raw_text(document_id: str) -> PlainTextResponse:
    """Return raw extracted text for a document."""
    try:
        detail = service.get_document_detail(document_id)
        return PlainTextResponse(content=detail.extracted_text, media_type="text/plain; charset=utf-8")
    except DocumentNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message,
        )


@router.get(
    "/documents/{document_id}/chunks",
    response_model=ChunkListResponse,
    status_code=status.HTTP_200_OK,
    summary="Get all chunks for a document",
    description="Retrieves all structured text chunks and metadata for a specific document.",
    responses={
        200: {"description": "List of document chunks returned."},
        404: {"description": "Document ID not found."},
    },
)
async def get_document_chunks(document_id: str) -> ChunkListResponse:
    """Return all generated chunks for a document."""
    try:
        return service.get_document_chunks(document_id)
    except DocumentNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message,
        )
    except Exception as e:
        logger.exception(f"Error fetching chunks for document '{document_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not fetch chunks: {str(e)}",
        )


@router.get(
    "/documents/{document_id}/chunks/{chunk_index}",
    response_model=DocumentChunk,
    status_code=status.HTTP_200_OK,
    summary="Get a specific chunk by index",
    description="Retrieves an individual chunk by its zero-based index.",
    responses={
        200: {"description": "Chunk returned."},
        404: {"description": "Document ID or chunk index not found."},
    },
)
async def get_document_chunk(document_id: str, chunk_index: int) -> DocumentChunk:
    """Return an individual chunk by zero-based index."""
    try:
        return service.get_document_chunk(document_id, chunk_index)
    except DocumentNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message,
        )


@router.post(
    "/documents/{document_id}/rechunk",
    response_model=ChunkListResponse,
    status_code=status.HTTP_200_OK,
    summary="Re-chunk a document with new parameters",
    description="Re-splits an existing document with modified chunk_size and chunk_overlap without re-uploading.",
    responses={
        200: {"description": "Document re-chunked successfully."},
        400: {"description": "Invalid chunk configuration."},
        404: {"description": "Document ID not found."},
    },
)
async def rechunk_document(
    document_id: str,
    chunk_size: Optional[int] = Query(
        default=None,
        ge=100,
        le=10000,
        description="New chunk size in characters (100 - 10,000)",
    ),
    chunk_overlap: Optional[int] = Query(
        default=None,
        ge=0,
        le=2000,
        description="New chunk overlap in characters (0 - 2,000)",
    ),
) -> ChunkListResponse:
    """Re-chunk an existing document with new parameters."""
    if chunk_size is not None and chunk_overlap is not None:
        if chunk_overlap >= chunk_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size}).",
            )
    try:
        return service.rechunk_document(
            document_id=document_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except DocumentNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a document",
    description="Deletes raw file, extracted text, chunks, and metadata for a document.",
    responses={
        200: {"description": "Document successfully deleted."},
        404: {"description": "Document ID not found."},
    },
)
async def delete_document(document_id: str) -> Dict[str, str]:
    """Delete a document by ID."""
    try:
        service.delete_document(document_id)
        return {
            "status": "success",
            "message": f"Document '{document_id}' successfully deleted.",
            "document_id": document_id,
        }
    except DocumentNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message,
        )


# ---------------------------------------------------------------------------
# Stage 3 — Embedding endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/documents/{document_id}/embed",
    response_model=DocumentEmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate and store embeddings for a document",
    description=(
        "Loads the processed chunks for the given document, generates 384-dimensional "
        "embeddings using sentence-transformers/all-MiniLM-L6-v2 (or the configured model), "
        "and persists them under knowledge_base/embeddings/. "
        "Returns embedding count and dimension — raw vectors are NOT returned."
    ),
    responses={
        200: {"description": "Embeddings generated and stored successfully."},
        404: {"description": "Document ID not found."},
        500: {"description": "Embedding inference or storage error."},
    },
)
async def embed_document(document_id: str) -> DocumentEmbeddingResponse:
    """Trigger embedding generation for all chunks of a document."""
    try:
        return service.generate_and_store_document_embeddings(document_id)
    except DocumentNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except RuntimeError as e:
        logger.exception("Embedding inference failed for document '%s': %s", document_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {e}",
        )
    except Exception as e:
        logger.exception("Unexpected error embedding document '%s': %s", document_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {e}",
        )


@router.get(
    "/documents/{document_id}/embeddings/status",
    response_model=EmbeddingStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Check embedding status for a document",
    description="Reports whether embeddings have been generated for a document and how many exist.",
    responses={
        200: {"description": "Embedding status returned."},
        404: {"description": "Document ID not found."},
    },
)
async def get_embedding_status(document_id: str) -> EmbeddingStatusResponse:
    """Return embedding generation status for a document."""
    try:
        return service.get_embedding_status(document_id)
    except DocumentNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message,
        )
    except Exception as e:
        logger.exception("Error checking embedding status for '%s': %s", document_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not check embedding status: {e}",
        )


@router.post(
    "/embed-query",
    response_model=QueryEmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Embed a natural-language query",
    description=(
        "Embeds the supplied query string using the same model as documents. "
        "Returns the vector dimensionality, L2 norm, and a 5-element preview. "
        "Full embedding arrays are intentionally omitted from the response."
    ),
    responses={
        200: {"description": "Query embedded successfully."},
        400: {"description": "Empty or invalid query string."},
        500: {"description": "Embedding inference error."},
    },
)
async def embed_query(body: QueryEmbeddingRequest) -> QueryEmbeddingResponse:
    """Embed a user query using the document embedding model."""
    try:
        return service.generate_query_embedding(body.query)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except RuntimeError as e:
        logger.exception("Query embedding inference failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query embedding failed: {e}",
        )
    except Exception as e:
        logger.exception("Unexpected error during query embedding: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {e}",
        )


# ---------------------------------------------------------------------------
# Stage 4 — FAISS Vector Database endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/vectorstore/status",
    response_model=VectorStoreStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get FAISS vector store status",
    description="Reports the live in-memory status and persistence health of the FAISS vector database.",
)
async def get_vector_store_status() -> VectorStoreStatusResponse:
    """Return live status of the FAISS vector store."""
    try:
        return service.get_vector_store_status()
    except Exception as e:
        logger.exception("Error retrieving vector store status: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve vector store status: {e}",
        )


@router.get(
    "/vectorstore/stats",
    response_model=VectorStoreStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get FAISS vector store statistics",
    description="Returns detailed statistics, vector counts, dimensions, and indexed document inventory.",
)
async def get_vector_store_stats() -> VectorStoreStatsResponse:
    """Return comprehensive statistics and inventory of the FAISS index."""
    try:
        return service.get_vector_store_stats()
    except Exception as e:
        logger.exception("Error retrieving vector store stats: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve vector store stats: {e}",
        )


@router.post(
    "/documents/{document_id}/index",
    response_model=IndexDocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Index document embeddings into FAISS",
    description=(
        "Inserts all chunk embeddings and metadata for a document into the FAISS index. "
        "Reuses existing embeddings if generated, or generates them automatically. "
        "Rejects duplicate documents with 409 Conflict unless overwrite=true is specified."
    ),
    responses={
        200: {"description": "Document successfully indexed into FAISS."},
        400: {"description": "Dimension mismatch or empty chunks."},
        404: {"description": "Document ID not found."},
        409: {"description": "Document is already indexed."},
        500: {"description": "Vector store or indexing error."},
    },
)
async def index_document(
    document_id: str,
    overwrite: bool = Query(
        default=False,
        description="If True, removes existing vectors for this document and re-indexes.",
    ),
) -> IndexDocumentResponse:
    """Index a document's chunks and embedding vectors into FAISS."""
    try:
        return service.index_document(document_id, overwrite=overwrite)
    except DocumentNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message,
        )
    except DuplicateDocumentException as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=e.message,
        )
    except DimensionMismatchException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message,
        )
    except VectorStoreException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message,
        )
    except Exception as e:
        logger.exception("Unexpected error indexing document '%s': %s", document_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index document: {e}",
        )


@router.post(
    "/vectorstore/save",
    response_model=VectorStoreActionResponse,
    status_code=status.HTTP_200_OK,
    summary="Persist FAISS index to disk",
    description="Explicitly flushes the in-memory FAISS index and companion metadata to local disk storage.",
)
async def save_vector_store() -> VectorStoreActionResponse:
    """Persist current FAISS index and metadata to local disk."""
    try:
        return service.save_vector_store()
    except Exception as e:
        logger.exception("Error saving vector store: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save FAISS index: {e}",
        )


@router.post(
    "/vectorstore/load",
    response_model=VectorStoreActionResponse,
    status_code=status.HTTP_200_OK,
    summary="Reload FAISS index from disk",
    description="Reloads the persisted FAISS index and companion metadata from local disk storage into memory.",
    responses={
        200: {"description": "FAISS index loaded successfully."},
        500: {"description": "Corrupted or incompatible index file."},
    },
)
async def load_vector_store() -> VectorStoreActionResponse:
    """Reload the FAISS index and companion metadata from local disk."""
    try:
        return service.load_vector_store()
    except CorruptedIndexException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message,
        )
    except Exception as e:
        logger.exception("Error loading vector store: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load FAISS index: {e}",
        )


@router.delete(
    "/vectorstore/documents/{document_id}",
    response_model=VectorStoreActionResponse,
    status_code=status.HTTP_200_OK,
    summary="Remove document from FAISS index",
    description="Removes all vectors and metadata for a document from the FAISS index without deleting raw files.",
    responses={
        200: {"description": "Document vectors successfully removed from FAISS."},
        404: {"description": "Document not found in vector store."},
    },
)
async def remove_document_from_vector_store(document_id: str) -> VectorStoreActionResponse:
    """Remove a document and its vectors from the FAISS vector store."""
    try:
        return service.remove_document_from_index(document_id)
    except DocumentNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message,
        )
    except Exception as e:
        logger.exception("Error removing document '%s' from vector store: %s", document_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove document from vector store: {e}",
        )


@router.post(
    "/search",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic vector search (Knowledge Base alias)",
    description="Convenience alias forwarding to /api/v1/retrieval/search.",
)
async def semantic_search_alias(request: RetrievalRequest) -> RetrievalResponse:
    """Convenience alias forwarding to /api/v1/retrieval/search."""
    return service.search(request)
