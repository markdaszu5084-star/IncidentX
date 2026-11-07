"""FastAPI router for Semantic Retrieval Engine (Stage 5)."""

import logging
from fastapi import APIRouter, HTTPException, status

from app.api.knowledge import service as knowledge_service
from app.knowledge.exceptions import DimensionMismatchException, VectorStoreException
from app.schemas.retrieval import RetrievalRequest, RetrievalResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/retrieval", tags=["Semantic Retrieval"])

# Reference shared knowledge service instance
service = knowledge_service


@router.post(
    "/search",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic vector search over incident knowledge base",
    description=(
        "Accepts a natural-language query, generates a 384-dimensional dense vector "
        "using sentence-transformers/all-MiniLM-L6-v2, searches the FAISS index using "
        "cosine similarity (inner product on normalized vectors), filters by relevance threshold, "
        "and synthesizes formatted context with source citations for downstream LLM reasoning."
    ),
    responses={
        200: {"description": "Search executed successfully with ranked results and context."},
        400: {"description": "Invalid query, blank text, or dimension mismatch."},
        500: {"description": "Inference or FAISS index search error."},
    },
)
async def semantic_search(request: RetrievalRequest) -> RetrievalResponse:
    """Execute semantic vector search against indexed document chunks."""
    try:
        return service.search(request)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
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
        logger.exception("Unexpected error during semantic search: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic retrieval failed: {e}",
        )

