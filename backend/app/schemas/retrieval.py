"""Pydantic schemas for the Semantic Retrieval Engine (Stage 5)."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class RetrievalRequest(BaseModel):
    """Request model for semantic vector search over knowledge base chunks."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Natural-language question, diagnostic query, or incident description",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum number of relevant chunks to retrieve",
    )
    score_threshold: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Minimum cosine similarity score threshold (-1.0 to 1.0; 0.0 default)",
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Optional filter to restrict retrieval to a specific document ID",
    )
    max_context_characters: int = Field(
        default=8000,
        ge=100,
        le=50000,
        description="Maximum total characters permitted in the aggregated context_text",
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_blank(cls, v: str) -> str:
        """Ensure query is not empty or composed solely of whitespace."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query string cannot be empty or whitespace.")
        return stripped


class SearchResultItem(BaseModel):
    """An individual retrieved chunk with provenance metadata and similarity score."""

    chunk_id: str = Field(description="Deterministic chunk identifier (e.g. doc_123_c0001)")
    document_id: str = Field(description="Parent document identifier")
    filename: str = Field(description="Original document filename")
    chunk_index: int = Field(description="Sequential zero-based index of the chunk")
    page_number: Optional[int] = Field(default=None, description="Source page number if available")
    text: str = Field(description="Extracted and normalized chunk text content")
    similarity_score: float = Field(
        description="Cosine similarity score between query and chunk vector in range [-1.0, 1.0]"
    )
    character_count: int = Field(description="Character length of chunk text")
    word_count: int = Field(description="Word count of chunk text")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provenance attributes (source, chunk size, overlap)",
    )


class SourceReference(BaseModel):
    """Source reference for LLM citation and attribution."""

    chunk_id: str = Field(description="Referenced chunk identifier")
    document_id: str = Field(description="Referenced parent document identifier")
    filename: str = Field(description="Referenced source filename")
    page_number: Optional[int] = Field(default=None, description="Source page number if available")
    similarity_score: float = Field(description="Cosine similarity score of this source")


class RetrievalResponse(BaseModel):
    """Structured response containing ranked search results and prepared context."""

    status: str = Field(default="success", description="Retrieval operation status")
    query: str = Field(description="Normalized query string searched")
    total_results: int = Field(description="Number of chunks matching criteria and score threshold")
    top_k: int = Field(description="Requested maximum number of results")
    score_threshold: float = Field(description="Minimum score threshold applied")
    results: List[SearchResultItem] = Field(
        default_factory=list,
        description="Ranked list of matching chunks sorted by descending similarity",
    )
    context_text: str = Field(
        default="",
        description="Formatted, deduplicated context block prepared for Stage 6 LLM prompt injection",
    )
    total_context_characters: int = Field(
        default=0,
        description="Character count of the prepared context block",
    )
    sources: List[SourceReference] = Field(
        default_factory=list,
        description="Summary source citations corresponding to the context chunks",
    )

