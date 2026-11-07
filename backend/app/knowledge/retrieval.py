"""
Semantic Retrieval Engine for IncidentX Knowledge Base — Stage 5.

Orchestrates natural-language query processing, query vector embedding generation,
FAISS cosine similarity search, relevance ranking, result filtering, deduplication,
and structured context synthesis for downstream LLM reasoning (Stage 6).
"""

import logging
from typing import List, Optional, Tuple

from app.config import get_settings
from app.knowledge.embeddings import EmbeddingService, get_embedding_service
from app.knowledge.vector_store import FaissVectorStore
from app.schemas.retrieval import (
    RetrievalRequest,
    RetrievalResponse,
    SearchResultItem,
    SourceReference,
)

logger = logging.getLogger(__name__)


class SemanticRetrievalEngine:
    """Intelligent semantic search and context preparation engine using FAISS."""

    def __init__(
        self,
        vector_store: Optional[FaissVectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.settings = get_settings()
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or FaissVectorStore()

    @staticmethod
    def normalize_query(query: str) -> str:
        """Validate and normalize natural-language query string.

        Parameters
        ----------
        query : str
            Raw user input query.

        Returns
        -------
        str
            Cleaned, single-spaced normalized query string.

        Raises
        ------
        ValueError
            If query is empty or composed exclusively of whitespace.
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query string cannot be empty or non-string.")

        cleaned = " ".join(query.strip().split())
        if not cleaned:
            raise ValueError("Query string cannot be empty or whitespace.")

        if len(cleaned) > 4000:
            raise ValueError("Query string exceeds maximum length of 4,000 characters.")

        return cleaned

    def search(self, request: RetrievalRequest) -> RetrievalResponse:
        """Execute semantic search against the FAISS index and prepare LLM context.

        Parameters
        ----------
        request : RetrievalRequest
            Search configuration with query, top_k, score_threshold, and context limits.

        Returns
        -------
        RetrievalResponse
            Ranked search results and aggregated prompt-ready context text.
        """
        # 1. Normalize query
        clean_query = self.normalize_query(request.query)

        # 2. Check if vector store has any indexed documents
        if self.vector_store.total_vectors == 0:
            logger.info("Search requested on empty vector store for query: '%s'", clean_query)
            return RetrievalResponse(
                status="success",
                query=clean_query,
                total_results=0,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                results=[],
                context_text="",
                total_context_characters=0,
                sources=[],
            )

        # 3. Generate query vector embedding using Stage 3 embedding service
        query_vector = self.embedding_service.embed_query(clean_query)

        # 4. Search FAISS index with inner product (cosine similarity on unit vectors)
        raw_matches = self.vector_store.search_vectors(
            query_vector=query_vector,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            document_id_filter=request.document_id,
        )

        # 5. Package matching chunks into ranked SearchResultItems
        results: List[SearchResultItem] = []
        for meta, score in raw_matches:
            results.append(
                SearchResultItem(
                    chunk_id=meta.chunk_id,
                    document_id=meta.document_id,
                    filename=meta.filename,
                    chunk_index=meta.chunk_index,
                    page_number=meta.page_number,
                    text=meta.text,
                    similarity_score=round(float(score), 4),
                    character_count=meta.character_count,
                    word_count=meta.word_count,
                    metadata=meta.metadata,
                )
            )

        # 6. Synthesize structured, deduplicated context for Stage 6 LLM consumption
        context_text, sources = self.prepare_context(
            results=results,
            max_characters=request.max_context_characters,
        )

        logger.info(
            "Semantic search completed for '%s': retrieved %d chunks (threshold=%.2f, top_k=%d).",
            clean_query[:50],
            len(results),
            request.score_threshold,
            request.top_k,
        )

        return RetrievalResponse(
            status="success",
            query=clean_query,
            total_results=len(results),
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            results=results,
            context_text=context_text,
            total_context_characters=len(context_text),
            sources=sources,
        )

    def prepare_context(
        self,
        results: List[SearchResultItem],
        max_characters: int = 8000,
    ) -> Tuple[str, List[SourceReference]]:
        """Combine relevant chunks into a structured prompt context block.

        Features:
        - Deduplicates chunks by chunk_id.
        - Preserves provenance references (Document, Chunk ID, Page, Score).
        - Enforces character limit budget.
        - Provides citations list matching the included context chunks.

        Parameters
        ----------
        results : List[SearchResultItem]
            Ranked list of search results.
        max_characters : int
            Character budget for the aggregated context string.

        Returns
        -------
        Tuple[str, List[SourceReference]]
            (Formatted context string, list of included source references)
        """
        if not results:
            return "", []

        seen_chunks = set()
        context_blocks: List[str] = []
        sources: List[SourceReference] = []
        current_chars = 0

        for r in results:
            if r.chunk_id in seen_chunks:
                continue
            seen_chunks.add(r.chunk_id)

            page_str = f", Page {r.page_number}" if r.page_number is not None else ""
            header = (
                f"--- SOURCE: {r.filename} (Chunk: {r.chunk_id}{page_str}, "
                f"Relevance: {r.similarity_score:.4f}) ---"
            )
            block = f"{header}\n{r.text}\n"

            # Check character budget
            if current_chars + len(block) > max_characters and context_blocks:
                logger.info(
                    "Context character budget reached (%d / %d chars). Stopping chunk aggregation.",
                    current_chars,
                    max_characters,
                )
                break

            context_blocks.append(block)
            current_chars += len(block)

            sources.append(
                SourceReference(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    filename=r.filename,
                    page_number=r.page_number,
                    similarity_score=r.similarity_score,
                )
            )

        assembled_context = "\n".join(context_blocks).strip()
        return assembled_context, sources

