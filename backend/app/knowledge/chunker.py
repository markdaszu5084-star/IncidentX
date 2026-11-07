"""
Intelligent Text Chunking Module using LangChain's RecursiveCharacterTextSplitter.

Transforms extracted text into structured, overlapping chunks optimized for RAG
retrieval while preserving incident context, page numbers, and document metadata.
"""

from typing import Any, Dict, List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import get_settings
from app.knowledge.extractor import ExtractedDocumentContent
from app.schemas.knowledge import ChunkingConfig, DocumentChunk


# Separators prioritized for incident runbooks and technical documents:
# 1. Double newlines (paragraphs, markdown sections, step blocks)
# 2. Single newlines (command steps, log lines, list items)
# 3. Sentence boundaries (. ? !)
# 4. Clause delimiters (; ,)
# 5. Word boundaries (spaces)
# 6. Fallback characters
RUNBOOK_SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    "; ",
    ", ",
    " ",
    "",
]


class IntelligentChunker:
    """Configurable text chunker for incident response documentation."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        settings = get_settings()
        self.chunk_size = chunk_size if chunk_size is not None else settings.default_chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.default_chunk_overlap

        # Validate configuration
        config = ChunkingConfig(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=RUNBOOK_SEPARATORS,
            keep_separator=True,
            strip_whitespace=True,
        )

    def chunk_text(
        self,
        text: str,
        document_id: str,
        filename: str,
        page_number: Optional[int] = None,
        start_index: int = 0,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Split a block of text into structured DocumentChunks.
        """
        cleaned = text.strip() if text else ""
        if not cleaned:
            return []

        # If text is smaller than or equal to chunk size, return single chunk
        if len(cleaned) <= self.chunk_size:
            chunk_id = f"{document_id}_c{start_index:04d}"
            words = cleaned.split()
            meta = {
                "source": filename,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            }
            if page_number is not None:
                meta["page_number"] = page_number
            if custom_metadata:
                meta.update(custom_metadata)

            return [
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    filename=filename,
                    chunk_index=start_index,
                    text=cleaned,
                    character_count=len(cleaned),
                    word_count=len(words),
                    page_number=page_number,
                    metadata=meta,
                )
            ]

        # Use LangChain splitter
        raw_chunks = self.splitter.split_text(cleaned)
        chunks: List[DocumentChunk] = []

        current_index = start_index
        for raw_chunk in raw_chunks:
            chunk_str = raw_chunk.strip()
            # Avoid producing empty chunks
            if not chunk_str:
                continue

            chunk_id = f"{document_id}_c{current_index:04d}"
            words = chunk_str.split()
            meta = {
                "source": filename,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            }
            if page_number is not None:
                meta["page_number"] = page_number
            if custom_metadata:
                meta.update(custom_metadata)

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    filename=filename,
                    chunk_index=current_index,
                    text=chunk_str,
                    character_count=len(chunk_str),
                    word_count=len(words),
                    page_number=page_number,
                    metadata=meta,
                )
            )
            current_index += 1

        return chunks

    def chunk_extracted_content(
        self,
        document_id: str,
        filename: str,
        content: ExtractedDocumentContent,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Split complete extracted document content into chunks.
        Preserves page boundaries for multi-page documents where page text is available.
        """
        if not content.text or not content.text.strip():
            return []

        all_chunks: List[DocumentChunk] = []

        # If page-level details with text are present (e.g. from PDF)
        has_page_texts = (
            content.page_details
            and any(p.get("text") and p.get("text").strip() for p in content.page_details)
        )

        if has_page_texts:
            global_index = 0
            for page in content.page_details:
                page_text = page.get("text", "").strip()
                page_num = page.get("page_number")
                if not page_text:
                    continue

                page_chunks = self.chunk_text(
                    text=page_text,
                    document_id=document_id,
                    filename=filename,
                    page_number=page_num,
                    start_index=global_index,
                    custom_metadata=custom_metadata,
                )
                all_chunks.extend(page_chunks)
                global_index += len(page_chunks)

        else:
            # Single stream text (e.g. TXT file)
            all_chunks = self.chunk_text(
                text=content.text,
                document_id=document_id,
                filename=filename,
                page_number=None,
                start_index=0,
                custom_metadata=custom_metadata,
            )

        return all_chunks

