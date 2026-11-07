"""Text extraction and preprocessing engine for PDF and TXT documents."""

import io
import re
import unicodedata
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import pypdf
from app.knowledge.exceptions import CorruptedDocumentException


class ExtractedDocumentContent(BaseModel):
    """Container for extracted and normalized document text and metrics."""

    text: str = Field(description="Full extracted and cleaned plain text")
    character_count: int = Field(description="Total character count")
    word_count: int = Field(description="Total word count")
    page_count: Optional[int] = Field(default=None, description="Number of pages if PDF")
    page_details: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-page extracted text and metrics"
    )


def clean_text(raw_text: str) -> str:
    """
    Deterministically clean and normalize extracted document text for incident RAG.

    Steps:
    1. Unicode normalization (NFKC).
    2. Strip zero-width & non-printable control characters except newline and tab.
    3. Normalize line endings (CRLF / CR -> LF).
    4. Strip trailing whitespace from individual lines.
    5. Collapse redundant intra-line horizontal whitespace.
    6. Limit consecutive blank lines to at most 2 (preserving section breaks).
    """
    if not raw_text:
        return ""

    # 1. Unicode normalization
    text = unicodedata.normalize("NFKC", raw_text)

    # 2. Strip zero-width & non-printable control characters except newline and tab
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = "".join(ch for ch in text if ch in ("\n", "\t") or (ord(ch) >= 32 and ord(ch) != 127))

    # 3. Normalize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 4. Clean lines: strip trailing spaces while preserving intentional structure
    cleaned_lines = [re.sub(r"[ \t]+$", "", line) for line in text.split("\n")]
    text = "\n".join(cleaned_lines)

    # 5. Clean horizontal whitespace
    def clean_spaces(line: str) -> str:
        leading = len(line) - len(line.lstrip(" "))
        prefix = " " * min(leading, 4)
        rest = re.sub(r"[ \t]+", " ", line.lstrip(" "))
        return prefix + rest if line.strip() else ""

    text = "\n".join(clean_spaces(l) for l in text.split("\n"))

    # 6. Collapse excessive consecutive newlines (max 2 consecutive newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_from_txt(content: bytes) -> ExtractedDocumentContent:
    """
    Extract plain text from a text file (.txt).
    Tries UTF-8 decoding with Latin-1 fallback.
    """
    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            decoded = content.decode("latin-1")
        except Exception as e:
            raise CorruptedDocumentException(
                f"Failed to decode text document: {e}",
                details={"error": str(e)},
            )

    # Strip BOM
    decoded = decoded.lstrip("\ufeff")

    cleaned = clean_text(decoded)
    words = cleaned.split()

    return ExtractedDocumentContent(
        text=cleaned,
        character_count=len(cleaned),
        word_count=len(words),
        page_count=None,
        page_details=[{
            "page_number": None,
            "text": cleaned,
            "character_count": len(cleaned),
            "word_count": len(words),
        }],
    )


def extract_from_pdf(content: bytes) -> ExtractedDocumentContent:
    """
    Extract text page-by-page from a PDF document using pypdf.
    """
    try:
        reader = pypdf.PdfReader(io.BytesIO(content))
    except Exception as e:
        raise CorruptedDocumentException(
            f"Failed to read PDF document: {e}",
            details={"error": str(e)},
        )

    if reader.is_encrypted:
        try:
            # Attempt decrypt with empty password for standard protected PDFs
            reader.decrypt("")
        except Exception:
            raise CorruptedDocumentException(
                "PDF is encrypted or password-protected and cannot be processed.",
                details={"is_encrypted": True},
            )

    total_pages = len(reader.pages)
    if total_pages == 0:
        raise CorruptedDocumentException("PDF contains no readable pages.")

    page_texts: List[str] = []
    page_details: List[Dict[str, Any]] = []

    for page_idx, page in enumerate(reader.pages, start=1):
        try:
            raw_page_text = page.extract_text() or ""
        except Exception as e:
            raw_page_text = f"[Extraction error on page {page_idx}: {e}]"

        cleaned_page = clean_text(raw_page_text)
        page_texts.append(cleaned_page)
        page_details.append({
            "page_number": page_idx,
            "text": cleaned_page,
            "character_count": len(cleaned_page),
            "word_count": len(cleaned_page.split()),
        })

    full_text = "\n\n".join(t for t in page_texts if t)
    words = full_text.split()

    return ExtractedDocumentContent(
        text=full_text,
        character_count=len(full_text),
        word_count=len(words),
        page_count=total_pages,
        page_details=page_details,
    )


def extract_document_content(content: bytes, file_type: str) -> ExtractedDocumentContent:
    """
    Dispatcher to extract text and metrics from supported document types.
    """
    if file_type == "txt":
        return extract_from_txt(content)
    elif file_type == "pdf":
        return extract_from_pdf(content)
    else:
        raise CorruptedDocumentException(f"Unsupported file type '{file_type}' for extraction.")

