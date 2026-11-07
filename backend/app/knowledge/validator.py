"""Document validation for Knowledge Base uploads, enforcing security and type constraints."""

import re
from pathlib import Path
from typing import Tuple
from app.knowledge.exceptions import (
    CorruptedDocumentException,
    FileSizeExceededException,
    InvalidFileTypeException,
)

# Supported extensions for Phase 2 Stage 1
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

# Known binary executable / archive signatures that must be rejected
DISALLOWED_BINARY_HEADERS = [
    b"MZ",  # Windows PE / Executable
    b"\x7fELF",  # Linux ELF Binary
    b"PK\x03\x04",  # ZIP archive (including JAR, DOCX disguised)
    b"7z\xbc\xaf\x27\x1c",  # 7-Zip
    b"\x1f\x8b",  # GZIP
]


def sanitize_filename(filename: str) -> str:
    """
    Sanitize an uploaded filename to prevent directory traversal and filesystem attacks.
    Strips directory separators, handles stem/suffix properly, and removes trailing underscores before the extension.
    """
    if not filename:
        return "unnamed_document.txt"

    # Extract base name ignoring any client-supplied path
    base_name = Path(filename).name
    p = Path(base_name)
    stem = p.stem
    suffix = p.suffix.lower()

    # Sanitize stem: allow alphanumeric, dots, dashes, underscores
    clean_stem = re.sub(r"[^\w\.\-]", "_", stem)
    clean_stem = re.sub(r"_{2,}", "_", clean_stem)
    clean_stem = clean_stem.strip("._")

    if not clean_stem:
        clean_stem = "document"

    clean_suffix = re.sub(r"[^\w\.]", "", suffix)
    return f"{clean_stem}{clean_suffix}"


def validate_document_upload(
    filename: str,
    content: bytes,
    max_size_bytes: int = 15 * 1024 * 1024,
) -> Tuple[str, str]:
    """
    Validate an uploaded document's metadata, extension, size, and magic byte signatures.

    Returns:
        Tuple of (sanitized_filename, normalized_file_type) where file_type is 'pdf' or 'txt'.

    Raises:
        InvalidFileTypeException: If extension is unsupported or signature is mismatched.
        FileSizeExceededException: If file is empty or exceeds max_size_bytes.
        CorruptedDocumentException: If file is malformed.
    """
    # 1. Sanitize filename
    clean_filename = sanitize_filename(filename)
    extension = Path(clean_filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise InvalidFileTypeException(
            f"Unsupported file format '{extension}'. Only PDF (.pdf) and Text (.txt) files are supported.",
            details={"filename": clean_filename, "supported": list(SUPPORTED_EXTENSIONS)},
        )

    # 2. File size validation
    file_size = len(content)
    if file_size == 0:
        raise FileSizeExceededException(
            "Uploaded file is empty (0 bytes).",
            details={"filename": clean_filename, "file_size": 0},
        )

    if file_size > max_size_bytes:
        limit_mb = max_size_bytes / (1024 * 1024)
        actual_mb = file_size / (1024 * 1024)
        raise FileSizeExceededException(
            f"File size exceeds maximum allowed limit ({actual_mb:.2f}MB > {limit_mb:.1f}MB).",
            details={"filename": clean_filename, "actual_bytes": file_size, "limit_bytes": max_size_bytes},
        )

    # 3. Format-specific magic byte and content validation
    if extension == ".pdf":
        # Standard PDF files begin with '%PDF-'
        if not content.startswith(b"%PDF-"):
            raise InvalidFileTypeException(
                "Invalid PDF document: missing standard '%PDF-' magic header signature.",
                details={"filename": clean_filename},
            )
        file_type = "pdf"

    elif extension == ".txt":
        # Check against dangerous binary signatures
        for sig in DISALLOWED_BINARY_HEADERS:
            if content.startswith(sig):
                raise InvalidFileTypeException(
                    "Invalid text file: binary executable or archive header signature detected.",
                    details={"filename": clean_filename},
                )
        # Check for binary null bytes
        if b"\x00" in content[:2048]:
            raise InvalidFileTypeException(
                "Invalid text file: binary null bytes detected in file content.",
                details={"filename": clean_filename},
            )
        file_type = "txt"

    else:
        raise InvalidFileTypeException(f"Unsupported file extension '{extension}'.")

    return clean_filename, file_type
