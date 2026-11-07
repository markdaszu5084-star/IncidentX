"""File and content validation for log ingestion, enforcing security boundaries."""

from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


ALLOWED_EXTENSIONS = {".log", ".txt", ".json", ".csv"}
BINARY_SIGNATURES = [
    b"MZ",  # Windows Executable / DLL
    b"\x7fELF",  # Linux ELF Executable
    b"PK\x03\x04",  # ZIP archive / APK / JAR
    b"\x1f\x8b",  # GZIP compressed archive
    b"7z\xbc\xaf\x27\x1c",  # 7-Zip
    b"%PDF",  # PDF document
]


class ValidationResult(BaseModel):
    """Result of file and content validation."""

    is_valid: bool = Field(description="True if file passed all security checks")
    error_message: Optional[str] = Field(
        default=None, description="Detailed explanation if validation failed"
    )
    cleaned_text: Optional[str] = Field(
        default=None, description="Decoded and normalized text content if valid"
    )
    byte_size: int = Field(default=0, description="Size in bytes")
    line_count: int = Field(default=0, description="Total line count in the content")


def validate_log_file(
    filename: str,
    content: bytes,
    max_size_bytes: int = 10 * 1024 * 1024,
    allowed_extensions: Optional[List[str]] = None,
) -> ValidationResult:
    """
    Validate uploaded file metadata and raw byte content against security constraints.

    Checks:
    1. Extension whitelist.
    2. Size limits (non-empty and below max threshold).
    3. Binary magic byte signatures (preventing executable uploads).
    4. Text decodability (rejecting binary or corrupt payloads).
    """
    if allowed_extensions is None:
        valid_exts = ALLOWED_EXTENSIONS
    else:
        valid_exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in allowed_extensions}

    file_size = len(content)

    # 1. Check size limits
    if file_size == 0:
        return ValidationResult(
            is_valid=False,
            error_message="Uploaded file is empty (0 bytes).",
            byte_size=0,
            line_count=0,
        )

    if file_size > max_size_bytes:
        limit_mb = max_size_bytes / (1024 * 1024)
        actual_mb = file_size / (1024 * 1024)
        return ValidationResult(
            is_valid=False,
            error_message=f"File exceeds maximum allowed size ({actual_mb:.2f}MB > {limit_mb:.1f}MB).",
            byte_size=file_size,
            line_count=0,
        )

    # 2. Check filename extension
    ext = Path(filename).suffix.lower()
    if not ext or ext not in valid_exts:
        return ValidationResult(
            is_valid=False,
            error_message=(
                f"Disallowed file extension '{ext}'. "
                f"Permitted extensions: {', '.join(sorted(valid_exts))}"
            ),
            byte_size=file_size,
            line_count=0,
        )

    # 3. Check for binary magic headers
    for sig in BINARY_SIGNATURES:
        if content.startswith(sig):
            return ValidationResult(
                is_valid=False,
                error_message="Invalid file content: binary executable or archive detected.",
                byte_size=file_size,
                line_count=0,
            )

    # Null bytes check: text log files should not contain null bytes
    if b"\x00" in content[:4096]:
        return ValidationResult(
            is_valid=False,
            error_message="Invalid file content: binary null bytes detected in header.",
            byte_size=file_size,
            line_count=0,
        )

    # 4. Attempt text decoding
    try:
        decoded_text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            # Fallback to Latin-1 if valid ascii/extended
            decoded_text = content.decode("latin-1")
        except Exception:
            return ValidationResult(
                is_valid=False,
                error_message="File cannot be decoded as valid UTF-8 or text.",
                byte_size=file_size,
                line_count=0,
            )

    # Strip BOM if present
    decoded_text = decoded_text.lstrip("\ufeff")

    # Compute line count
    lines = decoded_text.splitlines()
    line_count = len(lines)

    if line_count == 0 or not any(l.strip() for l in lines):
        return ValidationResult(
            is_valid=False,
            error_message="Uploaded file contains no readable text lines.",
            byte_size=file_size,
            line_count=0,
        )

    return ValidationResult(
        is_valid=True,
        cleaned_text=decoded_text,
        byte_size=file_size,
        line_count=line_count,
    )

