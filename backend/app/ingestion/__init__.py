"""Log ingestion, validation, parsing, and statistical analysis package."""

from app.ingestion.validator import validate_log_file, ValidationResult

__all__ = ["validate_log_file", "ValidationResult"]

