"""Pydantic schemas for log events, parsed summaries, and statistical metrics."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LogLevel(str, Enum):
    """Standardized log severity levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    WARNING = "WARN"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    FATAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def normalize(cls, level_str: Optional[str]) -> "LogLevel":
        """Normalize various casing and alias strings into canonical LogLevel."""
        if not level_str:
            return cls.UNKNOWN
        clean = level_str.strip().upper()
        if clean in ("WARNING", "WARN"):
            return cls.WARN
        if clean in ("CRITICAL", "FATAL", "EMERG", "ALERT", "PANIC"):
            return cls.CRITICAL
        if clean in ("ERROR", "ERR"):
            return cls.ERROR
        if clean in ("INFO", "INFORMATIONAL", "NOTICE"):
            return cls.INFO
        if clean in ("DEBUG", "TRACE", "VERBOSE"):
            return cls.DEBUG
        return cls.UNKNOWN


class LogEvent(BaseModel):
    """Structured representation of an individual log event with deterministic citation ID."""

    id: str = Field(
        description="Deterministic citation identifier (e.g., L0001, L0002) cited by hypotheses"
    )
    line_number: int = Field(description="Line number in the original log source")
    timestamp: Optional[datetime] = Field(default=None, description="Parsed ISO timestamp")
    raw_timestamp: Optional[str] = Field(default=None, description="Raw unparsed timestamp string")
    level: LogLevel = Field(default=LogLevel.UNKNOWN, description="Normalized log level")
    service: str = Field(default="unknown-service", description="Originating service or component")
    message: str = Field(description="Cleaned log message content")
    raw_text: str = Field(description="Original unedited raw line text")
    attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted key-value attributes (e.g. status_code, trace_id, exception)",
    )


class ErrorSignature(BaseModel):
    """Normalized signature of recurring errors across log events."""

    signature: str = Field(description="Parameterized message pattern stripping dynamic IDs/ports")
    count: int = Field(default=1, description="Number of occurrences")
    level: LogLevel = Field(description="Highest severity level observed for this error pattern")
    services: List[str] = Field(
        default_factory=list, description="Services in which this signature appeared"
    )
    first_seen: Optional[datetime] = Field(default=None, description="Earliest timestamp")
    last_seen: Optional[datetime] = Field(default=None, description="Latest timestamp")
    sample_event_ids: List[str] = Field(
        default_factory=list,
        description="Sample deterministic log citation IDs exhibiting this pattern",
    )


class ServiceStats(BaseModel):
    """Aggregated operational metrics for an individual service."""

    service: str = Field(description="Service name")
    total_events: int = Field(default=0, description="Total log lines emitted")
    error_count: int = Field(default=0, description="Count of ERROR and CRITICAL logs")
    warn_count: int = Field(default=0, description="Count of WARN logs")
    error_rate: float = Field(
        default=0.0, description="Error percentage relative to service total (0.0 - 100.0)"
    )


class ParsedLogSummary(BaseModel):
    """Deterministic statistical summary of ingested log events."""

    total_events: int = Field(default=0, description="Total number of processed events")
    error_count: int = Field(default=0, description="Total ERROR events")
    critical_count: int = Field(default=0, description="Total CRITICAL/FATAL events")
    warning_count: int = Field(default=0, description="Total WARN events")
    info_count: int = Field(default=0, description="Total INFO/DEBUG events")
    error_rate: float = Field(
        default=0.0, description="Global error percentage (0.0 - 100.0)"
    )
    time_start: Optional[datetime] = Field(default=None, description="Earliest event timestamp")
    time_end: Optional[datetime] = Field(default=None, description="Latest event timestamp")
    duration_seconds: Optional[float] = Field(
        default=None, description="Total incident window span in seconds"
    )
    services: List[ServiceStats] = Field(
        default_factory=list, description="Service-by-service statistics"
    )
    top_error_signatures: List[ErrorSignature] = Field(
        default_factory=list, description="Top normalized error patterns sorted by frequency"
    )
    critical_event_ids: List[str] = Field(
        default_factory=list,
        description="Key deterministic IDs of critical/fatal events for evidence citation",
    )

