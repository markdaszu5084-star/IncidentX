"""
Deterministic Error Signature Extraction & Statistical Analysis.

Groups recurring error patterns, normalizes dynamic tokens (IPs, UUIDs, hex addresses),
and generates deterministic statistical summaries with citation references.
"""

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional
from app.schemas.log import (
    ErrorSignature,
    LogEvent,
    LogLevel,
    ParsedLogSummary,
    ServiceStats,
)

# Regex normalization rules for error signatures
HEX_ADDR_REGEX = re.compile(r"\b0x[0-9a-fA-F]+\b")
UUID_REGEX = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
IPV4_REGEX = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b")
TIMESTAMP_INLINE_REGEX = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\b")
NUMERIC_ID_REGEX = re.compile(r"\b(?<=id[=:])\d+\b", re.IGNORECASE)
MULTI_SPACE_REGEX = re.compile(r"\s+")


def normalize_error_signature(message: str) -> str:
    """
    Deterministically sanitize transient data from log messages to extract canonical error patterns.
    Replaces UUIDs, memory addresses, IP addresses, and timestamps with stable placeholders.
    """
    text = message.strip()
    text = HEX_ADDR_REGEX.sub("<HEX>", text)
    text = UUID_REGEX.sub("<UUID>", text)
    text = IPV4_REGEX.sub("<IP:PORT>", text)
    text = TIMESTAMP_INLINE_REGEX.sub("<TIMESTAMP>", text)
    text = NUMERIC_ID_REGEX.sub("<ID>", text)
    text = MULTI_SPACE_REGEX.sub(" ", text).strip()
    # Limit length to keep signature clean
    return text[:250]


def analyze_log_events(events: List[LogEvent]) -> ParsedLogSummary:
    """
    Perform deterministic statistical analysis across all parsed log events.
    Computes global counts, service distributions, timeline span, and clustered error signatures.
    """
    if not events:
        return ParsedLogSummary()

    total_events = len(events)
    error_count = 0
    critical_count = 0
    warning_count = 0
    info_count = 0

    service_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "error": 0, "warn": 0}
    )

    signature_tracker: Dict[str, Dict[str, Any]] = {}
    critical_event_ids: List[str] = []

    valid_timestamps = []

    for event in events:
        # Service tracking
        svc = event.service
        service_counts[svc]["total"] += 1

        # Timestamp tracking
        if event.timestamp:
            valid_timestamps.append(event.timestamp)

        # Level classification
        if event.level == LogLevel.CRITICAL:
            critical_count += 1
            service_counts[svc]["error"] += 1
            critical_event_ids.append(event.id)
        elif event.level == LogLevel.ERROR:
            error_count += 1
            service_counts[svc]["error"] += 1
            if len(critical_event_ids) < 15:
                critical_event_ids.append(event.id)
        elif event.level == LogLevel.WARN:
            warning_count += 1
            service_counts[svc]["warn"] += 1
        elif event.level in (LogLevel.INFO, LogLevel.DEBUG):
            info_count += 1

        # Error signature clustering (for ERROR and CRITICAL events)
        if event.level in (LogLevel.ERROR, LogLevel.CRITICAL):
            sig = normalize_error_signature(event.message)
            if sig not in signature_tracker:
                signature_tracker[sig] = {
                    "count": 0,
                    "level": event.level,
                    "services": set(),
                    "first_seen": event.timestamp,
                    "last_seen": event.timestamp,
                    "sample_event_ids": [],
                }

            entry = signature_tracker[sig]
            entry["count"] += 1
            entry["services"].add(event.service)
            if event.timestamp:
                if not entry["first_seen"] or event.timestamp < entry["first_seen"]:
                    entry["first_seen"] = event.timestamp
                if not entry["last_seen"] or event.timestamp > entry["last_seen"]:
                    entry["last_seen"] = event.timestamp

            # Keep up to 3 citation IDs per signature
            if len(entry["sample_event_ids"]) < 3:
                entry["sample_event_ids"].append(event.id)

            # Elevate signature level if critical
            if event.level == LogLevel.CRITICAL:
                entry["level"] = LogLevel.CRITICAL

    # Timeline bounds
    time_start = min(valid_timestamps) if valid_timestamps else None
    time_end = max(valid_timestamps) if valid_timestamps else None
    duration_seconds = (
        (time_end - time_start).total_seconds()
        if time_start and time_end
        else None
    )

    # Compute global error rate
    total_failures = error_count + critical_count
    global_error_rate = round((total_failures / total_events) * 100.0, 2) if total_events > 0 else 0.0

    # Build ServiceStats list
    services: List[ServiceStats] = []
    for svc, stats in service_counts.items():
        svc_total = stats["total"]
        svc_errors = stats["error"]
        svc_error_rate = round((svc_errors / svc_total) * 100.0, 2) if svc_total > 0 else 0.0
        services.append(
            ServiceStats(
                service=svc,
                total_events=svc_total,
                error_count=svc_errors,
                warn_count=stats["warn"],
                error_rate=svc_error_rate,
            )
        )
    # Sort services by error count descending
    services.sort(key=lambda s: s.error_count, reverse=True)

    # Build ErrorSignature list
    top_signatures: List[ErrorSignature] = []
    for sig_pattern, data in signature_tracker.items():
        top_signatures.append(
            ErrorSignature(
                signature=sig_pattern,
                count=data["count"],
                level=data["level"],
                services=sorted(list(data["services"])),
                first_seen=data["first_seen"],
                last_seen=data["last_seen"],
                sample_event_ids=data["sample_event_ids"],
            )
        )
    # Sort signatures by frequency descending
    top_signatures.sort(key=lambda s: s.count, reverse=True)

    return ParsedLogSummary(
        total_events=total_events,
        error_count=error_count,
        critical_count=critical_count,
        warning_count=warning_count,
        info_count=info_count,
        error_rate=global_error_rate,
        time_start=time_start,
        time_end=time_end,
        duration_seconds=duration_seconds,
        services=services,
        top_error_signatures=top_signatures[:10],  # Top 10 recurring error signatures
        critical_event_ids=critical_event_ids[:20],  # Top 20 citation IDs
    )
