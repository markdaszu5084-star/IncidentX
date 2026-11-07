"""
Deterministic Multi-Format Log Parser.

Parses unstructured, structured (JSON), syslog, and application log lines into
strongly-typed LogEvent instances with deterministic citation IDs (e.g. L0001, L0002).
"""

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from app.schemas.log import LogEvent, LogLevel


# Regex patterns for common timestamp formats
TIMESTAMP_PATTERNS = [
    # ISO 8601 / RFC 3339: 2026-11-06T22:30:15.123Z or 2026-11-06 22:30:15,123
    (
        re.compile(r"^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"),
        ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%d %H:%M:%S"],
    ),
    # Syslog format: Nov  6 22:30:15 or Nov 06 22:30:15
    (
        re.compile(r"^([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})"),
        ["%b %d %H:%M:%S", "%b  %d %H:%M:%S"],
    ),
    # Apache/Nginx bracketed format: [06/Nov/2026:22:30:15 +0000]
    (
        re.compile(r"\[(\d{2}/[A-Z][a-z]{2}/\d{4}:\d{2}:\d{2}:\d{2}\s+[+-]\d{4})\]"),
        ["%d/%b/%Y:%H:%M:%S %z"],
    ),
]

# Standard application log regex pattern:
# Example: 2026-11-06 22:30:15.123 [ERROR] [auth-service] Connection timeout to db:5432
APP_LOG_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s+"
    r"(?:\[?(?P<level>[A-Za-z]+)\]?)\s+"
    r"(?:\[?(?P<service>[A-Za-z0-9_\-\.]+)(?:\]?|:))\s+"
    r"(?P<message>.*)$"
)

# HTTP Access Log Pattern (Nginx/Apache Combined)
# 192.168.1.10 - - [06/Nov/2026:22:30:15 +0000] "GET /api/v1/users HTTP/1.1" 500 234
ACCESS_LOG_PATTERN = re.compile(
    r"^(?P<client_ip>\S+)\s+\S+\s+(?P<user>\S+)\s+\[(?P<timestamp>[^\]]+)\]\s+"
    r"\"(?P<method>[A-Z]+)\s+(?P<path>\S+)(?:\s+(?P<protocol>[^\"]+))?\"\s+"
    r"(?P<status_code>\d{3})\s+(?P<bytes>\S+)(?:\s+\"(?P<referrer>[^\"]*)\")?(?:\s+\"(?P<user_agent>[^\"]*)\")?"
)

# Level keyword search in unstructured text
LEVEL_KEYWORD_PATTERN = re.compile(
    r"\b(?P<level>CRITICAL|FATAL|PANIC|ERROR|ERR|WARNING|WARN|INFO|DEBUG|TRACE)\b",
    re.IGNORECASE,
)


def parse_timestamp_str(raw_ts: str) -> Optional[datetime]:
    """Deterministically parse a timestamp string into datetime."""
    if not raw_ts:
        return None

    clean = raw_ts.strip().strip("[]")
    
    # Fast path: Try standard ISO parsing
    try:
        if clean.endswith("Z"):
            clean = clean[:-1] + "+00:00"
        return datetime.fromisoformat(clean)
    except Exception:
        pass

    for _, formats in TIMESTAMP_PATTERNS:
        for fmt in formats:
            try:
                dt = datetime.strptime(clean, fmt)
                # If year is missing (syslog), default to current year
                if dt.year == 1900:
                    dt = dt.replace(year=datetime.now(timezone.utc).year)
                return dt
            except ValueError:
                continue

    return None


def parse_json_log_line(line: str, line_no: int, event_id: str) -> Optional[LogEvent]:
    """Parse a single JSON log line."""
    trimmed = line.strip()
    if not (trimmed.startswith("{") and trimmed.endswith("}")):
        return None

    try:
        data = json.loads(trimmed)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    # Extract level
    level_raw = (
        data.get("level")
        or data.get("severity")
        or data.get("loglevel")
        or data.get("log_level")
        or data.get("status")
    )
    level = LogLevel.normalize(str(level_raw) if level_raw is not None else None)

    # Extract timestamp
    raw_ts = (
        data.get("timestamp")
        or data.get("time")
        or data.get("@timestamp")
        or data.get("datetime")
        or data.get("date")
    )
    parsed_ts = parse_timestamp_str(str(raw_ts)) if raw_ts else None

    # Extract service
    service = (
        data.get("service")
        or data.get("service_name")
        or data.get("app")
        or data.get("application")
        or data.get("component")
        or "unknown-service"
    )

    # Extract message
    message = (
        data.get("message")
        or data.get("msg")
        or data.get("error")
        or data.get("description")
        or json.dumps(data)
    )

    # Remaining attributes
    attributes = {k: v for k, v in data.items() if k not in ("level", "severity", "timestamp", "time", "@timestamp", "service", "message", "msg")}

    return LogEvent(
        id=event_id,
        line_number=line_no,
        timestamp=parsed_ts,
        raw_timestamp=str(raw_ts) if raw_ts else None,
        level=level,
        service=str(service),
        message=str(message),
        raw_text=line,
        attributes=attributes,
    )


def parse_access_log_line(line: str, line_no: int, event_id: str) -> Optional[LogEvent]:
    """Parse HTTP access log line (Nginx/Apache)."""
    match = ACCESS_LOG_PATTERN.match(line)
    if not match:
        return None

    groups = match.groupdict()
    status_code = int(groups["status_code"])
    raw_ts = groups["timestamp"]
    parsed_ts = parse_timestamp_str(raw_ts)

    # Determine severity based on HTTP status code
    if status_code >= 500:
        level = LogLevel.ERROR
    elif status_code >= 400:
        level = LogLevel.WARN
    else:
        level = LogLevel.INFO

    message = f"HTTP {groups['method']} {groups['path']} returned status {status_code}"
    attributes = {
        "client_ip": groups.get("client_ip"),
        "method": groups.get("method"),
        "path": groups.get("path"),
        "status_code": status_code,
        "bytes_sent": groups.get("bytes"),
    }

    return LogEvent(
        id=event_id,
        line_number=line_no,
        timestamp=parsed_ts,
        raw_timestamp=raw_ts,
        level=level,
        service="web-server",
        message=message,
        raw_text=line,
        attributes=attributes,
    )


def parse_app_log_line(line: str, line_no: int, event_id: str) -> Optional[LogEvent]:
    """Parse standard timestamped application log lines."""
    match = APP_LOG_PATTERN.match(line)
    if not match:
        return None

    groups = match.groupdict()
    raw_ts = groups["timestamp"]
    parsed_ts = parse_timestamp_str(raw_ts)
    level = LogLevel.normalize(groups["level"])
    service = groups["service"].strip("[]:")
    message = groups["message"].strip()

    return LogEvent(
        id=event_id,
        line_number=line_no,
        timestamp=parsed_ts,
        raw_timestamp=raw_ts,
        level=level,
        service=service,
        message=message,
        raw_text=line,
        attributes={},
    )


def parse_unstructured_line(line: str, line_no: int, event_id: str) -> LogEvent:
    """Fallback parser for unstructured or arbitrary log lines."""
    raw_ts = None
    parsed_ts = None

    # Search for leading timestamp
    for pattern, _ in TIMESTAMP_PATTERNS:
        match = pattern.search(line)
        if match:
            raw_ts = match.group(1)
            parsed_ts = parse_timestamp_str(raw_ts)
            break

    # Search for log level keyword
    level_match = LEVEL_KEYWORD_PATTERN.search(line)
    if level_match:
        level = LogLevel.normalize(level_match.group("level"))
    else:
        level = LogLevel.UNKNOWN

    return LogEvent(
        id=event_id,
        line_number=line_no,
        timestamp=parsed_ts,
        raw_timestamp=raw_ts,
        level=level,
        service="application",
        message=line.strip(),
        raw_text=line,
        attributes={},
    )


def parse_log_content(content: str) -> List[LogEvent]:
    """
    Parse full log text into a list of structured LogEvents with deterministic citation IDs.
    Citation IDs are formatted sequentially as L0001, L0002, etc.
    """
    lines = content.splitlines()
    events: List[LogEvent] = []

    for idx, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        citation_id = f"L{idx:04d}"

        # 1. Try JSON parser
        event = parse_json_log_line(line, idx, citation_id)
        if event:
            events.append(event)
            continue

        # 2. Try App Log pattern
        event = parse_app_log_line(line, idx, citation_id)
        if event:
            events.append(event)
            continue

        # 3. Try HTTP access log pattern
        event = parse_access_log_line(line, idx, citation_id)
        if event:
            events.append(event)
            continue

        # 4. Fallback unstructured parser
        event = parse_unstructured_line(line, idx, citation_id)
        events.append(event)

    return events

