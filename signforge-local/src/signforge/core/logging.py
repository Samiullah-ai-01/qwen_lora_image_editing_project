"""
Structured JSON logging for SignForge.

Provides consistent, machine-readable logs with contextual information.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog
from structlog.typing import EventDict, WrappedLogger


class JSONRenderer:
    """Custom JSON renderer for structlog."""
    
    def __call__(self, logger: WrappedLogger, method_name: str, event_dict: EventDict) -> str:
        """Render the event dict as a JSON string."""
        # Ensure timestamp is present
        if "timestamp" not in event_dict:
            event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Add log level
        event_dict["level"] = method_name.upper()
        
        # Convert non-serializable objects
        for key, value in list(event_dict.items()):
            if isinstance(value, Path):
                event_dict[key] = str(value)
            elif isinstance(value, Exception):
                event_dict[key] = f"{type(value).__name__}: {value}"
            elif hasattr(value, "__dict__") and not isinstance(value, (str, int, float, bool, list, dict)):
                event_dict[key] = str(value)
        
        return json.dumps(event_dict, default=str, ensure_ascii=False)


class TextRenderer:
    """Human-readable text renderer for development."""
    
    def __call__(self, logger: WrappedLogger, method_name: str, event_dict: EventDict) -> str:
        """Render the event dict as readable text."""
        timestamp = event_dict.pop("timestamp", datetime.now(timezone.utc).isoformat())
        level = method_name.upper()
        event = event_dict.pop("event", "")
        component = event_dict.pop("component", "signforge")
        
        # Build message
        parts = [f"{timestamp} [{level:8}] {component}: {event}"]
        
        # Add extra fields
        if event_dict:
            extras = " | ".join(f"{k}={v}" for k, v in event_dict.items())
            parts.append(f"  {extras}")
        
        return "\n".join(parts)


def add_component(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add component name to the event dict."""
    if "component" not in event_dict:
        # Extract component from logger name
        record = event_dict.get("_record")
        if record and hasattr(record, "name"):
            event_dict["component"] = record.name.replace("signforge.", "")
        else:
            event_dict["component"] = "signforge"
    return event_dict


def add_request_id(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add request ID if available in context."""
    # This will be set by the request context
    import contextvars
    request_id_var = contextvars.ContextVar("request_id", default=None)
    request_id = request_id_var.get()
    if request_id:
        event_dict["request_id"] = request_id
    return event_dict


_configured = False
_log_file_handler: Optional[logging.FileHandler] = None


def configure_logging(
    level: str = "INFO",
    format: str = "json",
    log_file: Optional[Path] = None,
    rotate: bool = True,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """
    Configure the logging system.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format: Output format ('json' or 'text')
        log_file: Optional path to log file
        rotate: Whether to rotate log files
        max_bytes: Maximum bytes before rotation
        backup_count: Number of backup files to keep
    """
    global _configured, _log_file_handler
    
    if _configured:
        return
    
    # Select renderer
    if format == "json":
        renderer = JSONRenderer()
    else:
        renderer = TextRenderer()
    
    # Configure structlog
    processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        add_component,
        add_request_id,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        renderer,
    ]
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        if rotate:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
        else:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
        
        file_handler.setLevel(getattr(logging, level.upper()))
        root_logger.addHandler(file_handler)
        _log_file_handler = file_handler
    
    _configured = True
    
    # Log configuration
    logger = get_logger(__name__)
    logger.info(
        "logging_configured",
        level=level,
        format=format,
        log_file=str(log_file) if log_file else None,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured structlog logger
    """
    # Configure with defaults if not already configured
    if not _configured:
        configure_logging()
    
    return structlog.get_logger(name)


class LogContext:
    """Context manager for adding fields to all logs within scope."""
    
    def __init__(self, **fields: Any) -> None:
        """Initialize with context fields."""
        self.fields = fields
        self._token = None
    
    def __enter__(self) -> "LogContext":
        """Enter the context."""
        self._token = structlog.contextvars.bind_contextvars(**self.fields)
        return self
    
    def __exit__(self, *args: Any) -> None:
        """Exit the context."""
        if self._token:
            structlog.contextvars.unbind_contextvars(*self.fields.keys())


def log_to_file(
    message: dict,
    file_path: Path,
    append: bool = True,
) -> None:
    """
    Write a structured log message directly to a file.
    
    Useful for request logs, metrics, etc.
    
    Args:
        message: Dict to write as JSON
        file_path: Path to the log file
        append: Whether to append or overwrite
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure timestamp
    if "timestamp" not in message:
        message["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    mode = "a" if append else "w"
    with open(file_path, mode, encoding="utf-8") as f:
        f.write(json.dumps(message, default=str) + "\n")
