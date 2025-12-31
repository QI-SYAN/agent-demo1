"""Logging utilities for the agent."""

from __future__ import annotations
import logging
from typing import Optional
import structlog

def configure_logging(level: int = logging.INFO, json_output: bool = False) -> None:
    """Configure structlog + stdlib logging."""

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        timestamper,
        structlog.processors.add_log_level,
    ]

    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(level=level)

def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """Return a pre-configured logger."""

    return structlog.get_logger(name)