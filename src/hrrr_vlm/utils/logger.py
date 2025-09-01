"""Structured logging configuration."""

import logging
import os

import structlog


def configure_logger(
    log_level: str | int | None = None, *, enable_json: bool | None = None
) -> structlog.BoundLogger:
    """Configure structured logging.

    Args:
        log_level (`str`, `int`, optional): Logging level. If None, defaults to
            the LOG_LEVEL environment variable or INFO if not set.
        enable_json (`bool`, optional): If True, use JSON format for logs.
            If None, defaults to the LOG_JSON environment variable.

    Returns:
        `structlog.BoundLogger`: Configured logger instance.
    """
    use_json = (
        enable_json
        if enable_json is not None
        else os.getenv("JSON_LOGS", "false").strip().lower() in {"true", "1"}
    )
    log_level = (
        log_level
        if log_level is not None
        else os.getenv("LOG_LEVEL", "INFO").strip().upper()
    )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
            if use_json
            else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Set the log level on the root logger
    logging.getLogger().setLevel(log_level)

    return structlog.get_logger()
