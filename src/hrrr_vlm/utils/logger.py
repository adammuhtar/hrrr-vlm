"""Structured logging configuration.

Two responsibilities are kept separate on purpose:

* :func:`configure_logging` performs one-time, process-wide setup of structlog
  and the standard library root handler. Call it from entry points (scripts,
  ``__main__`` blocks, test fixtures) - never at library import time.
* :func:`get_logger` returns a cheap, lazily-bound logger and is safe to call
  at module scope before :func:`configure_logging` has run.

Both structlog events and standard library records (e.g. from third-party
libraries such as herbie, botocore, or matplotlib) are rendered through a
single :class:`structlog.stdlib.ProcessorFormatter` installed on the root
handler, so console and JSON output stay uniform for the whole process and
JSON mode emits machine-parseable lines only.
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TextIO

import structlog

if TYPE_CHECKING:
    from collections.abc import Sequence

    from structlog.types import Processor

__all__ = [
    "DEFAULT_SUPPRESSED_LIBS",
    "LogSettings",
    "LoggerProtocol",
    "configure_logging",
    "get_logger",
    "reset_logging",
]

# Canonical level names plus stdlib aliases (WARN, FATAL), excluding NOTSET.
_LEVEL_NAME_TO_NO: dict[str, int] = {
    name: no
    for name, no in logging.getLevelNamesMapping().items()
    if no != logging.NOTSET
}
_JSON_TRUTHY: frozenset[str] = frozenset({"true", "1", "yes", "on"})
_PRODUCTION_ENVIRONMENTS: frozenset[str] = frozenset({"prod", "production"})

# Libraries in this project's dependency stack that flood DEBUG/INFO.
DEFAULT_SUPPRESSED_LIBS: tuple[str, ...] = (
    "PIL",
    "boto3",
    "botocore",
    "fsspec",
    "matplotlib",
    "numba",
    "s3transfer",
    "urllib3",
)


@dataclass(slots=True)
class _LoggingState:
    """Tracks the handler installed by :func:`configure_logging`.

    Held as object attributes rather than bare module-level globals so they can
    be flipped without a ``global`` statement: rebinding a module global is
    discouraged (ruff ``PLW0603``), whereas mutating an attribute of an
    existing object is not.

    Attributes:
        configured (`bool`): Whether :func:`configure_logging` has run.
        handler (`logging.Handler`, optional): The root handler owned by this
            module, so reconfiguration replaces it instead of stacking
            duplicates.
    """

    configured: bool = False
    handler: logging.Handler | None = None


# Guards :func:`configure_logging` against redundant reconfiguration.
_state: _LoggingState = _LoggingState()


class LoggerProtocol(Protocol):
    """Protocol describing methods exposed by a structlog logger.

    Log methods take the event message positionally followed by arbitrary
    key-value pairs, e.g. ``logger.info("model loaded", epochs=10)``.
    """

    def debug(self, event: str, /, **kwargs: Any) -> Any:
        """Log a debug-level event.

        Args:
            event (`str`): Event message.
            **kwargs: Additional key-value pairs to include in the log entry.
        """
        ...

    def info(self, event: str, /, **kwargs: Any) -> Any:
        """Log an info-level event.

        Args:
            event (`str`): Event message.
            **kwargs: Additional key-value pairs to include in the log entry.
        """
        ...

    def warning(self, event: str, /, **kwargs: Any) -> Any:
        """Log a warning-level event.

        Args:
            event (`str`): Event message.
            **kwargs: Additional key-value pairs to include in the log entry.
        """
        ...

    def error(self, event: str, /, **kwargs: Any) -> Any:
        """Log an error-level event.

        Args:
            event (`str`): Event message.
            **kwargs: Additional key-value pairs to include in the log entry.
        """
        ...

    def exception(self, event: str, /, **kwargs: Any) -> Any:
        """Log an error-level event with the active exception's traceback.

        Args:
            event (`str`): Event message.
            **kwargs: Additional key-value pairs to include in the log entry.
        """
        ...

    def critical(self, event: str, /, **kwargs: Any) -> Any:
        """Log a critical-level event.

        Args:
            event (`str`): Event message.
            **kwargs: Additional key-value pairs to include in the log entry.
        """
        ...

    def bind(self, /, **kwargs: Any) -> "LoggerProtocol":
        """Bind additional context to the logger.

        Args:
            **kwargs: Key-value pairs to bind to the logger.

        Returns:
            `LoggerProtocol`: A new logger instance with the bound context.
        """
        ...

    def new(self, /, **kwargs: Any) -> "LoggerProtocol":
        """Create a new logger with fresh context.

        Args:
            **kwargs: Key-value pairs to bind to the new logger.

        Returns:
            `LoggerProtocol`: A new logger instance with the bound context.
        """
        ...

    def unbind(self, /, *keys: str) -> "LoggerProtocol":
        """Unbind context from the logger.

        Args:
            *keys (`str`): Keys to unbind from the logger.

        Returns:
            `LoggerProtocol`: A new logger instance with the specified keys
                unbound.
        """
        ...


@dataclass(frozen=True, slots=True)
class LogSettings:
    """Resolved logging configuration derived from arguments and env vars.

    Attributes:
        environment (`str`): Current environment (e.g. "development",
            "production"). Production environments default to JSON output.
        log_level (`str`): Logging level name (e.g. "DEBUG", "INFO").
        level_no (`int`): Numeric logging level corresponding to `log_level`.
        use_json (`bool`): Whether to use JSON format for logs.
    """

    environment: str
    log_level: str
    level_no: int
    use_json: bool


def _resolve_log_settings(
    *,
    environment: str | None = None,
    log_level: str | int | None = None,
    force_json: bool | None = None,
) -> LogSettings:
    """Resolve logging settings from arguments and environment variables.

    Args:
        environment (`str`, optional): Current environment. If None, defaults
            to the ENVIRONMENT environment variable or "development" if not
            set.
        log_level (`str`, `int`, optional): Logging level name or number. If
            None, defaults to the LOG_LEVEL environment variable or INFO if
            not set.
        force_json (`bool`, optional): If True, use JSON format for logs. If
            None, defaults to the LOG_JSON environment variable; if that is
            also unset, JSON is used in production environments and console
            output everywhere else.

    Returns:
        `LogSettings`: Resolved logging settings.

    Raises:
        ValueError: If the resolved log level is not a level known to the
            standard library.
    """
    env: str = (
        (
            environment
            if environment is not None
            else os.getenv("ENVIRONMENT", "development")
        )
        .strip()
        .lower()
    )

    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO")
    level_str = (
        logging.getLevelName(log_level)
        if isinstance(log_level, int)
        else log_level.strip().upper()
    )
    if level_str not in _LEVEL_NAME_TO_NO:
        error_msg = (
            f"Invalid log level: {log_level!r}. "
            f"Must be one of: {', '.join(sorted(_LEVEL_NAME_TO_NO))}"
        )
        raise ValueError(error_msg)
    level_no = _LEVEL_NAME_TO_NO[level_str]

    if force_json is not None:
        use_json = force_json
    else:
        log_json_env = os.getenv("LOG_JSON")
        use_json = (
            log_json_env.strip().lower() in _JSON_TRUTHY
            if log_json_env is not None
            else env in _PRODUCTION_ENVIRONMENTS
        )

    return LogSettings(
        environment=env, log_level=level_str, level_no=level_no, use_json=use_json
    )


def _shared_processors() -> "list[Processor]":
    """Assemble the processors applied to structlog and foreign records alike.

    Returns:
        `list[Processor]`: Processors that enrich the event dict before
            rendering; used both in the structlog pipeline and as the
            ``foreign_pre_chain`` for standard library records.
    """
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]


def _renderer_processors(*, use_json: bool, colors: bool) -> "list[Processor]":
    """Assemble the final rendering processors for the chosen log format.

    Args:
        use_json (`bool`): Whether to render logs as JSON lines. When False, a
            console renderer is used instead.
        colors (`bool`): Whether the console renderer should emit ANSI colour
            codes. Ignored when `use_json` is True.

    Returns:
        `list[Processor]`: Rendering processors for the tail of a
            :class:`structlog.stdlib.ProcessorFormatter` chain.
    """
    if use_json:
        return [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    return [structlog.dev.ConsoleRenderer(colors=colors)]


def configure_logging(
    *,
    environment: str | None = None,
    log_level: str | int | None = None,
    force_json: bool | None = None,
    suppressed_libs: "Sequence[str]" = DEFAULT_SUPPRESSED_LIBS,
    stream: TextIO | None = None,
    force: bool = False,
) -> None:
    """Configure logging for the application.

    Sets up structlog and the standard library root logger to render through a
    shared :class:`structlog.stdlib.ProcessorFormatter`, so structlog events
    and third-party library records receive identical formatting (JSON or
    console). The call is idempotent: repeat calls are no-ops unless `force`
    is True. The handler installed here is tracked, so reconfiguring replaces
    it rather than stacking duplicates; handlers installed by other code (e.g.
    pytest's log capture) are left untouched.

    Args:
        environment (`str`, optional): Current environment (e.g.
            "development", "production"). If None, defaults to the ENVIRONMENT
            environment variable or "development" if not set. Production
            environments default to JSON output.
        log_level (`str`, `int`, optional): Logging level name or number (e.g.
            "DEBUG", `logging.INFO`). If None, defaults to the LOG_LEVEL
            environment variable or INFO if not set.
        force_json (`bool`, optional): If True, use JSON format for logs. If
            None, defaults to the LOG_JSON environment variable, then to
            whether `environment` is a production environment.
        suppressed_libs (`Sequence[str]`, optional): Loggers whose level is
            raised to at least WARNING (never below the configured level) to
            silence chatty dependencies. Defaults to DEFAULT_SUPPRESSED_LIBS.
        stream (`TextIO`, optional): Stream to which logs are written. If
            None, `sys.stderr` is resolved at call time.
        force (`bool`, optional): If True, reconfigure logging even if it has
            already been configured. Defaults to False.
    """
    if _state.configured and not force:
        return

    settings: LogSettings = _resolve_log_settings(
        environment=environment, log_level=log_level, force_json=force_json
    )
    target: TextIO = stream if stream is not None else sys.stderr

    shared = _shared_processors()
    colors = not settings.use_json and hasattr(target, "isatty") and target.isatty()
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            *_renderer_processors(use_json=settings.use_json, colors=colors),
        ],
        foreign_pre_chain=[*shared, structlog.stdlib.ExtraAdder()],
    )

    root = logging.getLogger()
    if _state.handler is not None:
        root.removeHandler(_state.handler)
        _state.handler.close()
    handler = logging.StreamHandler(target)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(settings.level_no)

    # Quieten noisy libraries without dropping below the configured level
    for lib in suppressed_libs:
        logging.getLogger(lib).setLevel(max(logging.WARNING, settings.level_no))

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    _state.handler = handler
    _state.configured = True


def reset_logging() -> None:
    """Undo :func:`configure_logging`, restoring the unconfigured state.

    Removes the root handler installed by :func:`configure_logging` (handlers
    installed by other code are left untouched) and resets structlog to its
    defaults. Intended for test isolation, e.g. in a pytest fixture.
    """
    if _state.handler is not None:
        logging.getLogger().removeHandler(_state.handler)
        _state.handler.close()
        _state.handler = None
    structlog.reset_defaults()
    _state.configured = False


def get_logger(name: str | None = None) -> LoggerProtocol:
    """Return a bound structlog logger instance.

    Callers pass their own ``__name__`` so that each log record is attributed
    to the emitting module via the ``add_logger_name`` processor.

    Args:
        name (`str`, optional): Name of the logger. If None, the logger name
            is inferred from the calling module on first use.

    Returns:
        `LoggerProtocol`: A structlog logger instance. The underlying
            structlog proxy defers binding until first use, so calling this
            before :func:`configure_logging` is safe, but the logger will not
            be fully configured until after that function is called.
    """
    if name is None:
        return structlog.get_logger()
    return structlog.get_logger(name)
