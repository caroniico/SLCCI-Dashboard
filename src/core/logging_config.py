"""
🔧 Centralized Logging Configuration
====================================
Best practices logging for the entire application.

Features:
- Structured JSON logging for production
- Colorized console output for development
- File rotation
- Error tracking with context
- Performance metrics

Usage:
    from src.core.logging_config import setup_logging, get_logger
    
    setup_logging(level="DEBUG", env="development")
    logger = get_logger(__name__)
    
    logger.info("Processing gate", extra={"gate_id": "fram_strait"})
    logger.error("Failed to load", exc_info=True)
"""

import logging
import logging.handlers
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from functools import wraps
import traceback


# === COLOR CODES FOR TERMINAL ===
class Colors:
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"


LEVEL_COLORS = {
    "DEBUG": Colors.CYAN,
    "INFO": Colors.GREEN,
    "WARNING": Colors.YELLOW,
    "ERROR": Colors.RED,
    "CRITICAL": Colors.BOLD + Colors.RED,
}


# === CUSTOM FORMATTERS ===

class ColoredFormatter(logging.Formatter):
    """Colorized formatter for development console output."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Add color to level name
        level_color = LEVEL_COLORS.get(record.levelname, Colors.WHITE)
        colored_level = f"{level_color}{record.levelname:8}{Colors.RESET}"
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        
        # Format message
        msg = record.getMessage()
        
        # Add extra context if available
        extra_str = ""
        if hasattr(record, 'extra_data'):
            extra_str = f" {Colors.MAGENTA}| {record.extra_data}{Colors.RESET}"
        
        # Format location
        location = f"{record.module}.{record.funcName}:{record.lineno}"
        
        return f"{Colors.BLUE}{timestamp}{Colors.RESET} {colored_level} {Colors.WHITE}{location:40}{Colors.RESET} {msg}{extra_str}"


class JSONFormatter(logging.Formatter):
    """JSON formatter for production/log aggregation."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info) if record.exc_info[0] else None,
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'pathname', 'process', 'processName', 'relativeCreated',
                          'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
                          'message', 'taskName']:
                try:
                    json.dumps(value)  # Check if serializable
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)
        
        return json.dumps(log_data)


class StreamlitFormatter(logging.Formatter):
    """Formatter optimized for Streamlit apps (no colors, clean output)."""
    
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        return f"[{timestamp}] {record.levelname:8} {record.module}: {record.getMessage()}"


# === SETUP FUNCTIONS ===

def setup_logging(
    level: str = "INFO",
    env: str = "development",
    log_dir: Optional[Path] = None,
    app_name: str = "nico"
) -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        env: Environment ("development", "production", "streamlit")
        log_dir: Directory for log files (default: logs/)
        app_name: Application name for log files
    """
    # Default log directory
    if log_dir is None:
        log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # === CONSOLE HANDLER ===
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    if env == "development":
        console_handler.setFormatter(ColoredFormatter())
    elif env == "streamlit":
        console_handler.setFormatter(StreamlitFormatter())
    else:  # production
        console_handler.setFormatter(JSONFormatter())
    
    root_logger.addHandler(console_handler)
    
    # === FILE HANDLER (Rotating) ===
    log_file = log_dir / f"{app_name}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)
    
    # === ERROR FILE (separate) ===
    error_file = log_dir / f"{app_name}_errors.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(error_handler)
    
    # Log startup
    root_logger.info(f"Logging configured: level={level}, env={env}, log_dir={log_dir}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# === DECORATORS FOR DEBUGGING ===

def log_call(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function calls with arguments and results.
    
    Usage:
        @log_call()
        def my_function(x, y):
            return x + y
    """
    def decorator(func):
        _logger = logger or logging.getLogger(func.__module__)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            call_id = id(args) % 10000  # Simple call ID
            
            # Log entry
            args_repr = [repr(a)[:50] for a in args[:3]]  # Limit args shown
            kwargs_repr = [f"{k}={repr(v)[:30]}" for k, v in list(kwargs.items())[:3]]
            signature = ", ".join(args_repr + kwargs_repr)
            _logger.debug(f"[{call_id}] CALL {func.__name__}({signature})")
            
            try:
                result = func(*args, **kwargs)
                result_repr = repr(result)[:100] if result is not None else "None"
                _logger.debug(f"[{call_id}] RETURN {func.__name__} -> {result_repr}")
                return result
            except Exception as e:
                _logger.error(f"[{call_id}] ERROR {func.__name__}: {type(e).__name__}: {e}", exc_info=True)
                raise
        
        return wrapper
    return decorator


def log_errors(logger: Optional[logging.Logger] = None, reraise: bool = True):
    """
    Decorator to log exceptions with full context.
    
    Usage:
        @log_errors()
        def risky_function():
            ...
    """
    def decorator(func):
        _logger = logger or logging.getLogger(func.__module__)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                _logger.error(
                    f"Exception in {func.__name__}: {type(e).__name__}: {e}",
                    exc_info=True,
                    extra={
                        "function": func.__name__,
                        "module": func.__module__,
                        "error_type": type(e).__name__,
                    }
                )
                if reraise:
                    raise
                return None
        
        return wrapper
    return decorator


# === STREAMLIT INTEGRATION ===

def setup_streamlit_logging(level: str = "INFO") -> logging.Logger:
    """
    Setup logging optimized for Streamlit apps.
    
    Returns the app logger for convenience.
    """
    setup_logging(level=level, env="streamlit")
    return get_logger("streamlit_app")


# === CONTEXT MANAGER FOR OPERATIONS ===

class LogContext:
    """
    Context manager for logging operations with timing.
    
    Usage:
        with LogContext(logger, "Loading data", gate_id="fram"):
            load_data()
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        level: int = logging.INFO,
        **context
    ):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.log(self.level, f"START: {self.operation}", extra=self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.log(
                self.level,
                f"DONE: {self.operation} ({duration:.3f}s)",
                extra={**self.context, "duration_seconds": duration}
            )
        else:
            self.logger.error(
                f"FAILED: {self.operation} ({duration:.3f}s) - {exc_type.__name__}: {exc_val}",
                extra={**self.context, "duration_seconds": duration, "error": str(exc_val)},
                exc_info=True
            )
        
        return False  # Don't suppress exceptions


# === AUTO-SETUP ON IMPORT ===

# Check environment and auto-configure
_env = os.getenv("NICO_ENV", "development")
_level = os.getenv("NICO_LOG_LEVEL", "INFO")

# Only auto-setup if not already configured
if not logging.getLogger().handlers:
    setup_logging(level=_level, env=_env)
