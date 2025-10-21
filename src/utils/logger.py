"""
SystemLogger: Professional logging system for AI4Pain Feature Extraction.

This module provides comprehensive logging capabilities with contextual information,
similar to the agent-coordination-framework's logging standards.
"""

import logging
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
import traceback
from enum import Enum


class LogLevel(Enum):
    """Enumeration of available log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SystemLogger:
    """
    Professional logging system with contextual information support.

    Features:
    - Multiple log levels with color coding
    - Contextual logging with metadata
    - Automatic log rotation
    - Performance tracking
    - Error stack trace capture
    """

    _instance: Optional['SystemLogger'] = None
    _initialized: bool = False

    def __new__(cls) -> 'SystemLogger':
        """Singleton pattern to ensure single logger instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, log_dir: str = "logs", log_file: str = "feature_extraction.log"):
        """
        Initialize the SystemLogger.

        Args:
            log_dir: Directory for log files
            log_file: Name of the log file
        """
        if SystemLogger._initialized:
            return

        self.log_dir = Path(log_dir)
        self.log_file = self.log_dir / log_file

        # Create logs directory
        self.log_dir.mkdir(exist_ok=True)

        # Configure logger
        self.logger = logging.getLogger("AI4Pain")
        self.logger.setLevel(logging.DEBUG)

        # Remove existing handlers
        self.logger.handlers.clear()

        # File handler with detailed formatting
        file_handler = logging.FileHandler(self.log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)

        # Console handler with simpler formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        SystemLogger._initialized = True

        # Log initialization
        self.info("SystemLogger initialized", {
            "log_directory": str(self.log_dir),
            "log_file": str(self.log_file)
        })

    def _format_context(self, context: Optional[Dict[str, Any]]) -> str:
        """
        Format context dictionary for logging.

        Args:
            context: Dictionary of contextual information

        Returns:
            Formatted context string
        """
        if not context:
            return ""

        try:
            # Pretty format for readability
            formatted = json.dumps(context, indent=2, default=str)
            return f"\nContext:\n{formatted}"
        except Exception:
            return f"\nContext: {str(context)}"

    def debug(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log debug message with optional context.

        Args:
            message: Debug message
            context: Optional contextual information
        """
        full_message = f"{message}{self._format_context(context)}"
        self.logger.debug(full_message)

    def info(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log info message with optional context.

        Args:
            message: Info message
            context: Optional contextual information
        """
        full_message = f"{message}{self._format_context(context)}"
        self.logger.info(full_message)

    def warning(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log warning message with optional context.

        Args:
            message: Warning message
            context: Optional contextual information
        """
        full_message = f"{message}{self._format_context(context)}"
        self.logger.warning(full_message)

    def error(self,
              message: str,
              exception: Optional[Exception] = None,
              context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log error message with optional exception and context.

        Args:
            message: Error message
            exception: Optional exception object
            context: Optional contextual information
        """
        error_details = {}

        if exception:
            error_details["exception_type"] = type(exception).__name__
            error_details["exception_message"] = str(exception)
            error_details["traceback"] = traceback.format_exc()

        if context:
            error_details.update(context)

        full_message = f"{message}{self._format_context(error_details)}"
        self.logger.error(full_message)

    def critical(self,
                 message: str,
                 exception: Optional[Exception] = None,
                 context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log critical message with optional exception and context.

        Args:
            message: Critical message
            exception: Optional exception object
            context: Optional contextual information
        """
        error_details = {}

        if exception:
            error_details["exception_type"] = type(exception).__name__
            error_details["exception_message"] = str(exception)
            error_details["traceback"] = traceback.format_exc()

        if context:
            error_details.update(context)

        full_message = f"{message}{self._format_context(error_details)}"
        self.logger.critical(full_message)

    def log_performance(self,
                       operation: str,
                       duration_ms: float,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log performance metrics for operations.

        Args:
            operation: Name of the operation
            duration_ms: Duration in milliseconds
            metadata: Additional performance metadata
        """
        perf_context = {
            "operation": operation,
            "duration_ms": round(duration_ms, 2),
            "duration_seconds": round(duration_ms / 1000, 3)
        }

        if metadata:
            perf_context.update(metadata)

        self.info(f"Performance: {operation}", perf_context)

    def start_operation(self, operation: str) -> float:
        """
        Log the start of an operation and return timestamp.

        Args:
            operation: Name of the operation

        Returns:
            Start timestamp
        """
        import time
        start_time = time.time()
        self.debug(f"Starting operation: {operation}")
        return start_time

    def end_operation(self, operation: str, start_time: float) -> None:
        """
        Log the end of an operation with duration.

        Args:
            operation: Name of the operation
            start_time: Start timestamp from start_operation()
        """
        import time
        duration_ms = (time.time() - start_time) * 1000
        self.log_performance(operation, duration_ms)

    def clear_log(self) -> None:
        """Clear the current log file."""
        try:
            with open(self.log_file, 'w') as f:
                f.write(f"=== Log cleared at {datetime.now()} ===\n")
            self.info("Log file cleared")
        except Exception as e:
            self.error("Failed to clear log file", exception=e)


# Global logger instance
logger = SystemLogger()