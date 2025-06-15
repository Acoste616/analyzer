"""Advanced logging system using Loguru."""

import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from loguru import logger
from pydantic import BaseModel


class LogConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"
    file: Optional[str] = None
    rotation: str = "1 day"
    retention: str = "30 days"
    compression: str = "gz"
    backtrace: bool = True
    diagnose: bool = True


def setup_logging(config: Optional[LogConfig] = None) -> None:
    """Setup logging with Loguru."""
    if config is None:
        config = LogConfig()
    
    # Remove default handler
    logger.remove()
    
    # Custom format for structured logging
    if config.format == "json":
        def json_formatter(record):
            return json.dumps({
                "timestamp": record["time"].isoformat(),
                "level": record["level"].name,
                "message": record["message"],
                "module": record["module"],
                "function": record["function"],
                "line": record["line"],
                "process": record["process"].id if record["process"] else None,
                "thread": record["thread"].id if record["thread"] else None,
                "extra": record["extra"]
            }) + "\n"
        
        # Console handler with JSON format
        logger.add(
            sys.stderr,
            format=json_formatter,
            level=config.level,
            backtrace=config.backtrace,
            diagnose=config.diagnose
        )
        
        # File handler with JSON format
        if config.file:
            logger.add(
                config.file,
                format=json_formatter,
                level=config.level,
                rotation=config.rotation,
                retention=config.retention,
                compression=config.compression,
                backtrace=config.backtrace,
                diagnose=config.diagnose
            )
    else:
        # Human-readable format
        format_str = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        
        # Console handler
        logger.add(
            sys.stderr,
            format=format_str,
            level=config.level,
            backtrace=config.backtrace,
            diagnose=config.diagnose,
            colorize=True
        )
        
        # File handler
        if config.file:
            logger.add(
                config.file,
                format=format_str,
                level=config.level,
                rotation=config.rotation,
                retention=config.retention,
                compression=config.compression,
                backtrace=config.backtrace,
                diagnose=config.diagnose
            )


def get_logger(name: Optional[str] = None) -> logger:
    """Get logger instance."""
    if name:
        return logger.bind(name=name)
    return logger


class ProcessingLogger:
    """Specialized logger for processing operations."""
    
    def __init__(self, process_id: str, logger_instance: logger = None):
        self.process_id = process_id
        self.logger = logger_instance or logger
        self.start_time = datetime.now()
        
    def log_start(self, operation: str, **kwargs):
        """Log operation start."""
        self.logger.info(
            "Operation started",
            process_id=self.process_id,
            operation=operation,
            **kwargs
        )
    
    def log_progress(self, operation: str, progress: float, **kwargs):
        """Log operation progress."""
        self.logger.info(
            "Operation progress",
            process_id=self.process_id,
            operation=operation,
            progress=progress,
            **kwargs
        )
    
    def log_success(self, operation: str, **kwargs):
        """Log operation success."""
        duration = (datetime.now() - self.start_time).total_seconds()
        self.logger.success(
            "Operation completed successfully",
            process_id=self.process_id,
            operation=operation,
            duration=duration,
            **kwargs
        )
    
    def log_error(self, operation: str, error: Exception, **kwargs):
        """Log operation error."""
        duration = (datetime.now() - self.start_time).total_seconds()
        self.logger.error(
            "Operation failed",
            process_id=self.process_id,
            operation=operation,
            error=str(error),
            error_type=type(error).__name__,
            duration=duration,
            **kwargs
        )
    
    def log_warning(self, operation: str, message: str, **kwargs):
        """Log operation warning."""
        self.logger.warning(
            message,
            process_id=self.process_id,
            operation=operation,
            **kwargs
        )


# Setup default logging
setup_logging(LogConfig())

# Export main logger
__all__ = ["logger", "setup_logging", "get_logger", "ProcessingLogger", "LogConfig"] 