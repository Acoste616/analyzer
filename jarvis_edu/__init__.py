"""
Jarvis EDU Extractor - Advanced AI-powered educational content processor.

This package provides tools for extracting educational content from various sources
(video, audio, images, web pages, documents) and converting them into structured
lessons in Polish with AI-generated datasets.
"""

__version__ = "0.1.0"
__author__ = "Jarvis EDU Team"
__email__ = "dev@jarvis-edu.com"

# Core exports - only import what exists
_available_exports = ["__version__"]

try:
    from .core.config import get_settings
    _available_exports.append("get_settings")
except ImportError:
    get_settings = None

try:
    from .core.logger import get_logger
    _available_exports.append("get_logger")
except ImportError:
    get_logger = None

__all__ = _available_exports 