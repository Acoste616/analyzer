"""
Jarvis EDU Extractor - Advanced AI-powered educational content processor.

This package provides tools for extracting educational content from various sources
(video, audio, images, web pages, documents) and converting them into structured
lessons in Polish with AI-generated datasets.
"""

__version__ = "0.1.0"
__author__ = "Jarvis EDU Team"
__email__ = "dev@jarvis-edu.com"

# Core exports
from .core.config import Settings, get_settings
from .core.logger import get_logger
from .extractors import (
    VideoExtractor,
    AudioExtractor, 
    ImageExtractor,
    WebExtractor,
    DocumentExtractor
)
from .processors import (
    LessonProcessor,
    TranslationProcessor,
    LLMProcessor
)
from .models import Lesson, ExtractedContent, ProcessingResult

__all__ = [
    "__version__",
    "Settings",
    "get_settings", 
    "get_logger",
    "VideoExtractor",
    "AudioExtractor",
    "ImageExtractor", 
    "WebExtractor",
    "DocumentExtractor",
    "LessonProcessor",
    "TranslationProcessor",
    "LLMProcessor",
    "Lesson",
    "ExtractedContent",
    "ProcessingResult",
] 