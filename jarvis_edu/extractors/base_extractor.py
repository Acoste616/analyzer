"""Base extractor class for content analysis."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

from ..core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedContent:
    """Container for extracted content."""
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    confidence: float = 0.0
    language: str = "pl"
    

class BaseExtractor(ABC):
    """Base class for all content extractors."""
    
    def __init__(self):
        self.logger = logger
        
    @abstractmethod
    async def extract(self, file_path: str) -> ExtractedContent:
        """Extract content from file."""
        pass
    
    @abstractmethod
    def supports_file(self, file_path: str) -> bool:
        """Check if extractor supports this file type."""
        pass
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get basic file information."""
        path = Path(file_path)
        
        # Safely get file size
        try:
            file_size = path.stat().st_size if path.exists() else 0
        except (OSError, PermissionError) as e:
            logger.warning(f"Cannot access file stats for {file_path}: {e}")
            file_size = 0
        
        return {
            "filename": path.name,
            "extension": path.suffix.lower(),
            "size": file_size,
            "stem": path.stem
        }
    
    def generate_title_from_filename(self, file_path: str) -> str:
        """Generate human-readable title from filename."""
        path = Path(file_path)
        stem = path.stem
        
        # Clean up common patterns
        stem = stem.replace("_", " ").replace("-", " ")
        
        # Remove common prefixes/suffixes
        prefixes_to_remove = ["undefined", "video", "audio", "photo", "img"]
        for prefix in prefixes_to_remove:
            if stem.lower().startswith(prefix):
                stem = stem[len(prefix):].strip()
        
        # Remove long number sequences (timestamps, IDs)
        import re
        stem = re.sub(r'\b\d{10,}\b', '', stem)
        stem = re.sub(r'\s+', ' ', stem).strip()
        
        # Capitalize first letter
        if stem:
            return stem[0].upper() + stem[1:] if len(stem) > 1 else stem.upper()
        else:
            return f"Lekcja z pliku {path.name}" 