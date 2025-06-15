"""Image content extractor using OCR."""

import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

from .base_extractor import BaseExtractor, ExtractedContent
from ..core.logger import get_logger

logger = get_logger(__name__)


class ImageExtractor(BaseExtractor):
    """Extract educational content from images using OCR."""
    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    def __init__(self):
        super().__init__()
        self._ocr_available = self._check_ocr_availability()
        self._easyocr_reader = None
        self._reader_lock = threading.Lock()
        self._thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="OCR")
    
    def _check_ocr_availability(self) -> bool:
        """Check if OCR libraries are available."""
        try:
            import easyocr
            return True
        except ImportError:
            try:
                import pytesseract
                from PIL import Image
                return True
            except ImportError:
                logger.warning("No OCR library available (easyocr or pytesseract)")
                return False
    
    def _get_easyocr_reader(self):
        """Get cached EasyOCR reader instance."""
        if self._easyocr_reader is None:
            with self._reader_lock:
                if self._easyocr_reader is None:  # Double-check locking
                    try:
                        import easyocr
                        self._easyocr_reader = easyocr.Reader(['pl', 'en'], gpu=False)
                        logger.debug("EasyOCR reader initialized")
                    except Exception as e:
                        logger.error(f"Failed to initialize EasyOCR reader: {e}")
                        self._easyocr_reader = None
        return self._easyocr_reader
    
    def supports_file(self, file_path: str) -> bool:
        """Check if this extractor supports the file type."""
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    async def extract(self, file_path: str) -> ExtractedContent:
        """Extract text and educational content from image."""
        file_info = self.get_file_info(file_path)
        
        # Try OCR extraction
        extracted_text = await self._extract_text_ocr(file_path)
        
        if not extracted_text or len(extracted_text.strip()) < 10:
            # Fallback: generate content based on filename and basic analysis
            return await self._create_fallback_content(file_path, file_info)
        
        # Analyze extracted text for educational content
        title = self._generate_title_from_content(extracted_text, file_info)
        summary = self._generate_summary(extracted_text)
        
        metadata = {
            **file_info,
            "extraction_method": "ocr",
            "text_length": len(extracted_text),
            "confidence": 0.8 if len(extracted_text) > 100 else 0.5
        }
        
        return ExtractedContent(
            title=title,
            summary=summary,
            content=extracted_text,
            metadata=metadata,
            confidence=metadata["confidence"]
        )
    
    async def _extract_text_ocr(self, file_path: str) -> str:
        """Extract text using OCR (async with thread pool)."""
        if not self._ocr_available:
            return ""
        
        try:
            # Run OCR operations in thread pool to avoid blocking async context
            loop = asyncio.get_event_loop()
            
            # Try EasyOCR first (better for multiple languages)
            try:
                text = await loop.run_in_executor(
                    self._thread_pool,
                    self._run_easyocr,
                    file_path
                )
                if text and text.strip():
                    return text.strip()
            except Exception as e:
                logger.debug(f"EasyOCR failed: {e}")
            
            # Fallback to Tesseract
            try:
                text = await loop.run_in_executor(
                    self._thread_pool,
                    self._run_tesseract,
                    file_path
                )
                if text:
                    return text.strip()
                
            except Exception as e:
                logger.debug(f"Tesseract failed: {e}")
                
        except Exception as e:
            logger.error(f"OCR extraction failed for {file_path}: {e}")
        
        return ""
    
    def _run_easyocr(self, file_path: str) -> str:
        """Run EasyOCR in thread (blocking operation)."""
        reader = self._get_easyocr_reader()
        if reader is None:
            return ""
        
        try:
            results = reader.readtext(file_path)
            text = ' '.join([result[1] for result in results if result[2] > 0.5])
            return text
        except Exception as e:
            logger.debug(f"EasyOCR processing failed: {e}")
            return ""
    
    def _run_tesseract(self, file_path: str) -> str:
        """Run Tesseract OCR in thread (blocking operation)."""
        try:
            import pytesseract
            from PIL import Image
            
            # Configure Tesseract for Polish and English
            config = '--oem 3 --psm 6 -l pol+eng'
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image, config=config)
            return text
        except Exception as e:
            logger.debug(f"Tesseract processing failed: {e}")
            return ""
    
    async def _create_fallback_content(self, file_path: str, file_info: Dict[str, Any]) -> ExtractedContent:
        """Create content when OCR fails."""
        title = self.generate_title_from_filename(file_path)
        
        content = f"""Lekcja wizualna - {title}

To jest materiał edukacyjny w formie obrazu. 

Informacje o pliku:
- Nazwa: {file_info['filename']}
- Rozmiar: {file_info['size'] / 1024:.1f} KB
- Format: {file_info['extension'].upper()}

Analiza treści wymaga ręcznego przeglądu obrazu."""
        
        summary = f"Materiał wizualny: {title}. Wymaga ręcznej analizy treści."
        
        metadata = {
            **file_info,
            "extraction_method": "fallback",
            "text_length": 0,
            "confidence": 0.3
        }
        
        return ExtractedContent(
            title=title,
            summary=summary,
            content=content,
            metadata=metadata,
            confidence=0.3
        )
    
    def _generate_title_from_content(self, text: str, file_info: Dict[str, Any]) -> str:
        """Generate meaningful title from extracted text."""
        # Look for educational indicators
        educational_keywords = [
            'lekcja', 'wykład', 'prezentacja', 'slajd', 'materiał',
            'kurs', 'szkolenie', 'tutorial', 'lesson', 'lecture'
        ]
        
        lines = text.split('\n')
        first_meaningful_line = ""
        
        # Find first substantial line
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if len(line) > 5 and not line.isdigit():
                first_meaningful_line = line
                break
        
        # Check if it looks like a title
        if first_meaningful_line:
            # If it's short and doesn't end with punctuation, likely a title
            if len(first_meaningful_line) < 80 and not first_meaningful_line.endswith('.'):
                return first_meaningful_line
        
        # Look for educational content indicators
        text_lower = text.lower()
        for keyword in educational_keywords:
            if keyword in text_lower:
                return f"Materiał edukacyjny - {self.generate_title_from_filename(file_info['filename'])}"
        
        # Fallback to filename-based title
        return self.generate_title_from_filename(file_info['filename'])
    
    def _generate_summary(self, text: str) -> str:
        """Generate summary from extracted text."""
        if len(text) < 100:
            return f"Krótki tekst zawierający {len(text.split())} słów."
        
        # Take first few sentences or first 200 characters
        sentences = text.replace('\n', ' ').split('. ')
        if len(sentences) > 1:
            summary = '. '.join(sentences[:2]) + '.'
        else:
            summary = text[:200] + ('...' if len(text) > 200 else '')
        
        word_count = len(text.split())
        return f"{summary} (Tekst zawiera około {word_count} słów)"
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, '_thread_pool') and self._thread_pool:
            try:
                self._thread_pool.shutdown(wait=False)
            except Exception:
                pass  # Ignore cleanup errors 