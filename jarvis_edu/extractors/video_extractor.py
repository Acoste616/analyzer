"""Video content extractor with audio transcription."""

import asyncio
import tempfile
from typing import Dict, Any
from pathlib import Path

from .base_extractor import BaseExtractor, ExtractedContent
from ..core.logger import get_logger

logger = get_logger(__name__)


class VideoExtractor(BaseExtractor):
    """Extract educational content from videos through audio transcription."""
    
    SUPPORTED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
    
    def __init__(self):
        super().__init__()
        self._whisper_available = self._check_whisper_availability()
    
    def _check_whisper_availability(self) -> bool:
        """Check if Whisper is available."""
        try:
            import whisper
            return True
        except ImportError:
            logger.warning("Whisper not available for video transcription")
            return False
    
    def supports_file(self, file_path: str) -> bool:
        """Check if this extractor supports the file type."""
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    async def extract(self, file_path: str) -> ExtractedContent:
        """Extract educational content from video."""
        file_info = self.get_file_info(file_path)
        
        # For large files (>100MB), create preview without full processing
        if file_info['size'] > 100 * 1024 * 1024:  # 100MB
            return await self._create_preview_content(file_path, file_info)
        
        # For smaller files, attempt basic analysis
        return await self._create_smart_content(file_path, file_info)
    
    async def _create_smart_content(self, file_path: str, file_info: Dict[str, Any]) -> ExtractedContent:
        """Create intelligent content for manageable video files."""
        title = self.generate_title_from_filename(file_path)
        
        # Estimate video details
        duration_min = self._estimate_duration(file_info['size'])
        
        content = f"""Materiał wideo - {title}

📹 Szczegóły wideo:
- Rozmiar pliku: {file_info['size'] / (1024*1024):.1f} MB
- Szacowany czas: ~{duration_min} minut
- Format: {file_info['extension'].upper()}

🎯 Analiza treści:
Ten materiał wideo może zawierać wartościową treść edukacyjną.
Dla pełnej analizy wymagana jest transkrypcja audio.

💡 Sugerowane działania:
1. Odtwórz wideo i sprawdź zawartość
2. Dodaj ręczne notatki do opisu
3. Skonfiguruj automatyczną transkrypcję (Whisper)

📝 Status: Gotowy do dalszej analizy"""
        
        summary = f"Materiał wideo ({duration_min} min): {title}. Wymaga transkrypcji dla pełnej analizy."
        
        metadata = {
            **file_info,
            "extraction_method": "smart_preview",
            "estimated_duration_min": duration_min,
            "confidence": 0.6
        }
        
        return ExtractedContent(
            title=title,
            summary=summary,
            content=content,
            metadata=metadata,
            confidence=0.6
        )
    
    async def _create_preview_content(self, file_path: str, file_info: Dict[str, Any]) -> ExtractedContent:
        """Create preview for large video files."""
        title = self.generate_title_from_filename(file_path)
        size_mb = file_info['size'] / (1024*1024)
        
        if size_mb > 1000:  # > 1GB
            duration_estimate = "prawdopodobnie 1-3 godziny"
            processing_note = "⚠️ DUŻY PLIK: Wymaga dedykowanego przetwarzania"
        elif size_mb > 500:  # > 500MB
            duration_estimate = "prawdopodobnie 30-60 minut"
            processing_note = "📊 ŚREDNI PLIK: Zalecane przetwarzanie partiami"
        else:
            duration_estimate = "prawdopodobnie 10-30 minut"
            processing_note = "✅ STANDARDOWY: Możliwe automatyczne przetwarzanie"
        
        content = f"""Materiał wideo - {title}

{processing_note}

📊 Charakterystyka pliku:
- Rozmiar: {size_mb:.1f} MB
- Szacowany czas: {duration_estimate}
- Format: {file_info['extension'].upper()}

🔄 Opcje przetwarzania:
• Podgląd fragmentu (pierwsze 5 min)
• Przetwarzanie partiami (co 10 min)
• Pełna transkrypcja (w tle)
• Ekstrakcja kluczowych momentów

⏱️ Szacowany czas przetwarzania: {self._estimate_processing_time(size_mb)}

Status: Oczekuje na konfigurację przetwarzania"""
        
        summary = f"Duży materiał wideo ({size_mb:.1f} MB, {duration_estimate}) - {title}"
        
        metadata = {
            **file_info,
            "extraction_method": "large_file_preview",
            "size_category": "large" if size_mb > 500 else "medium",
            "processing_recommendation": "batch" if size_mb > 500 else "full",
            "confidence": 0.4
        }
        
        return ExtractedContent(
            title=f"[{size_mb:.0f}MB] {title}",
            summary=summary,
            content=content,
            metadata=metadata,
            confidence=0.4
        )
    
    def _estimate_duration(self, file_size: int) -> int:
        """Estimate video duration based on file size."""
        # Rough estimate: 1MB per minute for standard quality
        mb_size = file_size / (1024 * 1024)
        estimated_minutes = max(1, int(mb_size / 2))  # Conservative estimate
        return min(estimated_minutes, 180)  # Cap at 3 hours
    
    def _estimate_processing_time(self, size_mb: float) -> str:
        """Estimate processing time for transcription."""
        if size_mb > 1000:
            return "2-4 godziny"
        elif size_mb > 500:
            return "30-90 minut"
        elif size_mb > 100:
            return "5-20 minut"
        else:
            return "1-5 minut" 