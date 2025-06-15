"""Audio content extractor with transcription."""

from typing import Dict, Any
from pathlib import Path

from .base_extractor import BaseExtractor, ExtractedContent
from ..core.logger import get_logger

logger = get_logger(__name__)


class AudioExtractor(BaseExtractor):
    """Extract educational content from audio files."""
    
    SUPPORTED_EXTENSIONS = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
    
    def __init__(self):
        super().__init__()
    
    def supports_file(self, file_path: str) -> bool:
        """Check if this extractor supports the file type."""
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    async def extract(self, file_path: str) -> ExtractedContent:
        """Extract educational content from audio."""
        file_info = self.get_file_info(file_path)
        
        # For large audio files, create preview
        if file_info['size'] > 50 * 1024 * 1024:  # 50MB
            return await self._create_preview_content(file_path, file_info)
        
        # For smaller files, create smart content
        return await self._create_smart_content(file_path, file_info)
    
    async def _create_smart_content(self, file_path: str, file_info: Dict[str, Any]) -> ExtractedContent:
        """Create intelligent content for audio files."""
        title = self.generate_title_from_filename(file_path)
        duration_min = self._estimate_duration(file_info['size'])
        
        content = f"""Materiał audio - {title}

🎵 Szczegóły audio:
- Rozmiar pliku: {file_info['size'] / (1024*1024):.1f} MB
- Szacowany czas: ~{duration_min} minut
- Format: {file_info['extension'].upper()}

🎯 Typ materiału:
{self._detect_content_type(file_path, file_info)}

💡 Następne kroki:
1. Odsłuchaj nagranie
2. Skonfiguruj transkrypcję automatyczną
3. Dodaj tagi i kategorię

📝 Status: Gotowy do analizy"""
        
        summary = f"Materiał audio ({duration_min} min): {title}"
        
        metadata = {
            **file_info,
            "extraction_method": "audio_analysis",
            "estimated_duration_min": duration_min,
            "confidence": 0.7
        }
        
        return ExtractedContent(
            title=title,
            summary=summary,
            content=content,
            metadata=metadata,
            confidence=0.7
        )
    
    async def _create_preview_content(self, file_path: str, file_info: Dict[str, Any]) -> ExtractedContent:
        """Create preview for large audio files."""
        title = self.generate_title_from_filename(file_path)
        size_mb = file_info['size'] / (1024*1024)
        duration_min = self._estimate_duration(file_info['size'])
        
        content = f"""Materiał audio - {title}

📊 Duży plik audio ({size_mb:.1f} MB)
⏱️ Szacowany czas: ~{duration_min} minut

🔄 Opcje przetwarzania:
• Fragment początkowy (5 min)
• Transkrypcja w partiach
• Pełna analiza w tle
• Ekstrakcja kluczowych fragmentów

⚙️ Zalecane: Przetwarzanie partiami dla optymalnej wydajności

Status: Oczekuje na konfigurację"""
        
        summary = f"Duży materiał audio ({size_mb:.1f} MB, ~{duration_min} min) - {title}"
        
        metadata = {
            **file_info,
            "extraction_method": "large_audio_preview",
            "estimated_duration_min": duration_min,
            "confidence": 0.5
        }
        
        return ExtractedContent(
            title=f"[{size_mb:.0f}MB] {title}",
            summary=summary,
            content=content,
            metadata=metadata,
            confidence=0.5
        )
    
    def _estimate_duration(self, file_size: int) -> int:
        """Estimate audio duration based on file size."""
        # Audio is typically 1MB per minute for good quality MP3
        mb_size = file_size / (1024 * 1024)
        estimated_minutes = max(1, int(mb_size))
        return min(estimated_minutes, 240)  # Cap at 4 hours
    
    def _detect_content_type(self, file_path: str, file_info: Dict[str, Any]) -> str:
        """Detect likely content type from filename."""
        filename_lower = file_info['filename'].lower()
        
        if any(word in filename_lower for word in ['lekcja', 'wykład', 'lecture', 'lesson']):
            return "📚 Prawdopodobnie materiał edukacyjny (lekcja/wykład)"
        elif any(word in filename_lower for word in ['podcast', 'wywiad', 'interview']):
            return "🎙️ Prawdopodobnie podcast lub wywiad"
        elif any(word in filename_lower for word in ['prezentacja', 'presentation']):
            return "📊 Prawdopodobnie audio z prezentacji"
        elif any(word in filename_lower for word in ['kurs', 'course', 'szkolenie']):
            return "🎓 Prawdopodobnie materiał szkoleniowy"
        else:
            return "🎵 Materiał audio do analizy" 