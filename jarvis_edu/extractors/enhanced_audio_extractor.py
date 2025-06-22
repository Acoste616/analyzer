"""Enhanced audio extractor with Whisper transcription and metadata analysis."""

import asyncio
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import subprocess
import json

from .base_extractor import BaseExtractor, ExtractedContent
from ..core.logger import get_logger
from ..processors.lm_studio import LMStudioProcessor
from ..core.system_paths import get_tool_path, is_tool_available, test_tool

logger = get_logger(__name__)


class EnhancedAudioExtractor(BaseExtractor):
    """Enhanced audio extractor with transcription and analysis."""
    
    SUPPORTED_EXTENSIONS = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
    
    def __init__(self):
        super().__init__()
        self.lm_studio = LMStudioProcessor()
        self._init_whisper()
        self._init_ffmpeg()
        self._init_mutagen()
    
    def _init_whisper(self):
        """Initialize Whisper model."""
        self._whisper_available = False
        self._whisper_model = None
        
        try:
            import whisper
            # Use base model for balance between speed and quality
            self._whisper_model = whisper.load_model("base")
            self._whisper_available = True
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.warning(f"Whisper not available: {e}")
    
    def _init_ffmpeg(self):
        """Check if ffmpeg is available for audio processing using system path manager."""
        self._ffmpeg_available = is_tool_available("ffmpeg")
        self._ffmpeg_path = get_tool_path("ffmpeg")
        
        if self._ffmpeg_available:
            # Test if tool actually works
            if test_tool("ffmpeg"):
                logger.info(f"FFmpeg available for audio processing at: {self._ffmpeg_path}")
            else:
                self._ffmpeg_available = False
                logger.warning("FFmpeg found but not working properly")
        else:
            logger.warning("FFmpeg not available - audio processing limited")
            logger.info("Install FFmpeg: https://ffmpeg.org/download.html#build-windows")
    
    def _init_mutagen(self):
        """Initialize mutagen for metadata extraction."""
        try:
            import mutagen
            self._mutagen_available = True
            logger.info("Mutagen available for metadata extraction")
        except ImportError:
            self._mutagen_available = False
            logger.warning("Mutagen not available - metadata extraction limited")
    
    def supports_file(self, file_path: str) -> bool:
        """Check if this extractor supports the file type."""
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    async def extract(self, file_path: str) -> ExtractedContent:
        """Extract content from audio with transcription and metadata analysis."""
        file_info = self.get_file_info(file_path)
        
        # Extract audio metadata
        audio_metadata = await self._extract_audio_metadata(file_path)
        
        # For large files, decide on processing strategy
        size_mb = file_info['size'] / (1024 * 1024)
        duration_min = audio_metadata.get('duration', 0) / 60
        
        if size_mb > 200 or duration_min > 60:  # Very large file or long duration
            return await self._create_large_audio_preview(file_path, file_info, audio_metadata)
        
        # Extract and transcribe audio
        transcript = await self._transcribe_audio(file_path, audio_metadata)
        
        # Combine all information
        combined_content = self._combine_audio_analysis(transcript, audio_metadata, file_info)
        
        # Generate lesson with AI if available
        if await self.lm_studio.is_available() and combined_content:
            lesson_data = await self._generate_audio_lesson_with_ai(
                combined_content, audio_metadata
            )
            title = lesson_data.get("title", self._generate_smart_audio_title(transcript, audio_metadata, file_path))
            summary = lesson_data.get("summary", "")
            content = lesson_data.get("content", combined_content)
            tags = lesson_data.get("tags", [])
            categories = lesson_data.get("categories", [])
        else:
            # Intelligent processing without AI
            title = self._generate_smart_audio_title(transcript, audio_metadata, file_path)
            summary = self._generate_audio_summary(transcript, audio_metadata, file_info)
            content = combined_content
            tags = self._generate_audio_tags(transcript, audio_metadata, file_info)
            categories = self._detect_audio_categories(transcript, audio_metadata)
        
        confidence = self._calculate_audio_confidence(transcript, audio_metadata)
        
        return ExtractedContent(
            title=title,
            summary=summary,
            content=content,
            metadata={
                **file_info,
                **audio_metadata,
                "extraction_method": "audio_transcription_analysis",
                "has_transcript": bool(transcript),
                "transcript_length": len(transcript) if transcript else 0,
                "suggested_tags": tags,
                "suggested_categories": categories,
                "confidence": confidence
            },
            confidence=confidence
        )
    
    async def _extract_audio_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract detailed audio metadata."""
        metadata = {
            "duration": 0,
            "bitrate": 0,
            "sample_rate": 0,
            "channels": 0,
            "format": "unknown",
            "title": "",
            "artist": "",
            "album": "",
            "date": "",
            "genre": ""
        }
        
        # Try mutagen first for ID3 tags
        if self._mutagen_available:
            try:
                from mutagen import File
                audiofile = File(file_path)
                
                if audiofile is not None:
                    # Duration
                    if hasattr(audiofile, 'info') and audiofile.info:
                        metadata['duration'] = audiofile.info.length
                        metadata['bitrate'] = getattr(audiofile.info, 'bitrate', 0)
                        metadata['sample_rate'] = getattr(audiofile.info, 'sample_rate', 0)
                        metadata['channels'] = getattr(audiofile.info, 'channels', 0)
                    
                    # ID3 tags
                    if audiofile.tags:
                        metadata['title'] = str(audiofile.tags.get('TIT2', [''])[0]) if audiofile.tags.get('TIT2') else ""
                        metadata['artist'] = str(audiofile.tags.get('TPE1', [''])[0]) if audiofile.tags.get('TPE1') else ""
                        metadata['album'] = str(audiofile.tags.get('TALB', [''])[0]) if audiofile.tags.get('TALB') else ""
                        metadata['date'] = str(audiofile.tags.get('TDRC', [''])[0]) if audiofile.tags.get('TDRC') else ""
                        metadata['genre'] = str(audiofile.tags.get('TCON', [''])[0]) if audiofile.tags.get('TCON') else ""
            
            except Exception as e:
                logger.debug(f"Mutagen metadata extraction failed: {e}")
        
        # Fallback to ffprobe for technical details
        if self._ffmpeg_available and metadata['duration'] == 0:
            try:
                cmd = [
                    get_tool_path("ffprobe") or "ffprobe",
                    '-v', 'quiet',
                    '-print_format', 'json',
                    '-show_streams',
                    '-show_format',
                    file_path
                ]
                
                result = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, _ = await result.communicate()
                
                if result.returncode == 0:
                    data = json.loads(stdout.decode())
                    
                    # Extract format info
                    if 'format' in data:
                        metadata['duration'] = float(data['format'].get('duration', 0))
                        metadata['bitrate'] = int(data['format'].get('bit_rate', 0))
                    
                    # Extract stream info
                    for stream in data.get('streams', []):
                        if stream['codec_type'] == 'audio':
                            metadata['sample_rate'] = int(stream.get('sample_rate', 0))
                            metadata['channels'] = int(stream.get('channels', 0))
                            metadata['format'] = stream.get('codec_name', 'unknown')
                            break
            
            except Exception as e:
                logger.debug(f"FFprobe metadata extraction failed: {e}")
        
        return metadata
    
    async def _transcribe_audio(self, file_path: str, metadata: Dict) -> str:
        """Transcribe audio using Whisper."""
        if not self._whisper_available:
            return ""
        
        try:
            duration = metadata.get('duration', 0)
            
            # For very long audio, process in chunks
            if duration > 3600:  # 1 hour
                return await self._transcribe_long_audio(file_path, duration)
            
            # For shorter audio, process directly or convert format
            if self._ffmpeg_available:
                # Convert to WAV for better Whisper compatibility
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                    temp_wav_path = temp_wav.name
                
                cmd = [
                    self._ffmpeg_path,
                    '-i', file_path,
                    '-acodec', 'pcm_s16le',
                    '-ar', '16000',
                    '-ac', '1',
                    '-y',
                    temp_wav_path
                ]
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                await process.communicate()
                
                if process.returncode == 0:
                    # Transcribe with Whisper
                    result = self._whisper_model.transcribe(
                        temp_wav_path,
                        language='pl',
                        task='transcribe'
                    )
                    
                    # Clean up
                    Path(temp_wav_path).unlink(missing_ok=True)
                    
                    return result.get('text', '')
                else:
                    # If conversion failed, try direct transcription
                    result = self._whisper_model.transcribe(
                        file_path,
                        language='pl',
                        task='transcribe'
                    )
                    return result.get('text', '')
            else:
                # Direct transcription without ffmpeg
                result = self._whisper_model.transcribe(
                    file_path,
                    language='pl',
                    task='transcribe'
                )
                return result.get('text', '')
        
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return ""
    
    async def _transcribe_long_audio(self, file_path: str, duration: float) -> str:
        """Transcribe long audio in chunks."""
        transcripts = []
        chunk_duration = 900  # 15 minutes per chunk
        
        for start_time in range(0, int(duration), chunk_duration):
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                    temp_audio_path = temp_audio.name
                
                # Extract chunk
                cmd = [
                    self._ffmpeg_path,
                    '-ss', str(start_time),
                    '-i', file_path,
                    '-t', str(min(chunk_duration, duration - start_time)),
                    '-acodec', 'pcm_s16le',
                    '-ar', '16000',
                    '-ac', '1',
                    '-y',
                    temp_audio_path
                ]
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                await process.communicate()
                
                if process.returncode == 0:
                    # Transcribe chunk
                    result = self._whisper_model.transcribe(
                        temp_audio_path,
                        language='pl',
                        task='transcribe'
                    )
                    
                    chunk_text = result.get('text', '')
                    if chunk_text:
                        transcripts.append(f"[{start_time//60}:{start_time%60:02d}] {chunk_text}")
                
                # Clean up
                Path(temp_audio_path).unlink(missing_ok=True)
                
            except Exception as e:
                logger.error(f"Failed to transcribe audio chunk at {start_time}: {e}")
        
        return '\n\n'.join(transcripts)
    
    def _combine_audio_analysis(self, transcript: str, metadata: Dict, file_info: Dict) -> str:
        """Combine all audio analysis results."""
        sections = []
        
        # Audio properties
        props = []
        if metadata.get("duration"):
            duration_min = int(metadata["duration"] / 60)
            duration_sec = int(metadata["duration"] % 60)
            props.append(f"Czas trwania: {duration_min}:{duration_sec:02d}")
        if metadata.get("bitrate"):
            props.append(f"Bitrate: {metadata['bitrate']} bps")
        if metadata.get("sample_rate"):
            props.append(f"Częstotliwość: {metadata['sample_rate']} Hz")
        if metadata.get("channels"):
            channels_desc = "mono" if metadata["channels"] == 1 else f"{metadata['channels']} kanały"
            props.append(f"Kanały: {channels_desc}")
        
        if props:
            sections.append("🎵 **Właściwości audio:**\n" + "\n".join(f"- {p}" for p in props))
        
        # ID3 metadata
        id3_props = []
        if metadata.get("title"):
            id3_props.append(f"Tytuł: {metadata['title']}")
        if metadata.get("artist"):
            id3_props.append(f"Wykonawca: {metadata['artist']}")
        if metadata.get("album"):
            id3_props.append(f"Album: {metadata['album']}")
        if metadata.get("date"):
            id3_props.append(f"Data: {metadata['date']}")
        if metadata.get("genre"):
            id3_props.append(f"Gatunek: {metadata['genre']}")
        
        if id3_props:
            sections.append("\n🏷️ **Metadane:**\n" + "\n".join(f"- {p}" for p in id3_props))
        
        # Transcript
        if transcript:
            word_count = len(transcript.split())
            sections.append(f"\n📝 **Transkrypcja ({word_count} słów):**\n{transcript}")
        
        return "\n".join(sections) if sections else "Brak wyekstraktowanej treści."
    
    async def _generate_audio_lesson_with_ai(self, content: str, metadata: Dict) -> Dict[str, Any]:
        """Generate audio lesson using LM Studio."""
        prompt = f"""Analyze this educational audio content and create a structured lesson in Polish:

Audio duration: {metadata.get('duration', 0) / 60:.1f} minutes
Title: {metadata.get('title', 'Unknown')}
Artist: {metadata.get('artist', 'Unknown')}
Content: {content[:3000]}

Generate:
1. Meaningful title based on the content
2. Summary (2-3 sentences)
3. Main topics covered
4. Relevant tags
5. Educational categories

Format as JSON."""
        
        try:
            response = await self.lm_studio.generate_lesson_from_prompt(prompt)
            return response
        except Exception as e:
            logger.error(f"LM Studio generation failed: {e}")
            return {}
    
    def _generate_smart_audio_title(self, transcript: str, metadata: Dict, file_path: str) -> str:
        """Generate intelligent title from audio content."""
        # Use ID3 title if available and meaningful
        id3_title = metadata.get('title', '').strip()
        if id3_title and len(id3_title) > 3:
            return id3_title
        
        # Extract title from transcript
        if transcript:
            sentences = transcript.split('.')[:3]
            for sentence in sentences:
                sentence = sentence.strip()
                if 10 < len(sentence) < 100:
                    # Check if it mentions the topic
                    topic_keywords = ['wykład', 'lekcja', 'podcast', 'wywiad', 'prezentacja']
                    if any(keyword in sentence.lower() for keyword in topic_keywords):
                        return sentence
        
        # Fallback with duration
        duration_min = int(metadata.get('duration', 0) / 60)
        base_title = self.generate_title_from_filename(file_path)
        return f"{base_title} ({duration_min} min)"
    
    def _generate_audio_summary(self, transcript: str, metadata: Dict, file_info: Dict) -> str:
        """Generate audio summary."""
        summaries = []
        
        # Duration and format
        duration_min = int(metadata.get('duration', 0) / 60)
        format_info = metadata.get('format', file_info.get('extension', '').replace('.', ''))
        summaries.append(f"Materiał audio {format_info.upper()} ({duration_min} min)")
        
        # Transcript summary
        if transcript:
            word_count = len(transcript.split())
            summaries.append(f"Transkrypcja zawiera {word_count} słów")
        
        return ". ".join(summaries) + "."
    
    def _generate_audio_tags(self, transcript: str, metadata: Dict, file_info: Dict) -> List[str]:
        """Generate audio tags."""
        tags = ["auto-generated", "audio"]
        
        # Duration-based tags
        duration_min = metadata.get('duration', 0) / 60
        if duration_min < 10:
            tags.append("short-audio")
        elif duration_min > 60:
            tags.append("long-audio")
        
        # Content-based tags from transcript
        if transcript:
            text_lower = transcript.lower()
            
            tag_keywords = {
                "podcast": ['podcast', 'odcinek', 'witajcie'],
                "wykład": ['wykład', 'lekcja', 'nauczymy'],
                "wywiad": ['wywiad', 'rozmowa', 'gość'],
                "programowanie": ['python', 'java', 'kod', 'function'],
                "biznes": ['firma', 'zarządzanie', 'marketing'],
                "edukacja": ['nauka', 'szkoła', 'student']
            }
            
            for tag, keywords in tag_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    tags.append(tag)
        
        # Quality tags
        if transcript:
            tags.append("with-transcript")
        
        # ID3 tags
        if metadata.get('artist'):
            tags.append("has-metadata")
        
        return list(set(tags))
    
    def _detect_audio_categories(self, transcript: str, metadata: Dict) -> List[str]:
        """Detect audio categories."""
        categories = []
        
        if transcript:
            text_lower = transcript.lower()
            
            category_keywords = {
                "Programowanie": ['python', 'java', 'javascript', 'kod', 'programowanie'],
                "Biznes": ['biznes', 'firma', 'zarządzanie', 'marketing', 'strategia'],
                "Edukacja": ['edukacja', 'nauka', 'szkoła', 'nauczanie', 'student'],
                "Technologia": ['technologia', 'komputer', 'internet', 'ai', 'sztuczna inteligencja'],
                "Podcasts": ['podcast', 'odcinek', 'rozmowa', 'wywiad'],
                "Nauki": ['nauka', 'badanie', 'eksperyment', 'teoria']
            }
            
            for category, keywords in category_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    categories.append(category)
        
        # Default category
        if not categories:
            categories.append("Materiały audio")
        
        return categories[:3]
    
    def _calculate_audio_confidence(self, transcript: str, metadata: Dict) -> float:
        """Calculate extraction confidence for audio."""
        confidence = 0.4  # Base
        
        # Transcript quality
        if transcript:
            word_count = len(transcript.split())
            if word_count > 500:
                confidence += 0.3
            elif word_count > 200:
                confidence += 0.2
            elif word_count > 50:
                confidence += 0.1
        
        # Metadata quality
        if metadata.get('duration') and metadata.get('duration') > 0:
            confidence += 0.1
        
        if metadata.get('title') or metadata.get('artist'):
            confidence += 0.1
        
        if metadata.get('bitrate') and metadata.get('bitrate') > 64000:
            confidence += 0.1
        
        return min(confidence, 0.95)
    
    async def _create_large_audio_preview(self, file_path: str, file_info: Dict, metadata: Dict) -> ExtractedContent:
        """Create preview for very large audio files."""
        title = f"[DUŻY PLIK] {self.generate_title_from_filename(file_path)}"
        size_mb = file_info['size'] / (1024*1024)
        duration_min = int(metadata.get('duration', 0) / 60)
        
        content = f"""🎵 **Duży plik audio**

**Rozmiar:** {size_mb:.1f} MB
**Czas trwania:** {duration_min} minut
**Format:** {metadata.get('format', 'unknown')}
**Bitrate:** {metadata.get('bitrate', 0)} bps

⚠️ **Plik zbyt duży do automatycznej analizy**

**Opcje przetwarzania:**
1. 🎵 Analiza pierwszych 10 minut
2. 📝 Transkrypcja w partiach (15 min każda)
3. 🎯 Transkrypcja wybranych fragmentów
4. 💾 Pełne przetwarzanie w tle (może trwać {duration_min} minut)

**Zalecenia:**
- Odsłuchaj fragment i dodaj ręczne notatki
- Wybierz kluczowe fragmenty do transkrypcji
- Rozważ podział na mniejsze części"""
        
        summary = f"Duży materiał audio ({size_mb:.0f} MB, {duration_min} min). Wymaga ręcznego wyboru metody przetwarzania."
        
        return ExtractedContent(
            title=title,
            summary=summary,
            content=content,
            metadata={
                **file_info,
                **metadata,
                "extraction_method": "large_audio_preview",
                "requires_manual_processing": True,
                "suggested_tags": ["large-file", "audio", "needs-processing"],
                "suggested_categories": ["Materiały audio"],
                "confidence": 0.4
            },
            confidence=0.4
        ) 