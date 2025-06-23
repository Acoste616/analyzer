"""Enhanced video extractor with Whisper transcription."""

import asyncio
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import subprocess
import json
import os
import torch
from datetime import datetime

from .base_extractor import BaseExtractor, ExtractedContent
from ..core.logger import get_logger
from ..processors.lm_studio import LMStudioProcessor
from ..core.system_paths import get_tool_path, is_tool_available, test_tool

logger = get_logger(__name__)


class EnhancedVideoExtractor(BaseExtractor):
    """Enhanced video extractor with audio transcription and analysis."""
    
    SUPPORTED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
    
    def __init__(self):
        super().__init__()
        self.lm_studio = LMStudioProcessor()
        self._init_whisper()
        self._init_ffmpeg()
    
    def supports_file(self, file_path: str) -> bool:
        """Check if this extractor supports the file type."""
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS
    
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
        """Check if ffmpeg is available using system path manager."""
        self._ffmpeg_available = is_tool_available("ffmpeg") and is_tool_available("ffprobe")
        self._ffmpeg_path = get_tool_path("ffmpeg")
        self._ffprobe_path = get_tool_path("ffprobe")
        
        if self._ffmpeg_available:
            # Test if tools actually work
            ffmpeg_works = test_tool("ffmpeg")
            ffprobe_works = test_tool("ffprobe")
            
            if ffmpeg_works and ffprobe_works:
                logger.info(f"FFmpeg available at: {self._ffmpeg_path}")
                logger.info(f"FFprobe available at: {self._ffprobe_path}")
            else:
                self._ffmpeg_available = False
                logger.warning("FFmpeg tools found but not working properly")
        else:
            logger.warning("FFmpeg not available - video processing limited")
            logger.info("Install FFmpeg: https://ffmpeg.org/download.html#build-windows")
    
    async def extract(self, file_path: str) -> ExtractedContent:
        """Extract content from video with transcription."""
        file_info = self.get_file_info(file_path)
        
        # Extract video metadata
        video_metadata = await self._extract_video_metadata(file_path)
        
        # For large files, decide on processing strategy
        size_mb = file_info['size'] / (1024 * 1024)
        
        if size_mb > 500:  # Very large file
            return await self._create_large_file_preview(file_path, file_info, video_metadata)
        
        # Extract audio and transcribe with hallucination detection
        transcript_data = await self._extract_audio_transcript(file_path)
        transcript = transcript_data['transcript']
        logger.info(f"Transcript extracted: {len(transcript)} characters, method: {transcript_data.get('processing_method', 'unknown')}")
        
        # Extract keyframes for visual analysis
        keyframes_analysis = await self._analyze_keyframes(file_path, video_metadata)
        
        # Combine all information
        combined_content = self._combine_video_analysis(
            transcript, keyframes_analysis, video_metadata
        )
        
        # Generate lesson with AI if available
        if await self.lm_studio.is_available() and combined_content:
            lesson_data = await self._generate_video_lesson_with_ai(
                combined_content, video_metadata
            )
            title = lesson_data.get("title", self._generate_smart_video_title(transcript, video_metadata, file_path))
            summary = lesson_data.get("summary", "")
            content = lesson_data.get("content", combined_content)
            tags = lesson_data.get("tags", [])
            categories = lesson_data.get("categories", [])
        else:
            # Intelligent processing without AI
            title = self._generate_smart_video_title(transcript, video_metadata, file_path)
            summary = self._generate_video_summary(transcript, video_metadata, keyframes_analysis)
            content = combined_content
            tags = self._generate_video_tags(transcript, video_metadata, keyframes_analysis)
            categories = self._detect_video_categories(transcript, video_metadata)
        
        confidence = self._calculate_video_confidence(transcript, video_metadata, keyframes_analysis)
        
        return ExtractedContent(
            title=title,
            summary=summary,
            content=content,
            metadata={
                **file_info,
                **video_metadata,
                "extraction_method": "video_transcription_analysis",
                "has_transcript": bool(transcript),
                "transcript_length": len(transcript) if transcript else 0,
                "keyframes_analyzed": len(keyframes_analysis.get("frames", [])),
                "suggested_tags": tags,
                "suggested_categories": categories,
                "confidence": confidence
            },
            confidence=confidence
        )
    
    async def extract_with_memory_management(self, file_path: str) -> ExtractedContent:
        """Extract with automatic memory management"""
        try:
            # Clear GPU cache before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            result = await self.extract(file_path)
            
            # Clear cache after processing large files
            file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            if file_size_mb > 100 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return result
            
        except torch.cuda.OutOfMemoryError:
            logger.error(f"GPU OOM for {file_path}, falling back to CPU")
            torch.cuda.empty_cache()
            
            # Retry with CPU
            original_device = self._whisper_model.device if hasattr(self, '_whisper_model') else None
            if original_device:
                self._whisper_model = self._whisper_model.to('cpu')
            
            result = await self.extract(file_path)
            
            # Restore to GPU if possible
            if original_device and torch.cuda.is_available():
                self._whisper_model = self._whisper_model.to(original_device)
                
            return result
    
    async def _extract_video_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract detailed video metadata using ffprobe."""
        metadata = {
            "duration": 0,
            "fps": 0,
            "resolution": "unknown",
            "codec": "unknown",
            "bitrate": 0,
            "has_audio": False
        }
        
        if not self._ffmpeg_available:
            return metadata
        
        try:
            cmd = [
                self._ffprobe_path,
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
                    if stream['codec_type'] == 'video':
                        metadata['fps'] = eval(stream.get('r_frame_rate', '0/1'))
                        metadata['resolution'] = f"{stream.get('width', 0)}x{stream.get('height', 0)}"
                        metadata['codec'] = stream.get('codec_name', 'unknown')
                    elif stream['codec_type'] == 'audio':
                        metadata['has_audio'] = True
                        metadata['audio_codec'] = stream.get('codec_name', 'unknown')
                        metadata['audio_bitrate'] = int(stream.get('bit_rate', 0))
        
        except Exception as e:
            logger.error(f"Failed to extract video metadata: {e}")
        
        return metadata
    
    async def _extract_and_transcribe_audio(self, file_path: str, metadata: Dict) -> str:
        """Extract audio and transcribe using Whisper."""
        if not self._whisper_available or not metadata.get('has_audio'):
            return ""
        
        try:
            # For videos longer than 30 minutes, process in chunks
            duration = metadata.get('duration', 0)
            if duration > 1800:  # 30 minutes
                return await self._transcribe_long_video(file_path, duration)
            
            # Extract audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            # Extract audio using ffmpeg
            cmd = [
                self._ffmpeg_path,
                '-i', file_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM codec
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite
                temp_audio_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
            if process.returncode == 0:
                # Transcribe with Whisper
                import whisper
                result = self._whisper_model.transcribe(
                    temp_audio_path,
                    language='pl',  # Force Polish
                    task='transcribe'
                )
                
                # Clean up
                Path(temp_audio_path).unlink(missing_ok=True)
                
                return result.get('text', '')
            
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
        
        return ""
    
    async def _transcribe_long_video(self, file_path: str, duration: float) -> str:
        """Transcribe long video in chunks."""
        transcripts = []
        chunk_duration = 600  # 10 minutes per chunk
        
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
                    '-vn',
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
                logger.error(f"Failed to transcribe chunk at {start_time}: {e}")
        
        return '\n\n'.join(transcripts)
    
    async def _analyze_keyframes(self, file_path: str, metadata: Dict) -> Dict[str, Any]:
        """Extract and analyze keyframes from video with progressive processing for long videos."""
        analysis = {
            "frames": [],
            "detected_content": [],
            "visual_summary": ""
        }
        
        if not self._ffmpeg_available:
            return analysis
        
        try:
            # Progresywne przetwarzanie keyframes na podstawie długości
            duration = metadata.get('duration', 60)
            
            # Dla długich video - mniej keyframes
            if duration > 7200:  # 2h+
                max_frames = 10
                interval = duration / max_frames
                logger.info(f"Long video mode: extracting {max_frames} keyframes from {duration/3600:.1f}h video")
            elif duration > 3600:  # 1h+
                max_frames = 20
                interval = duration / max_frames
                logger.info(f"Medium video mode: extracting {max_frames} keyframes from {duration/60:.1f}min video")
            else:
                max_frames = 30
                interval = max(30, duration / max_frames)  # Min 30s between frames
                logger.info(f"Short video mode: extracting {max_frames} keyframes from {duration/60:.1f}min video")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract keyframes with optimized processing
                for i in range(max_frames):
                    timestamp = i * interval
                    frame_path = Path(temp_dir) / f"frame_{i:03d}.jpg"
                    
                    cmd = [
                        self._ffmpeg_path,
                        '-ss', str(timestamp),
                        '-i', file_path,
                        '-vframes', '1',
                        '-q:v', '2',
                        '-y',
                        str(frame_path)
                    ]
                    
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    await process.communicate()
                    
                    if process.returncode == 0 and frame_path.exists():
                        # Analyze frame with image extractor
                        from .enhanced_image_extractor import EnhancedImageExtractor
                        image_extractor = EnhancedImageExtractor()
                        frame_content = await image_extractor.extract(str(frame_path))
                        
                        analysis["frames"].append({
                            "timestamp": timestamp,
                            "content": frame_content.content,
                            "detected_text": bool(frame_content.metadata.get("ocr_text_length", 0) > 10)
                        })
                        
                        # Detect content types
                        if "code-screenshot" in frame_content.metadata.get("visual_features", []):
                            analysis["detected_content"].append("code")
                        if "likely-presentation" in frame_content.metadata.get("visual_features", []):
                            analysis["detected_content"].append("presentation")
            
            # Generate visual summary
            if analysis["frames"]:
                if "presentation" in analysis["detected_content"]:
                    analysis["visual_summary"] = "Wideo zawiera slajdy prezentacji"
                elif "code" in analysis["detected_content"]:
                    analysis["visual_summary"] = "Wideo zawiera fragmenty kodu"
                else:
                    analysis["visual_summary"] = "Wideo z treścią wizualną"
        
        except Exception as e:
            logger.error(f"Keyframe analysis failed: {e}")
        
        return analysis
    
    def _combine_video_analysis(self, transcript: str, keyframes: Dict, metadata: Dict) -> str:
        """Combine all video analysis results."""
        sections = []
        
        # Video properties
        props = []
        if metadata.get("duration"):
            duration_min = int(metadata["duration"] / 60)
            duration_sec = int(metadata["duration"] % 60)
            props.append(f"Czas trwania: {duration_min}:{duration_sec:02d}")
        if metadata.get("resolution") != "unknown":
            props.append(f"Rozdzielczość: {metadata['resolution']}")
        if metadata.get("fps"):
            props.append(f"FPS: {metadata['fps']}")
        
        if props:
            sections.append("📹 **Właściwości wideo:**\n" + "\n".join(f"- {p}" for p in props))
        
        # Transcript
        if transcript:
            word_count = len(transcript.split())
            sections.append(f"\n📝 **Transkrypcja ({word_count} słów):**\n{transcript}")
        
        # Visual analysis
        if keyframes.get("visual_summary"):
            sections.append(f"\n🎨 **Analiza wizualna:** {keyframes['visual_summary']}")
        
        # Detected content in frames
        if keyframes.get("frames"):
            frames_with_text = sum(1 for f in keyframes["frames"] if f.get("detected_text"))
            if frames_with_text > 0:
                sections.append(f"\n📊 **Wykryto tekst w {frames_with_text} klatkach**")
        
        return "\n".join(sections) if sections else "Brak wyekstraktowanej treści."
    
    async def _generate_video_lesson_with_ai(self, content: str, metadata: Dict) -> Dict[str, Any]:
        """Generate video lesson using LM Studio."""
        prompt = f"""Analyze this educational video content and create a structured lesson in Polish:

Video duration: {metadata.get('duration', 0) / 60:.1f} minutes
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
    
    def _generate_smart_video_title(self, transcript: str, metadata: Dict, file_path: str) -> str:
        """Generate intelligent title from video content."""
        if transcript:
            # Extract potential title from first sentences
            sentences = transcript.split('.')[:3]
            for sentence in sentences:
                sentence = sentence.strip()
                if 10 < len(sentence) < 100:
                    # Check if it mentions the topic
                    topic_keywords = ['wykład', 'lekcja', 'tutorial', 'kurs', 'prezentacja']
                    if any(keyword in sentence.lower() for keyword in topic_keywords):
                        return sentence
            
            # Extract topic from content
            words = transcript.lower().split()
            common_topics = {
                'programowanie': ['python', 'java', 'javascript', 'kod', 'funkcja'],
                'matematyka': ['równanie', 'funkcja', 'liczba', 'obliczenia'],
                'nauka': ['eksperyment', 'badanie', 'teoria', 'hipoteza']
            }
            
            for topic, keywords in common_topics.items():
                if any(keyword in words for keyword in keywords):
                    duration_min = int(metadata.get('duration', 0) / 60)
                    return f"Wykład: {topic.capitalize()} ({duration_min} min)"
        
        # Fallback with duration
        duration_min = int(metadata.get('duration', 0) / 60)
        base_title = self.generate_title_from_filename(file_path)
        return f"{base_title} ({duration_min} min)"
    
    def _generate_video_summary(self, transcript: str, metadata: Dict, keyframes: Dict) -> str:
        """Generate video summary."""
        summaries = []
        
        # Duration
        duration_min = int(metadata.get('duration', 0) / 60)
        summaries.append(f"Materiał wideo ({duration_min} min)")
        
        # Transcript summary
        if transcript:
            word_count = len(transcript.split())
            summaries.append(f"Transkrypcja zawiera {word_count} słów")
        
        # Visual content
        if keyframes.get("visual_summary"):
            summaries.append(keyframes["visual_summary"])
        
        # Audio presence
        if not metadata.get('has_audio'):
            summaries.append("Brak ścieżki audio")
        
        return ". ".join(summaries) + "."
    
    def _generate_video_tags(self, transcript: str, metadata: Dict, keyframes: Dict) -> List[str]:
        """Generate video tags."""
        tags = ["auto-generated", "video"]
        
        # Duration-based tags
        duration_min = metadata.get('duration', 0) / 60
        if duration_min < 5:
            tags.append("short-video")
        elif duration_min > 30:
            tags.append("long-video")
        
        # Content-based tags from transcript
        if transcript:
            text_lower = transcript.lower()
            
            tag_keywords = {
                "programowanie": ['python', 'java', 'kod', 'function', 'class'],
                "tutorial": ['tutorial', 'poradnik', 'jak', 'krok po kroku'],
                "wykład": ['wykład', 'lekcja', 'prezentacja'],
                "matematyka": ['równanie', 'funkcja', 'obliczenia'],
                "nauka": ['eksperyment', 'badanie', 'teoria']
            }
            
            for tag, keywords in tag_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    tags.append(tag)
        
        # Visual content tags
        if "code" in keyframes.get("detected_content", []):
            tags.append("screencast")
        if "presentation" in keyframes.get("detected_content", []):
            tags.append("prezentacja")
        
        # Quality tags
        if transcript:
            tags.append("with-transcript")
        
        confidence = metadata.get('confidence', 0)
        if confidence > 0.8:
            tags.append("high-confidence")
        elif confidence > 0.5:
            tags.append("medium-confidence")
        else:
            tags.append("needs-review")
        
        return list(set(tags))
    
    def _detect_video_categories(self, transcript: str, metadata: Dict) -> List[str]:
        """Detect video categories."""
        categories = []
        
        if transcript:
            text_lower = transcript.lower()
            
            # Same category detection as images
            category_keywords = {
                "Programowanie": ['python', 'java', 'javascript', 'kod', 'programowanie', 'developer'],
                "Matematyka": ['matematyka', 'równanie', 'liczba', 'wykres', 'funkcja'],
                "Nauki ścisłe": ['fizyka', 'chemia', 'biologia', 'eksperyment', 'laboratorium'],
                "Języki": ['język', 'gramatyka', 'słownictwo', 'english', 'deutsch'],
                "Biznes": ['biznes', 'marketing', 'zarządzanie', 'firma', 'strategia'],
                "Tutoriale": ['tutorial', 'poradnik', 'jak', 'krok po kroku', 'instrukcja']
            }
            
            for category, keywords in category_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    categories.append(category)
        
        # Video-specific category
        categories.append("Materiały wideo")
        
        return categories[:3]
    
    def _calculate_video_confidence(self, transcript: str, metadata: Dict, keyframes: Dict) -> float:
        """Calculate extraction confidence for video."""
        confidence = 0.3  # Base
        
        # Transcript quality
        if transcript:
            word_count = len(transcript.split())
            if word_count > 1000:
                confidence += 0.4
            elif word_count > 500:
                confidence += 0.3
            elif word_count > 100:
                confidence += 0.2
            else:
                confidence += 0.1
        
        # Metadata quality
        if metadata.get('duration') and metadata.get('resolution') != "unknown":
            confidence += 0.1
        
        # Keyframe analysis
        if keyframes.get("frames"):
            confidence += 0.1
        
        if keyframes.get("detected_content"):
            confidence += 0.1
        
        return min(confidence, 0.95)
    
    async def _create_large_file_preview(self, file_path: str, file_info: Dict, metadata: Dict) -> ExtractedContent:
        """Create preview for very large video files."""
        title = f"[DUŻY PLIK] {self.generate_title_from_filename(file_path)}"
        size_mb = file_info['size'] / (1024*1024)
        duration_min = int(metadata.get('duration', 0) / 60)
        
        content = f"""📹 **Duży plik wideo**

**Rozmiar:** {size_mb:.1f} MB
**Czas trwania:** {duration_min} minut
**Rozdzielczość:** {metadata.get('resolution', 'unknown')}

⚠️ **Plik zbyt duży do automatycznej analizy**

**Opcje przetwarzania:**
1. 🎬 Analiza pierwszych 5 minut
2. 📊 Ekstrakcja klatek kluczowych
3. 🎯 Transkrypcja wybranych fragmentów
4. 💾 Pełne przetwarzanie w tle (może trwać {duration_min * 2} minut)

**Zalecenia:**
- Obejrzyj wideo i dodaj ręczne notatki
- Wybierz kluczowe fragmenty do transkrypcji
- Rozważ podział na mniejsze części"""
        
        summary = f"Duży materiał wideo ({size_mb:.0f} MB, {duration_min} min). Wymaga ręcznego wyboru metody przetwarzania."
        
        return ExtractedContent(
            title=title,
            summary=summary,
            content=content,
            metadata={
                **file_info,
                **metadata,
                "extraction_method": "large_file_preview",
                "requires_manual_processing": True,
                "suggested_tags": ["large-file", "video", "needs-processing"],
                "suggested_categories": ["Materiały wideo"],
                "confidence": 0.4
            },
            confidence=0.4
        )

    def _clean_whisper_transcript(self, transcript: str) -> str:
        """Clean and validate Whisper transcript to remove hallucinations."""
        if not transcript or not transcript.strip():
            return ""
        
        import re
        from collections import Counter
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', transcript.strip())
        
        # Detect hallucination patterns (repeated words/phrases)
        words = cleaned.split()
        
        # Check for excessive repetition of single words
        if len(words) > 10:
            word_counts = Counter(words)
            total_words = len(words)
            
            # If any single word appears more than 30% of the time, it's likely hallucination
            for word, count in word_counts.items():
                if count / total_words > 0.3 and len(word) <= 3:
                    logger.warning(f"Detected Whisper hallucination: '{word}' repeated {count} times")
                    return ""
        
        # Check for repetitive patterns like "w w w w w"
        if len(set(words)) < len(words) * 0.1:  # Less than 10% unique words
            logger.warning("Detected Whisper hallucination: excessive repetition")
            return ""
        
        # Check for meaningless patterns
        meaningless_patterns = [
            r'^[\s\w\-\:]*(\d+\-\d+\-\d+:\s*)*\d+\-\d+\-\d+.*$',  # Pattern like "3-2-1: 3-3-3:"
            r'^[w\s]*$',  # Only 'w' and spaces
            r'^[a-z]\s+[a-z]\s+[a-z].*$',  # Single letters repeated
        ]
        
        for pattern in meaningless_patterns:
            if re.match(pattern, cleaned, re.IGNORECASE):
                logger.warning(f"Detected meaningless Whisper pattern: {cleaned[:50]}...")
                return ""
        
        # Additional cleanup
        # Remove timestamps and artifacts
        cleaned = re.sub(r'\d+:\d+:\d+', '', cleaned)
        cleaned = re.sub(r'\[.*?\]', '', cleaned)
        cleaned = re.sub(r'\(.*?\)', '', cleaned)
        
        # Remove very short meaningless outputs
        if len(cleaned.strip()) < 10:
            return ""
        
        return cleaned.strip()

    async def _extract_audio_transcript(self, file_path: str) -> Dict[str, Any]:
        """Extract audio with intelligent sampling for long videos"""
        
        # Sprawdź długość video
        metadata = await self._extract_video_metadata(file_path)
        duration = metadata.get('duration', 0)
        
        if duration > 7200:  # Ponad 2 godziny
            # Próbkuj co 10 minut zamiast całości
            return await self._extract_sampled_transcript(file_path, duration)
        else:
            # Normalna ekstrakcja dla krótszych
            return await self._extract_full_transcript(file_path)

    async def _extract_full_transcript(self, file_path: str) -> Dict[str, Any]:
        """Extract full audio transcript with hallucination detection."""
        transcript_data = {
            'transcript': '',
            'language': 'unknown',
            'confidence': 0.0,
            'duration': 0.0,
            'processing_method': 'whisper'
        }
        
        if not self._whisper_available:
            logger.warning("Whisper model not available")
            return transcript_data
            
        try:
            # Extract audio using FFmpeg
            temp_audio = None
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    temp_audio = tmp.name
                
                # Enhanced FFmpeg command for better audio extraction
                ffmpeg_cmd = [
                    self._ffmpeg_path, '-y',
                    '-i', file_path,
                    '-vn',  # No video
                    '-acodec', 'pcm_s16le',  # 16-bit PCM
                    '-ar', '16000',  # 16kHz (Whisper optimal)
                    '-ac', '1',  # Mono
                    '-af', 'volume=2.0,highpass=f=80,lowpass=f=8000',  # Audio enhancement
                    temp_audio
                ]
                
                result = subprocess.run(
                    ffmpeg_cmd,
                    capture_output=True,
                    text=True,
                    timeout=120  # 2 minutes for audio extraction
                )
                
                if result.returncode != 0:
                    logger.error(f"FFmpeg audio extraction failed: {result.stderr}")
                    return transcript_data
                
                # Check if audio file was created and has content
                if not os.path.exists(temp_audio) or os.path.getsize(temp_audio) < 1000:
                    logger.warning(f"No audio content extracted from {file_path}")
                    transcript_data['transcript'] = "Brak dźwięku w pliku wideo"
                    return transcript_data
                
                # Transcribe with Whisper
                logger.info("Rozpoczynam transkrypcję Whisper...")
                
                # Use Whisper with specific parameters to reduce hallucination
                result = self._whisper_model.transcribe(
                    temp_audio,
                    language='pl',  # Force Polish to reduce language confusion
                    task='transcribe',
                    verbose=False,
                    temperature=0.0,  # Reduce randomness
                    best_of=1,  # Don't sample multiple times
                    beam_size=5,  # Conservative beam search
                    patience=1.0,
                    length_penalty=1.0,
                    suppress_tokens="",
                    initial_prompt="",  # No initial prompt to avoid biasing
                    condition_on_previous_text=False,  # Prevent context bleeding
                    fp16=torch.cuda.is_available(),
                    compression_ratio_threshold=2.4,  # Detect repetitive text
                    logprob_threshold=-1.0,  # Higher threshold for confidence
                    no_speech_threshold=0.6,  # Higher threshold for silence detection
                )
                
                raw_transcript = result.get('text', '').strip()
                
                # Apply hallucination cleaning
                cleaned_transcript = self._clean_whisper_transcript(raw_transcript)
                
                if cleaned_transcript:
                    transcript_data.update({
                        'transcript': cleaned_transcript,
                        'language': result.get('language', 'pl'),
                        'confidence': 0.8,  # High confidence for clean transcript
                        'duration': result.get('duration', 0.0),
                        'processing_method': 'whisper_enhanced'
                    })
                    logger.info(f"Transkrypcja zakończona: {len(cleaned_transcript)} znaków")
                else:
                    # If cleaning removed everything, it was likely hallucination
                    transcript_data.update({
                        'transcript': "Wykryto halucynację Whisper - prawdopodobnie brak wyraźnej mowy w audio",
                        'confidence': 0.1,
                        'processing_method': 'whisper_hallucination_detected'
                    })
                    logger.warning("Whisper transcript cleaned due to hallucination detection")
                
            finally:
                # Clean up temp file
                if temp_audio and os.path.exists(temp_audio):
                    try:
                        os.unlink(temp_audio)
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"Transcript extraction failed: {e}")
            transcript_data['transcript'] = f"Błąd transkrypcji: {str(e)}"
            
        return transcript_data
    
    async def _extract_sampled_transcript(self, file_path: str, duration: float) -> Dict[str, Any]:
        """Próbkuj długie video co N minut"""
        logger.info(f"Rozpoczynam próbkowaną transkrypcję dla {duration/3600:.1f}h video")
        
        samples = []
        sample_duration = 120  # 2 minuty próbki
        sample_interval = 600  # Co 10 minut
        
        # Check for existing checkpoint
        checkpoint_data = await self._load_checkpoint(file_path)
        start_time = 0
        if checkpoint_data:
            start_time = checkpoint_data.get('last_processed_time', 0)
            samples = checkpoint_data.get('samples', [])
            logger.info(f"Resuming from checkpoint at {start_time/60:.1f} minutes")
        
        for current_time in range(start_time, int(duration), sample_interval):
            try:
                # Save checkpoint before processing
                await self._save_checkpoint(file_path, {
                    'last_processed_time': current_time,
                    'samples': samples,
                    'total_duration': duration
                })
                
                # Ekstraktuj fragment
                sample = await self._extract_video_segment(
                    file_path, current_time, sample_duration
                )
                
                if sample['transcript']:
                    samples.append({
                        'timestamp': current_time,
                        'time_formatted': f"{current_time//3600:02d}:{(current_time%3600)//60:02d}:{current_time%60:02d}",
                        'text': sample['transcript'],
                        'confidence': sample.get('confidence', 0.5)
                    })
                    logger.info(f"Processed sample at {current_time//60:.1f}min: {len(sample['transcript'])} chars")
                
                # Clear GPU memory periodically
                if torch.cuda.is_available() and len(samples) % 5 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Failed to process sample at {current_time}: {e}")
                continue
        
        # Połącz próbki w spójną transkrypcję
        full_transcript = self._merge_samples(samples)
        
        # Clear checkpoint on completion
        await self._clear_checkpoint(file_path)
        
        return {
            'transcript': full_transcript,
            'method': 'sampled',
            'samples_count': len(samples),
            'total_duration': duration,
            'coverage_percent': (len(samples) * sample_duration / duration) * 100,
            'processing_method': 'whisper_sampled'
        }
    
    async def _extract_video_segment(self, file_path: str, start_time: int, duration: int) -> Dict[str, Any]:
        """Ekstraktuj i transkrybuj fragment video"""
        import tempfile
        
        segment_data = {
            'transcript': '',
            'confidence': 0.0,
            'duration': duration
        }
        
        temp_audio = None
        try:
            # Create temporary file for segment
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                temp_audio = tmp.name
            
            # Extract audio segment
            ffmpeg_cmd = [
                self._ffmpeg_path, '-y',
                '-ss', str(start_time),
                '-i', file_path,
                '-t', str(duration),
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-af', 'volume=2.0,highpass=f=80,lowpass=f=8000',
                temp_audio
            ]
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0 and os.path.exists(temp_audio) and os.path.getsize(temp_audio) > 1000:
                # Transcribe segment
                transcription = self._whisper_model.transcribe(
                    temp_audio,
                    language='pl',
                    task='transcribe',
                    verbose=False,
                    temperature=0.0,
                    condition_on_previous_text=False
                )
                
                raw_transcript = transcription.get('text', '').strip()
                cleaned_transcript = self._clean_whisper_transcript(raw_transcript)
                
                if cleaned_transcript:
                    segment_data.update({
                        'transcript': cleaned_transcript,
                        'confidence': 0.8
                    })
            
        except Exception as e:
            logger.error(f"Segment extraction failed: {e}")
        finally:
            if temp_audio and os.path.exists(temp_audio):
                try:
                    os.unlink(temp_audio)
                except:
                    pass
        
        return segment_data
    
    def _merge_samples(self, samples: List[Dict]) -> str:
        """Połącz próbki w spójną transkrypcję"""
        if not samples:
            return ""
        
        merged_parts = []
        for sample in samples:
            timestamp = sample['time_formatted']
            text = sample['text']
            merged_parts.append(f"[{timestamp}] {text}")
        
        return '\n\n'.join(merged_parts)
    
    def _get_checkpoint_file(self, file_path: str) -> Path:
        """Get checkpoint file for resume capability"""
        import hashlib
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()
        checkpoint_dir = Path("temp/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir / f"{file_hash}.json"

    async def _save_checkpoint(self, file_path: str, state: Dict):
        """Save processing checkpoint"""
        try:
            checkpoint_file = self._get_checkpoint_file(file_path)
            
            checkpoint_data = {
                'file_path': str(file_path),
                'state': state,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                
            logger.debug(f"Checkpoint saved: {checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    async def _load_checkpoint(self, file_path: str) -> Optional[Dict]:
        """Load processing checkpoint"""
        try:
            checkpoint_file = self._get_checkpoint_file(file_path)
            
            if not checkpoint_file.exists():
                return None
                
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                
            # Check if checkpoint is not too old (max 24 hours)
            from datetime import timedelta
            checkpoint_time = datetime.fromisoformat(checkpoint_data['timestamp'])
            if datetime.now() - checkpoint_time > timedelta(hours=24):
                await self._clear_checkpoint(file_path)
                return None
                
            logger.info(f"Loaded checkpoint from {checkpoint_time}")
            return checkpoint_data.get('state')
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    async def _clear_checkpoint(self, file_path: str):
        """Clear processing checkpoint"""
        try:
            checkpoint_file = self._get_checkpoint_file(file_path)
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                logger.debug("Checkpoint cleared")
        except Exception as e:
            logger.error(f"Failed to clear checkpoint: {e}") 