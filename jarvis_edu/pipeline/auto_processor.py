"""Main auto processor pipeline."""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
import json

from ..core.logger import get_logger
from ..storage.database import ProcessingQueue, FileStatus, ProcessingFile
from ..storage.knowledge_base import KnowledgeBase
from ..watcher.folder_monitor import FolderWatcher
from ..processors.lm_studio import LMStudioProcessor
from ..models.content import ExtractedContent, ContentType
from ..extractors import (
    EnhancedImageExtractor,
    EnhancedVideoExtractor,
    EnhancedAudioExtractor
)

logger = get_logger(__name__)


class AutoProcessor:
    """Main auto processor that coordinates all components."""
    
    def __init__(
        self,
        watch_folder: str = "C:\\Users\\barto\\OneDrive\\Pulpit\\mediapobrane\\pobrane_media",
        output_folder: str = "output"
    ):
        """Initialize auto processor."""
        self.watch_folder = Path(watch_folder)
        self.output_folder = Path(output_folder)
        
        # Initialize components
        self.processing_queue = ProcessingQueue()
        self.knowledge_base = KnowledgeBase()
        self.lm_studio = LMStudioProcessor()
        
        # Initialize content extractors
        self.extractors = [
            EnhancedImageExtractor(),
            EnhancedVideoExtractor(),
            EnhancedAudioExtractor()
        ]
        
        # Create output structure
        self.output_folder.mkdir(parents=True, exist_ok=True)
        (self.output_folder / "lessons").mkdir(exist_ok=True)
        (self.output_folder / "content").mkdir(exist_ok=True)
        
        # Initialize folder watcher
        self.folder_watcher = FolderWatcher(
            watch_path=str(self.watch_folder),
            processor_callback=self.process_file,
            db_queue=self.processing_queue
        )
        
        self.running = False
        self.processing_tasks = {}
        
        logger.info(f"AutoProcessor initialized")
    
    async def start(self):
        """Start the auto processor."""
        try:
            # Check LM Studio connection
            if not await self.lm_studio.is_available():
                logger.warning("LM Studio not available - will use fallback processing")
            else:
                logger.info("LM Studio connection verified")
            
            # Start folder watcher
            self.folder_watcher.start()
            
            self.running = True
            logger.info("AutoProcessor started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start AutoProcessor: {e}")
            raise
    
    def stop(self):
        """Stop the auto processor."""
        try:
            self.running = False
            self.folder_watcher.stop()
            logger.info("AutoProcessor stopped")
        except Exception as e:
            logger.error(f"Error stopping AutoProcessor: {e}")
    
    async def process_file(self, file_entry: ProcessingFile):
        """Process a single file through the pipeline."""
        file_id = file_entry.id
        filepath = file_entry.filepath
        
        try:
            logger.info(f"Starting processing: {Path(filepath).name}")
            
            # Update status to processing
            self.processing_queue.update_status(file_id, FileStatus.PROCESSING)
            
            # Analyze content and create intelligent lesson
            lesson_id = await self.analyze_and_create_lesson(file_entry)
            
            # Update status to completed
            self.processing_queue.update_status(
                file_id, 
                FileStatus.COMPLETED,
                metadata_update={
                    "lesson_id": lesson_id,
                    "completed_at": datetime.now().isoformat()
                }
            )
            
            logger.info(f"Successfully processed: {Path(filepath).name}")
            
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            
            self.processing_queue.update_status(
                file_id,
                FileStatus.FAILED,
                error_message=str(e)
            )
    
    async def analyze_and_create_lesson(self, file_entry: ProcessingFile) -> str:
        """Analyze file content and create intelligent lesson."""
        # Generate unique lesson ID with timestamp and UUID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        lesson_id = f"lesson_{timestamp}_{unique_id}"
        
        try:
            # Find appropriate extractor for file type
            extractor = self._get_extractor_for_file(file_entry.filepath)
            
            if extractor:
                # Extract content using appropriate extractor
                extracted_content = await extractor.extract(file_entry.filepath)
                
                # Guard against None content
                if extracted_content is None:
                    logger.warning(f"Extractor returned None content for {file_entry.filepath}")
                    return await self._create_fallback_lesson(file_entry, lesson_id)
                
                # Save full content to separate file to avoid database bloat
                content_file_path = await self._save_content_to_file(lesson_id, extracted_content.content)
                
                # Create lesson with extracted content (without storing raw content in metadata)
                lesson = self.knowledge_base.add_lesson(
                    lesson_id=lesson_id,
                    title=extracted_content.title,
                    summary=extracted_content.summary,
                    content_path=content_file_path,
                    tags=self._generate_tags(extracted_content),
                    categories=self._generate_categories(extracted_content),
                    difficulty=self._determine_difficulty(extracted_content),
                    duration=self._estimate_lesson_duration(extracted_content),
                    topics_count=1,
                    metadata={
                        "source_file": file_entry.filepath,
                        "content_type": file_entry.content_type.value,
                        "extraction_method": extracted_content.metadata.get("extraction_method", "unknown"),
                        "confidence": extracted_content.confidence,
                        "content_length": len(extracted_content.content) if extracted_content.content else 0,
                        "language": extracted_content.language,
                        # Include only lightweight metadata, not raw content
                        **{k: v for k, v in extracted_content.metadata.items() 
                           if k not in ['content', 'full_text', 'raw_data']}
                    }
                )
                
                logger.info(f"Created intelligent lesson: {lesson_id} - {extracted_content.title}")
                return lesson_id
                
            else:
                # Fallback to basic lesson if no extractor available
                return await self._create_fallback_lesson(file_entry, lesson_id)
                
        except Exception as e:
            logger.error(f"Failed to analyze content for {file_entry.filepath}: {e}")
            # Create fallback lesson
            return await self._create_fallback_lesson(file_entry, lesson_id)
    
    def _get_extractor_for_file(self, file_path: str) -> Optional[Any]:
        """Find appropriate extractor for file type."""
        for extractor in self.extractors:
            if extractor.supports_file(file_path):
                return extractor
        return None
    
    async def _save_content_to_file(self, lesson_id: str, content: str) -> str:
        """Save extracted content to separate file to avoid database bloat."""
        try:
            # Create content file path
            content_file = self.output_folder / "content" / f"{lesson_id}.txt"
            
            # Ensure directory exists
            content_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save content to file
            with open(content_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Return relative path
            return str(content_file.relative_to(self.output_folder))
            
        except Exception as e:
            logger.error(f"Failed to save content for lesson {lesson_id}: {e}")
            return ""  # Return empty string if save fails
    
    async def _create_fallback_lesson(self, file_entry: ProcessingFile, lesson_id: str) -> str:
        """Create fallback lesson when extraction fails."""
        from ..extractors.base_extractor import BaseExtractor
        base_extractor = BaseExtractor()
        title = base_extractor.generate_title_from_filename(file_entry.filepath)
        
        self.knowledge_base.add_lesson(
            lesson_id=lesson_id,
            title=title,
            summary=f"Materiał edukacyjny: {title}. Wymaga ręcznej analizy.",
            content_path="",
            tags=["needs-review", "auto-generated"],
            categories=["Nieskategoryzowane"],
            difficulty="intermediate",
            duration=30,
            topics_count=1,
            metadata={
                "source_file": file_entry.filepath,
                "content_type": file_entry.content_type.value,
                "extraction_method": "fallback",
                "confidence": 0.3
            }
        )
        
        logger.info(f"Created fallback lesson: {lesson_id}")
        return lesson_id
    
    def _generate_tags(self, extracted_content) -> List[str]:
        """Generate appropriate tags based on extracted content."""
        if extracted_content is None:
            return ["auto-generated", "fallback"]
            
        tags = ["auto-generated"]
        
        # Add tags based on extraction method
        method = extracted_content.metadata.get("extraction_method", "")
        if method == "ocr":
            tags.append("ocr-extracted")
        elif method == "transcription":
            tags.append("audio-transcribed")
        elif method in ["smart_preview", "audio_analysis"]:
            tags.append("analyzed")
        
        # Add confidence-based tags
        if extracted_content.confidence > 0.8:
            tags.append("high-confidence")
        elif extracted_content.confidence > 0.5:
            tags.append("medium-confidence")
        else:
            tags.append("low-confidence")
        
        # Add content-type tags
        if "video" in extracted_content.metadata.get("content_type", ""):
            tags.append("video")
        elif "audio" in extracted_content.metadata.get("content_type", ""):
            tags.append("audio")
        elif "image" in extracted_content.metadata.get("content_type", ""):
            tags.append("image")
        
        return tags
    
    def _generate_categories(self, extracted_content) -> List[str]:
        """Generate appropriate categories based on content."""
        if extracted_content is None:
            return ["Nieskategoryzowane"]
            
        # Look for educational keywords in content
        content_lower = (extracted_content.content or "").lower()
        title_lower = (extracted_content.title or "").lower()
        
        categories = []
        
        # Subject detection
        if any(word in content_lower + title_lower for word in ['matematyka', 'math', 'liczby']):
            categories.append("Matematyka")
        elif any(word in content_lower + title_lower for word in ['programowanie', 'kod', 'python', 'javascript']):
            categories.append("Programowanie")
        elif any(word in content_lower + title_lower for word in ['historia', 'history']):
            categories.append("Historia")
        elif any(word in content_lower + title_lower for word in ['język', 'language', 'gramatyka']):
            categories.append("Języki")
        elif any(word in content_lower + title_lower for word in ['nauka', 'science', 'fizyka', 'chemia']):
            categories.append("Nauki ścisłe")
        
        # Format-based categories
        if extracted_content.metadata.get("extraction_method") == "ocr":
            categories.append("Materiały wizualne")
        elif "video" in extracted_content.metadata.get("content_type", ""):
            categories.append("Materiały wideo")
        elif "audio" in extracted_content.metadata.get("content_type", ""):
            categories.append("Materiały audio")
        
        # Default category if none detected
        if not categories:
            categories.append("Ogólne")
        
        return categories
    
    def _determine_difficulty(self, extracted_content) -> str:
        """Determine lesson difficulty based on content analysis."""
        if extracted_content is None or not extracted_content.content:
            return "intermediate"  # Default difficulty
            
        content = extracted_content.content.lower()
        
        # Basic difficulty indicators
        beginner_indicators = ['podstawy', 'wprowadzenie', 'podstawowe', 'basic', 'intro']
        advanced_indicators = ['zaawansowane', 'advanced', 'ekspert', 'complex', 'złożone']
        
        if any(indicator in content for indicator in advanced_indicators):
            return "advanced"
        elif any(indicator in content for indicator in beginner_indicators):
            return "beginner"
        else:
            return "intermediate"
    
    def _estimate_lesson_duration(self, extracted_content) -> int:
        """Estimate lesson duration based on content."""
        if extracted_content is None:
            return 30  # Default duration
            
        # Use duration from metadata if available
        if "estimated_duration_min" in extracted_content.metadata:
            return extracted_content.metadata["estimated_duration_min"]
        
        # Estimate based on content length
        content_length = len(extracted_content.content) if extracted_content.content else 0
        
        if content_length > 2000:
            return 60  # 1 hour for long content
        elif content_length > 1000:
            return 45  # 45 minutes for medium content
        elif content_length > 500:
            return 30  # 30 minutes for standard content
        else:
            return 15  # 15 minutes for short content 