"""Main auto processor pipeline."""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import uuid
import json

from ..core.logger import get_logger
from ..storage.database import ProcessingQueue, FileStatus, ProcessingFile
from ..storage.knowledge_base import KnowledgeBase
from ..watcher.folder_monitor import FolderWatcher
from ..processors.lm_studio import LMStudioProcessor
from ..models.content import ExtractedContent, ContentType

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
            
            # Create dummy lesson
            lesson_id = await self.create_dummy_lesson(file_entry)
            
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
    
    async def create_dummy_lesson(self, file_entry: ProcessingFile) -> str:
        """Create a dummy lesson for testing."""
        lesson_id = f"lesson_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        source_name = Path(file_entry.filepath).stem
        
        try:
            # Add to knowledge base
            self.knowledge_base.add_lesson(
                lesson_id=lesson_id,
                title=f"Lekcja: {source_name}",
                summary=f"Automatycznie wygenerowana lekcja z pliku {source_name}",
                content_path="",
                tags=["auto-generated", "test"],
                categories=["Test"],
                difficulty="intermediate",
                duration=30,
                topics_count=1,
                metadata={
                    "source_file": file_entry.filepath,
                    "content_type": file_entry.content_type.value
                }
            )
            
            logger.info(f"Created dummy lesson: {lesson_id}")
            return lesson_id
            
        except Exception as e:
            logger.error(f"Failed to create dummy lesson: {e}")
            raise 