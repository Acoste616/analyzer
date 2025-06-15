"""Automatic folder monitoring for new files."""

import asyncio
from pathlib import Path
from typing import Set, Callable, Optional, Dict, Any
from datetime import datetime
import hashlib
import json

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from ..core.config import get_settings
from ..core.logger import get_logger
from ..models.content import ContentType
from ..storage.database import ProcessingQueue, FileStatus

logger = get_logger(__name__)


class MediaFileHandler(FileSystemEventHandler):
    """Handler for media file events."""
    
    SUPPORTED_EXTENSIONS = {
        '.mp4', '.avi', '.mov', '.mkv', '.webm',  # Video
        '.mp3', '.wav', '.flac', '.ogg', '.m4a',  # Audio
        '.jpg', '.jpeg', '.png', '.gif', '.bmp',  # Images
        '.pdf', '.docx', '.pptx', '.txt', '.md'   # Documents
    }
    
    def __init__(self, processor_callback: Callable, db_queue: ProcessingQueue):
        self.processor_callback = processor_callback
        self.db_queue = db_queue
        self.processed_files: Set[str] = set()
        self._load_processed_files()
        
    def _load_processed_files(self):
        """Load previously processed files from database."""
        processed = self.db_queue.get_processed_files()
        self.processed_files = {f.file_hash for f in processed}
        
    def _get_file_hash(self, filepath: Path) -> str:
        """Calculate file hash for deduplication."""
        try:
            hasher = hashlib.md5()
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {filepath}: {e}")
            # Fallback to simple hash based on file info
            stat = filepath.stat()
            return hashlib.md5(f"{filepath.name}_{stat.st_size}_{stat.st_mtime}".encode()).hexdigest()
        
    def _detect_content_type(self, filepath: Path) -> ContentType:
        """Detect content type from file extension."""
        ext = filepath.suffix.lower()
        
        if ext in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}:
            return ContentType.VIDEO
        elif ext in {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}:
            return ContentType.AUDIO
        elif ext in {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}:
            return ContentType.IMAGE
        elif ext in {'.pdf', '.docx', '.pptx', '.txt', '.md'}:
            return ContentType.DOCUMENT
        else:
            return ContentType.OTHER
            
    def on_created(self, event: FileSystemEvent):
        """Handle file creation event."""
        if event.is_directory:
            return
            
        self._process_file(event.src_path)
        
    def on_modified(self, event: FileSystemEvent):
        """Handle file modification event."""
        if event.is_directory:
            return
            
        # Wait a bit to ensure file write is complete
        asyncio.create_task(self._delayed_process(event.src_path, delay=2))
        
    async def _delayed_process(self, filepath: str, delay: int = 2):
        """Process file after delay to ensure write completion."""
        await asyncio.sleep(delay)
        self._process_file(filepath)
        
    def _process_file(self, filepath: str):
        """Process a single file."""
        path = Path(filepath)
        
        try:
            # Check if file exists and is readable
            if not path.exists() or not path.is_file():
                return
            
            # Check if supported
            if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                logger.debug(f"Unsupported file type: {path.name}")
                return
                
            # Check if already processed
            file_hash = self._get_file_hash(path)
            if file_hash in self.processed_files:
                logger.info(f"File already processed: {path.name}")
                return
                
            # Add to processing queue
            content_type = self._detect_content_type(path)
            
            file_entry = self.db_queue.add_file(
                filepath=str(path),
                file_hash=file_hash,
                content_type=content_type,
                metadata={
                    "original_name": path.name,
                    "size_mb": path.stat().st_size / (1024 * 1024),
                    "created": datetime.fromtimestamp(path.stat().st_ctime).isoformat(),
                    "extension": path.suffix.lower()
                }
            )
            
            # Mark as processing
            self.processed_files.add(file_hash)
            
            # Trigger async processing
            asyncio.create_task(
                self.processor_callback(file_entry)
            )
            
            logger.info(f"Added to processing queue: {path.name}")
            
        except Exception as e:
            logger.error(f"Error processing file {filepath}: {e}")


class FolderWatcher:
    """Automatic folder watcher for new media files."""
    
    def __init__(
        self, 
        watch_path: str,
        processor_callback: Callable,
        db_queue: ProcessingQueue,
        scan_interval: int = 60
    ):
        self.watch_path = Path(watch_path)
        self.processor_callback = processor_callback
        self.db_queue = db_queue
        self.scan_interval = scan_interval
        self.observer = Observer()
        self.handler = MediaFileHandler(processor_callback, db_queue)
        self.running = False
        
        # Ensure watch path exists
        self.watch_path.mkdir(parents=True, exist_ok=True)
        
    def start(self):
        """Start watching folder."""
        try:
            # Initial scan
            self._initial_scan()
            
            # Start file system observer
            self.observer.schedule(
                self.handler,
                str(self.watch_path),
                recursive=True
            )
            self.observer.start()
            self.running = True
            
            logger.info(f"Started watching: {self.watch_path}")
            
        except Exception as e:
            logger.error(f"Failed to start folder watcher: {e}")
            raise
        
    def stop(self):
        """Stop watching folder."""
        try:
            if self.observer.is_alive():
                self.observer.stop()
                self.observer.join()
            self.running = False
            logger.info("Folder watcher stopped")
        except Exception as e:
            logger.error(f"Error stopping folder watcher: {e}")
        
    def _initial_scan(self):
        """Scan folder for existing files."""
        logger.info("Performing initial folder scan...")
        
        try:
            file_count = 0
            for ext in self.handler.SUPPORTED_EXTENSIONS:
                for file_path in self.watch_path.rglob(f"*{ext}"):
                    if file_path.is_file():
                        self.handler._process_file(str(file_path))
                        file_count += 1
                        
            logger.info(f"Initial scan complete - found {file_count} files")
            
        except Exception as e:
            logger.error(f"Error during initial scan: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get watcher statistics."""
        return {
            "watch_path": str(self.watch_path),
            "running": self.running,
            "supported_extensions": list(self.handler.SUPPORTED_EXTENSIONS),
            "processed_files_count": len(self.handler.processed_files)
        } 