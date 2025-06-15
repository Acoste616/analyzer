"""Database models and processing queue management."""

from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json
from dataclasses import dataclass

from sqlalchemy import create_engine, Column, String, DateTime, JSON, Integer, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from ..core.config import get_settings
from ..core.logger import get_logger
from ..models.content import ContentType

logger = get_logger(__name__)

Base = declarative_base()


class FileStatus(Enum):
    """File processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"


@dataclass
class ProcessingFile:
    """File in processing queue."""
    id: str
    filepath: str
    file_hash: str
    content_type: ContentType
    status: FileStatus
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0


class FileEntry(Base):
    """Database model for file entries."""
    __tablename__ = 'file_entries'
    
    id = Column(String, primary_key=True)
    filepath = Column(String, nullable=False)
    file_hash = Column(String, unique=True, nullable=False)
    content_type = Column(String, nullable=False)
    status = Column(SQLEnum(FileStatus), default=FileStatus.PENDING)
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    error_message = Column(String)
    retry_count = Column(Integer, default=0)


class ProcessingQueue:
    """Processing queue manager with database persistence."""
    
    def __init__(self, db_url: Optional[str] = None):
        """Initialize processing queue with database."""
        if not db_url:
            settings = get_settings()
            db_url = f"sqlite:///{settings.database.sqlite_path}"
            
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        logger.info(f"Processing queue initialized with database: {db_url}")
    
    def add_file(
        self,
        filepath: str,
        file_hash: str,
        content_type: ContentType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessingFile:
        """Add file to processing queue."""
        session = self.Session()
        
        try:
            # Check if file already exists
            existing = session.query(FileEntry).filter_by(file_hash=file_hash).first()
            if existing:
                logger.info(f"File already in queue: {Path(filepath).name}")
                return self._to_processing_file(existing)
            
            # Create new entry
            file_id = f"file_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file_hash[:8]}"
            
            entry = FileEntry(
                id=file_id,
                filepath=filepath,
                file_hash=file_hash,
                content_type=content_type.value,
                metadata=metadata or {},
                status=FileStatus.PENDING
            )
            
            session.add(entry)
            session.commit()
            
            result = self._to_processing_file(entry)
            logger.info(f"Added to processing queue: {Path(filepath).name}")
            return result
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to add file to queue: {e}")
            raise
        finally:
            session.close()
    
    def get_pending_files(self, limit: int = 10) -> List[ProcessingFile]:
        """Get pending files from queue."""
        session = self.Session()
        
        try:
            entries = (session.query(FileEntry)
                      .filter(FileEntry.status.in_([FileStatus.PENDING, FileStatus.RETRY]))
                      .order_by(FileEntry.created_at)
                      .limit(limit)
                      .all())
            
            return [self._to_processing_file(entry) for entry in entries]
            
        finally:
            session.close()
    
    def update_status(
        self,
        file_id: str,
        status: FileStatus,
        error_message: Optional[str] = None,
        metadata_update: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update file processing status."""
        session = self.Session()
        
        try:
            entry = session.query(FileEntry).filter_by(id=file_id).first()
            if not entry:
                logger.warning(f"File not found in queue: {file_id}")
                return
            
            entry.status = status
            entry.updated_at = datetime.utcnow()
            
            if error_message:
                entry.error_message = error_message
                entry.retry_count += 1
            
            if metadata_update:
                entry.metadata.update(metadata_update)
            
            session.commit()
            
            logger.info(f"Updated file status: {file_id} -> {status.value}")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update file status: {e}")
            raise
        finally:
            session.close()
    
    def get_processing_files(self) -> List[ProcessingFile]:
        """Get currently processing files."""
        session = self.Session()
        
        try:
            entries = (session.query(FileEntry)
                      .filter_by(status=FileStatus.PROCESSING)
                      .all())
            
            return [self._to_processing_file(entry) for entry in entries]
            
        finally:
            session.close()
    
    def get_processed_files(self) -> List[ProcessingFile]:
        """Get successfully processed files."""
        session = self.Session()
        
        try:
            entries = (session.query(FileEntry)
                      .filter_by(status=FileStatus.COMPLETED)
                      .order_by(FileEntry.updated_at.desc())
                      .all())
            
            return [self._to_processing_file(entry) for entry in entries]
            
        finally:
            session.close()
    
    def get_failed_files(self) -> List[ProcessingFile]:
        """Get failed files."""
        session = self.Session()
        
        try:
            entries = (session.query(FileEntry)
                      .filter_by(status=FileStatus.FAILED)
                      .order_by(FileEntry.updated_at.desc())
                      .all())
            
            return [self._to_processing_file(entry) for entry in entries]
            
        finally:
            session.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing queue statistics."""
        session = self.Session()
        
        try:
            total = session.query(FileEntry).count()
            pending = session.query(FileEntry).filter_by(status=FileStatus.PENDING).count()
            processing = session.query(FileEntry).filter_by(status=FileStatus.PROCESSING).count()
            completed = session.query(FileEntry).filter_by(status=FileStatus.COMPLETED).count()
            failed = session.query(FileEntry).filter_by(status=FileStatus.FAILED).count()
            
            return {
                "total": total,
                "pending": pending,
                "processing": processing,
                "completed": completed,
                "failed": failed,
                "success_rate": (completed / total * 100) if total > 0 else 0
            }
            
        finally:
            session.close()
    
    def cleanup_old_entries(self, days_old: int = 30) -> int:
        """Clean up old completed entries."""
        session = self.Session()
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days_old)
            
            count = (session.query(FileEntry)
                    .filter(FileEntry.status == FileStatus.COMPLETED)
                    .filter(FileEntry.updated_at < cutoff_date)
                    .delete())
            
            session.commit()
            
            if count > 0:
                logger.info(f"Cleaned up {count} old file entries")
            
            return count
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to cleanup old entries: {e}")
            return 0
        finally:
            session.close()
    
    def _to_processing_file(self, entry: FileEntry) -> ProcessingFile:
        """Convert database entry to ProcessingFile."""
        return ProcessingFile(
            id=entry.id,
            filepath=entry.filepath,
            file_hash=entry.file_hash,
            content_type=ContentType(entry.content_type),
            status=entry.status,
            metadata=entry.metadata,
            created_at=entry.created_at,
            updated_at=entry.updated_at,
            error_message=entry.error_message,
            retry_count=entry.retry_count
        ) 