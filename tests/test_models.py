"""Tests for data models."""

import pytest
from datetime import datetime

from jarvis_edu.models.content import (
    ExtractedContent, 
    VideoMetadata, 
    TextChunk,
    ContentType,
    Language
)
from jarvis_edu.models.base import TimestampMixin, IDMixin


class TestBaseModels:
    """Test base model functionality."""
    
    def test_timestamp_mixin(self):
        """Test timestamp mixin."""
        
        class TestModel(TimestampMixin):
            pass
            
        model = TestModel()
        
        # Check that created_at is set
        assert model.created_at is not None
        assert isinstance(model.created_at, datetime)
        assert model.updated_at is None
        
        # Test update timestamp
        model.update_timestamp()
        assert model.updated_at is not None
        
    def test_id_mixin(self):
        """Test ID mixin."""
        
        class TestModel(IDMixin):
            pass
            
        model = TestModel()
        
        # Check that ID is generated
        assert model.id is not None
        assert isinstance(model.id, str)
        assert len(model.id) > 0


class TestContentModels:
    """Test content models."""
    
    def test_video_metadata(self):
        """Test video metadata model."""
        metadata = VideoMetadata(
            duration=120.5,
            fps=30.0,
            resolution=(1920, 1080),
            has_audio=True
        )
        
        assert metadata.duration == 120.5
        assert metadata.fps == 30.0
        assert metadata.resolution == (1920, 1080)
        assert metadata.has_audio is True
        
    def test_video_metadata_validation(self):
        """Test video metadata validation."""
        
        # Test negative duration
        with pytest.raises(ValueError):
            VideoMetadata(
                duration=-10.0,
                fps=30.0,
                resolution=(1920, 1080),
                has_audio=True
            )
            
        # Test zero FPS
        with pytest.raises(ValueError):
            VideoMetadata(
                duration=120.0,
                fps=0.0,
                resolution=(1920, 1080),
                has_audio=True
            )
    
    def test_text_chunk(self):
        """Test text chunk model."""
        chunk = TextChunk(
            text="This is test content",
            start_time=0.0,
            end_time=5.0,
            confidence=0.95
        )
        
        assert chunk.text == "This is test content"
        assert chunk.start_time == 0.0
        assert chunk.end_time == 5.0
        assert chunk.confidence == 0.95
    
    def test_text_chunk_validation(self):
        """Test text chunk validation."""
        
        # Test invalid confidence
        with pytest.raises(ValueError):
            TextChunk(
                text="Test",
                confidence=1.5  # > 1.0
            )
            
        with pytest.raises(ValueError):
            TextChunk(
                text="Test", 
                confidence=-0.1  # < 0.0
            )
    
    def test_extracted_content(self):
        """Test extracted content model."""
        content = ExtractedContent(
            content_type=ContentType.VIDEO,
            language=Language.ENGLISH,
            raw_text="This is extracted content",
            source_path="/path/to/video.mp4"
        )
        
        assert content.content_type == ContentType.VIDEO
        assert content.language == Language.ENGLISH
        assert content.raw_text == "This is extracted content"
        assert content.word_count == 4  # "This is extracted content"
        assert content.character_count > 0
        
    def test_extracted_content_methods(self):
        """Test extracted content methods."""
        chunks = [
            TextChunk(text="First chunk", start_time=0.0, end_time=5.0),
            TextChunk(text="Second chunk", start_time=5.0, end_time=10.0),
            TextChunk(text="Third chunk", start_time=10.0, end_time=15.0)
        ]
        
        content = ExtractedContent(
            content_type=ContentType.VIDEO,
            raw_text="",
            text_chunks=chunks
        )
        
        # Test getting text by time range
        timerange_text = content.get_text_by_timerange(0.0, 10.0)
        assert "First chunk" in timerange_text
        assert "Second chunk" in timerange_text
        assert "Third chunk" not in timerange_text
        
        # Test adding text chunk
        new_chunk = TextChunk(text="Fourth chunk")
        original_count = len(content.text_chunks)
        content.add_text_chunk(new_chunk)
        
        assert len(content.text_chunks) == original_count + 1
        assert content.text_chunks[-1].text == "Fourth chunk" 