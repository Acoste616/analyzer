"""Knowledge base management with categories and tags."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, Column, String, DateTime, JSON, Integer, Table, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

from ..core.config import get_settings
from ..core.logger import get_logger

logger = get_logger(__name__)

Base = declarative_base()

# Association tables
lesson_tags = Table('lesson_tags', Base.metadata,
    Column('lesson_id', String, ForeignKey('lessons.id')),
    Column('tag_id', Integer, ForeignKey('tags.id'))
)

lesson_categories = Table('lesson_categories', Base.metadata,
    Column('lesson_id', String, ForeignKey('lessons.id')),
    Column('category_id', Integer, ForeignKey('categories.id'))
)


class Lesson(Base):
    """Lesson model with tags and categories."""
    __tablename__ = 'lessons'
    
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    summary = Column(String)
    content_path = Column(String)
    difficulty = Column(String)
    duration = Column(Integer)
    topics_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    metadata = Column(JSON)
    
    # Relationships
    tags = relationship("Tag", secondary=lesson_tags, back_populates="lessons")
    categories = relationship("Category", secondary=lesson_categories, back_populates="lessons")


class Tag(Base):
    """Tag model for lessons."""
    __tablename__ = 'tags'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    lessons = relationship("Lesson", secondary=lesson_tags, back_populates="tags")


class Category(Base):
    """Category model for lessons."""
    __tablename__ = 'categories'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(String)
    parent_id = Column(Integer, ForeignKey('categories.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    lessons = relationship("Lesson", secondary=lesson_categories, back_populates="categories")
    children = relationship("Category")


class KnowledgeBase:
    """Main knowledge base manager."""
    
    def __init__(self, db_url: Optional[str] = None):
        if not db_url:
            settings = get_settings()
            db_url = f"sqlite:///{settings.database.sqlite_path}"
            
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
    def add_lesson(
        self,
        lesson_id: str,
        title: str,
        summary: str,
        content_path: str,
        tags: List[str],
        categories: List[str],
        difficulty: str = "intermediate",
        duration: int = 30,
        topics_count: int = 1,
        metadata: Optional[Dict] = None
    ) -> Lesson:
        """Add new lesson to knowledge base."""
        session = self.Session()
        
        try:
            lesson = Lesson(
                id=lesson_id,
                title=title,
                summary=summary,
                content_path=content_path,
                difficulty=difficulty,
                duration=duration,
                topics_count=topics_count,
                metadata=metadata or {}
            )
            
            # Add tags
            for tag_name in tags:
                tag = session.query(Tag).filter_by(name=tag_name).first()
                if not tag:
                    tag = Tag(name=tag_name)
                    session.add(tag)
                lesson.tags.append(tag)
            
            # Add categories
            for cat_name in categories:
                category = session.query(Category).filter_by(name=cat_name).first()
                if not category:
                    category = Category(name=cat_name)
                    session.add(category)
                lesson.categories.append(category)
            
            session.add(lesson)
            session.commit()
            
            logger.info(f"Added lesson to knowledge base: {title}")
            return lesson
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to add lesson: {e}")
            raise
        finally:
            session.close()
    
    def search_lessons(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        difficulty: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search lessons with filters."""
        session = self.Session()
        
        try:
            q = session.query(Lesson)
            
            if query:
                q = q.filter(
                    (Lesson.title.contains(query)) |
                    (Lesson.summary.contains(query))
                )
            
            if tags:
                q = q.join(Lesson.tags).filter(Tag.name.in_(tags))
            
            if categories:
                q = q.join(Lesson.categories).filter(Category.name.in_(categories))
            
            if difficulty:
                q = q.filter(Lesson.difficulty == difficulty)
            
            lessons = q.order_by(Lesson.created_at.desc()).limit(limit).all()
            
            return [self._lesson_to_dict(lesson) for lesson in lessons]
            
        finally:
            session.close()
    
    def get_all_tags(self) -> List[Dict[str, Any]]:
        """Get all tags with usage count."""
        session = self.Session()
        
        try:
            tags = session.query(Tag).all()
            return [
                {
                    "name": tag.name,
                    "count": len(tag.lessons),
                    "created_at": tag.created_at.isoformat()
                }
                for tag in tags
            ]
        finally:
            session.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        session = self.Session()
        
        try:
            return {
                "total_lessons": session.query(Lesson).count(),
                "total_tags": session.query(Tag).count(),
                "total_categories": session.query(Category).count(),
                "difficulties": {
                    "beginner": session.query(Lesson).filter_by(difficulty="beginner").count(),
                    "intermediate": session.query(Lesson).filter_by(difficulty="intermediate").count(),
                    "advanced": session.query(Lesson).filter_by(difficulty="advanced").count()
                }
            }
        finally:
            session.close()
    
    def _lesson_to_dict(self, lesson: Lesson) -> Dict[str, Any]:
        """Convert lesson object to dictionary."""
        return {
            "id": lesson.id,
            "title": lesson.title,
            "summary": lesson.summary,
            "content_path": lesson.content_path,
            "difficulty": lesson.difficulty,
            "duration": lesson.duration,
            "topics_count": lesson.topics_count,
            "tags": [tag.name for tag in lesson.tags],
            "categories": [cat.name for cat in lesson.categories],
            "created_at": lesson.created_at.isoformat(),
            "updated_at": lesson.updated_at.isoformat() if lesson.updated_at else None,
            "metadata": lesson.metadata
        } 