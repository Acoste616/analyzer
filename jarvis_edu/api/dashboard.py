"""Web dashboard for knowledge management."""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import json
from typing import List, Optional
from ..storage.knowledge_base import KnowledgeBase
from ..storage.database import ProcessingQueue
from ..core.logger import get_logger

logger = get_logger(__name__)


def create_dashboard_app() -> FastAPI:
    """Create FastAPI dashboard application."""
    
    app = FastAPI(title="Jarvis EDU Dashboard", version="1.0.0")
    
    # Initialize services
    knowledge_base = KnowledgeBase()
    processing_queue = ProcessingQueue()
    
    # Simple HTML dashboard
    DASHBOARD_HTML = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Jarvis EDU Dashboard</title>
        <meta charset="UTF-8">
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-900 text-white p-8">
        <h1 class="text-3xl font-bold mb-8">Jarvis EDU Knowledge Base</h1>
        
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div class="bg-gray-800 p-6 rounded-lg">
                <h3 class="text-lg font-semibold mb-2">Lekcje</h3>
                <p class="text-3xl font-bold text-blue-400" id="lesson-count">0</p>
            </div>
            <div class="bg-gray-800 p-6 rounded-lg">
                <h3 class="text-lg font-semibold mb-2">Przetwarzane</h3>
                <p class="text-3xl font-bold text-yellow-400" id="processing-count">0</p>
            </div>
            <div class="bg-gray-800 p-6 rounded-lg">
                <h3 class="text-lg font-semibold mb-2">Ukończone</h3>
                <p class="text-3xl font-bold text-green-400" id="completed-count">0</p>
            </div>
        </div>
        
        <div class="bg-gray-800 p-6 rounded-lg">
            <h3 class="text-lg font-semibold mb-4">Ostatnie lekcje</h3>
            <div id="lessons-list" class="space-y-4">
                <p class="text-gray-400">Ładowanie...</p>
            </div>
        </div>
        
        <script>
            async function loadData() {
                try {
                    const response = await fetch('/api/lessons');
                    const data = await response.json();
                    
                    document.getElementById('lesson-count').textContent = data.stats.total_lessons;
                    document.getElementById('processing-count').textContent = data.stats.processing;
                    document.getElementById('completed-count').textContent = data.stats.completed;
                    
                    const lessonsList = document.getElementById('lessons-list');
                    if (data.lessons.length > 0) {
                        lessonsList.innerHTML = data.lessons.slice(0, 5).map(lesson => `
                            <div class="border border-gray-700 p-4 rounded">
                                <h4 class="font-semibold">${lesson.title}</h4>
                                <p class="text-sm text-gray-300">${lesson.summary}</p>
                                <div class="mt-2">
                                    ${lesson.tags.map(tag => `<span class="bg-blue-600 text-xs px-2 py-1 rounded mr-1">#${tag}</span>`).join('')}
                                </div>
                            </div>
                        `).join('');
                    } else {
                        lessonsList.innerHTML = '<p class="text-gray-400">Brak lekcji</p>';
                    }
                } catch (error) {
                    console.error('Error loading data:', error);
                }
            }
            
            loadData();
            setInterval(loadData, 30000); // Refresh every 30 seconds
        </script>
    </body>
    </html>
    """
    
    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        """Serve the main dashboard."""
        return HTMLResponse(content=DASHBOARD_HTML)
    
    @app.get("/api/lessons")
    async def get_lessons():
        """Get lessons with statistics."""
        try:
            lessons = knowledge_base.search_lessons(limit=10)
            kb_stats = knowledge_base.get_stats()
            queue_stats = processing_queue.get_stats()
            
            return {
                "lessons": lessons,
                "stats": {
                    "total_lessons": kb_stats["total_lessons"],
                    "processing": queue_stats["processing"],
                    "completed": queue_stats["completed"],
                    "failed": queue_stats["failed"]
                }
            }
        except Exception as e:
            logger.error(f"Error getting lessons: {e}")
            return {"lessons": [], "stats": {"total_lessons": 0, "processing": 0, "completed": 0, "failed": 0}}
    
    return app 