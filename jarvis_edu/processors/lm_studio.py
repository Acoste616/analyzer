"""LM Studio integration for local LLM processing."""

import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

from ..core.config import get_settings
from ..core.logger import get_logger
from ..models.content import ExtractedContent, Language

logger = get_logger(__name__)


class LMStudioProcessor:
    """LM Studio API client for local LLM processing."""
    
    def __init__(self, base_url: str = "http://localhost:1234/v1"):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        
    async def is_available(self) -> bool:
        """Check if LM Studio is running and available."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/models") as resp:
                    return resp.status == 200
        except:
            return False
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection and get available models."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/models") as resp:
                    if resp.status == 200:
                        models = await resp.json()
                        return {
                            "status": "connected",
                            "models": models.get("data", []),
                            "endpoint": self.base_url
                        }
                    else:
                        return {
                            "status": "error",
                            "message": f"HTTP {resp.status}",
                            "endpoint": self.base_url
                        }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "endpoint": self.base_url
            }
    
    async def generate_lesson(
        self, 
        content: ExtractedContent,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate lesson structure using LM Studio."""
        
        if not system_prompt:
            system_prompt = """Jesteś ekspertem edukacyjnym. Na podstawie dostarczonej treści:
1. Stwórz strukturalną lekcję po polsku
2. Wyodrębnij kluczowe pojęcia i tematy
3. Dodaj przykłady i ćwiczenia
4. Wygeneruj quiz sprawdzający wiedzę
5. Określ poziom trudności i wymagania wstępne
6. Dodaj tagi i kategorie dla łatwego wyszukiwania

Odpowiedź w formacie JSON z polami:
- title: tytuł lekcji
- summary: krótkie podsumowanie
- topics: lista tematów z opisami
- difficulty: poziom trudności (beginner/intermediate/advanced)
- prerequisites: wymagania wstępne
- tags: lista tagów
- categories: lista kategorii
- quiz: lista pytań quizowych
- exercises: lista ćwiczeń praktycznych
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Przetworz następującą treść:\n\n{content.raw_text[:5000]}"}
        ]
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json={
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 4000,
                        "stream": False
                    }
                ) as resp:
                    result = await resp.json()
                    
                    # NAPRAWKA: Sprawdzenie struktury odpowiedzi
                    if 'choices' not in result:
                        logger.error(f"LM Studio response missing 'choices': {result}")
                        return self._fallback_lesson_structure(content)
                    
                    if not result['choices'] or len(result['choices']) == 0:
                        logger.error(f"LM Studio response has empty 'choices': {result}")
                        return self._fallback_lesson_structure(content)
                    
                    choice = result['choices'][0]
                    if 'message' not in choice or 'content' not in choice['message']:
                        logger.error(f"LM Studio response missing message/content: {choice}")
                        return self._fallback_lesson_structure(content)
                    
                    # Parse JSON from response
                    response_text = choice["message"]["content"]
                    
                    # Extract JSON if wrapped in markdown
                    if "```json" in response_text:
                        json_start = response_text.find("```json") + 7
                        json_end = response_text.find("```", json_start)
                        response_text = response_text[json_start:json_end]
                    
                    return json.loads(response_text)
                    
        except Exception as e:
            logger.error(f"LM Studio generation failed: {e}")
            return self._fallback_lesson_structure(content)
    
    async def generate_lesson_from_prompt(self, prompt: str) -> Dict[str, Any]:
        """Generate lesson from custom prompt."""
        messages = [
            {"role": "system", "content": "You are an educational content expert. Respond in JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json={
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 2000,
                        "stream": False  # NAPRAWKA: Usunięto response_format - może powodować problemy
                    }
                ) as resp:
                    result = await resp.json()
                    
                    # NAPRAWKA: Sprawdzenie struktury odpowiedzi
                    if 'choices' not in result:
                        logger.error(f"LM Studio response missing 'choices': {result}")
                        return {}
                    
                    if not result['choices'] or len(result['choices']) == 0:
                        logger.error(f"LM Studio response has empty 'choices': {result}")
                        return {}
                    
                    choice = result['choices'][0]
                    if 'message' not in choice or 'content' not in choice['message']:
                        logger.error(f"LM Studio response missing message/content: {choice}")
                        return {}
                    
                    response_text = choice["message"]["content"]
                    
                    # Parse JSON
                    try:
                        return json.loads(response_text)
                    except json.JSONDecodeError as json_err:
                        logger.warning(f"JSON parsing failed: {json_err}. Response: {response_text}")
                        # Fallback structure
                        return {
                            "title": "Lekcja bez tytułu",
                            "summary": "Podsumowanie niedostępne",
                            "content": response_text,
                            "tags": ["ai-processed"],
                            "categories": ["Ogólne"]
                        }
                        
        except Exception as e:
            logger.error(f"LM Studio prompt generation failed: {e}")
            return {}
    
    def _fallback_lesson_structure(self, content: ExtractedContent) -> Dict[str, Any]:
        """Fallback lesson structure if LM Studio fails."""
        return {
            "title": "Lekcja bez tytułu",
            "summary": content.raw_text[:200] + "...",
            "topics": [{"title": "Główny temat", "description": "Do uzupełnienia"}],
            "difficulty": "intermediate",
            "prerequisites": [],
            "tags": ["do-sprawdzenia", "auto-generated"],
            "categories": ["Nieskategoryzowane"],
            "quiz": [],
            "exercises": []
        } 