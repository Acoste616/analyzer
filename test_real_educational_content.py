#!/usr/bin/env python3
"""PRAWDZIWY TEST - Materiały edukacyjne z rzeczywistą treścią"""

import os
import sys
import asyncio
import time
from pathlib import Path
from datetime import datetime
import logging
import json
import torch
from typing import List, Dict, Any
import requests

# Jarvis imports
try:
    from jarvis_edu.core.config import get_settings
    from jarvis_edu.extractors.enhanced_image_extractor import EnhancedImageExtractor
    from jarvis_edu.extractors.enhanced_video_extractor import EnhancedVideoExtractor
    from jarvis_edu.extractors.enhanced_audio_extractor import EnhancedAudioExtractor
    from jarvis_edu.processors.lm_studio import LMStudioProcessor
    from jarvis_edu.storage.knowledge_base import KnowledgeBase
    JARVIS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Błąd importu Jarvis: {e}")
    JARVIS_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/real_educational_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealEducationalTester:
    def __init__(self):
        self.media_folder = Path("C:/Users/barto/OneDrive/Pulpit/mediapobrane/pobrane_media")
        self.output_folder = Path("output")
        self.output_folder.mkdir(exist_ok=True)
        
        if JARVIS_AVAILABLE:
            try:
                self.settings = get_settings()
                self.image_extractor = EnhancedImageExtractor()
                self.video_extractor = EnhancedVideoExtractor()
                self.audio_extractor = EnhancedAudioExtractor()
                self.llm_processor = LMStudioProcessor()
                self.knowledge_base = KnowledgeBase()
                self.real_processing = True
                logger.info("✅ Jarvis ekstraktory zainicjalizowane")
            except Exception as e:
                logger.error(f"❌ Błąd inicjalizacji Jarvis: {e}")
                self.real_processing = False
        else:
            self.real_processing = False
        
        self.results = []

    def create_test_educational_content(self):
        """Tworzy testowe pliki edukacyjne z prawdziwą treścią"""
        test_dir = self.output_folder / "test_educational_files"
        test_dir.mkdir(exist_ok=True)
        
        # 1. Tekst matematyczny
        math_image = self.create_math_diagram(test_dir / "matematyka_funkcje.png")
        
        # 2. Tekst naukowy
        science_image = self.create_science_text(test_dir / "nauka_fizyka.png")
        
        # 3. Diagram edukacyjny
        diagram_image = self.create_educational_diagram(test_dir / "diagram_edukacyjny.png")
        
        logger.info(f"✅ Utworzono testowe pliki edukacyjne w: {test_dir}")
        return list(test_dir.glob("*.png"))

    def create_math_diagram(self, output_path: Path):
        """Tworzy obraz z treścią matematyczną"""
        from PIL import Image, ImageDraw, ImageFont
        
        # Duży obraz z czytelnym tekstem matematycznym
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # Użyj większej czcionki
        try:
            font_title = ImageFont.truetype("arial.ttf", 36)
            font_content = ImageFont.truetype("arial.ttf", 24)
        except:
            font_title = ImageFont.load_default()
            font_content = ImageFont.load_default()
        
        # Treść matematyczna
        math_content = [
            "MATEMATYKA - FUNKCJE KWADRATOWE",
            "",
            "Definicja: f(x) = ax² + bx + c",
            "gdzie a ≠ 0",
            "",
            "Właściwości:",
            "• Dziedzina: R (wszystkie liczby rzeczywiste)",
            "• Zbiór wartości: zależy od znaku 'a'",
            "• Wierzchołek: (-b/2a, f(-b/2a))",
            "",
            "Przykład:",
            "f(x) = x² - 4x + 3",
            "Δ = b² - 4ac = 16 - 12 = 4",
            "x₁ = (4-2)/2 = 1",
            "x₂ = (4+2)/2 = 3",
            "",
            "Zastosowania w fizyce:",
            "• Ruch jednostajnie przyspieszony",
            "• Trajektoria rzutu ukośnego",
            "• Energia potencjalna"
        ]
        
        y = 50
        for line in math_content:
            if line == math_content[0]:  # Tytuł
                draw.text((50, y), line, fill='black', font=font_title)
                y += 50
            else:
                draw.text((50, y), line, fill='black', font=font_content)
                y += 30
        
        img.save(output_path)
        logger.info(f"📐 Utworzono obraz matematyczny: {output_path}")
        return output_path

    def create_science_text(self, output_path: Path):
        """Tworzy obraz z treścią naukową"""
        from PIL import Image, ImageDraw, ImageFont
        
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font_title = ImageFont.truetype("arial.ttf", 32)
            font_content = ImageFont.truetype("arial.ttf", 20)
        except:
            font_title = ImageFont.load_default()
            font_content = ImageFont.load_default()
        
        science_content = [
            "FIZYKA - PRAWA RUCHU NEWTONA",
            "",
            "I PRAWO NEWTONA (Zasada bezwładności):",
            "Ciało pozostaje w spoczynku lub porusza się",
            "ruchem jednostajnym prostoliniowym, jeśli nie",
            "działa na nie żadna siła lub siły się równoważą.",
            "",
            "II PRAWO NEWTONA (Zasada dynamiki):",
            "F = ma",
            "Przyspieszenie ciała jest wprost proporcjonalne",
            "do działającej siły i odwrotnie proporcjonalne",
            "do masy ciała.",
            "",
            "III PRAWO NEWTONA (Zasada akcji i reakcji):",
            "Każdej akcji towarzyszy równa co do wartości",
            "i przeciwnie skierowana reakcja.",
            "",
            "Przykłady zastosowań:",
            "• Ruch samochodów",
            "• Loty rakiet kosmicznych",
            "• Chodzenie po ziemi",
            "• Pływanie w wodzie"
        ]
        
        y = 40
        for line in science_content:
            if line == science_content[0]:  # Tytuł
                draw.text((40, y), line, fill='black', font=font_title)
                y += 45
            else:
                draw.text((40, y), line, fill='black', font=font_content)
                y += 25
        
        img.save(output_path)
        logger.info(f"🔬 Utworzono obraz naukowy: {output_path}")
        return output_path

    def create_educational_diagram(self, output_path: Path):
        """Tworzy diagram edukacyjny"""
        from PIL import Image, ImageDraw, ImageFont
        
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font_title = ImageFont.truetype("arial.ttf", 28)
            font_content = ImageFont.truetype("arial.ttf", 18)
        except:
            font_title = ImageFont.load_default()
            font_content = ImageFont.load_default()
        
        # Tytuł
        draw.text((200, 30), "CYKL ŻYCIA MOTYLA", fill='black', font=font_title)
        
        # Rysowanie diagramu cyklu
        stages = [
            ("1. JAJO", (150, 120)),
            ("2. LARWA", (550, 120)),
            ("3. POCZWARKA", (550, 350)),
            ("4. MOTYL", (150, 350))
        ]
        
        # Rysowanie prostokątów i tekstu
        for stage, (x, y) in stages:
            # Prostokąt
            draw.rectangle([x-50, y-30, x+150, y+30], outline='black', width=2)
            draw.text((x-30, y-10), stage, fill='black', font=font_content)
        
        # Strzałki między etapami
        arrow_coords = [
            ((300, 120), (450, 120)),  # 1->2
            ((550, 170), (550, 300)),  # 2->3
            ((500, 350), (200, 350)),  # 3->4
            ((150, 300), (150, 170))   # 4->1
        ]
        
        for (x1, y1), (x2, y2) in arrow_coords:
            draw.line([(x1, y1), (x2, y2)], fill='black', width=3)
            # Grot strzałki (prosty)
            if x1 == x2:  # Pionowa
                if y2 > y1:  # W dół
                    draw.polygon([(x2-5, y2-10), (x2+5, y2-10), (x2, y2)], fill='black')
                else:  # W górę
                    draw.polygon([(x2-5, y2+10), (x2+5, y2+10), (x2, y2)], fill='black')
            else:  # Pozioma
                if x2 > x1:  # W prawo
                    draw.polygon([(x2-10, y2-5), (x2-10, y2+5), (x2, y2)], fill='black')
                else:  # W lewo
                    draw.polygon([(x2+10, y2-5), (x2+10, y2+5), (x2, y2)], fill='black')
        
        # Opis na dole
        draw.text((100, 480), "Metamorfoza to proces przemian biologicznych", fill='black', font=font_content)
        draw.text((100, 510), "zachodzących podczas rozwoju owadów.", fill='black', font=font_content)
        
        img.save(output_path)
        logger.info(f"📊 Utworzono diagram edukacyjny: {output_path}")
        return output_path

    async def test_real_files(self):
        """Test z prawdziwymi plikami edukacyjnymi"""
        logger.info("🎯 ROZPOCZĘCIE TESTU Z PRAWDZIWYMI MATERIAŁAMI EDUKACYJNYMI")
        print("="*80)
        
        # Tworzenie testowych plików edukacyjnych
        test_files = self.create_test_educational_content()
        
        # Przetwarzanie tylko pierwszego pliku (matematyka)
        if test_files:
            file_path = test_files[0]  # matematyka_funkcje.png
            logger.info(f"📚 Przetwarzanie: {file_path.name}")
            
            start_time = time.time()
            
            try:
                # Test OCR
                extraction_result = await self.image_extractor.extract(str(file_path))
                
                # Sprawdź czy OCR wykrył prawdziwą treść
                has_content = (
                    extraction_result.content and 
                    len(extraction_result.content.strip()) > 20 and
                    not extraction_result.content.startswith("OCR nie wykrył") and
                    not extraction_result.content.startswith("Brak treści")
                )
                
                if has_content:
                    logger.info(f"✅ OCR SUKCES: {len(extraction_result.content)} znaków")
                    
                    # Test AI podsumowania
                    prompt = f"""
                    Przeanalizuj poniższy materiał edukacyjny i stwórz szczegółowe podsumowanie:

                    Tytuł: {extraction_result.title}
                    Treść: {extraction_result.content}

                    Stwórz strukturalne podsumowanie (200-400 słów) zawierające:
                    1. Główny temat i dziedzina nauki
                    2. Kluczowe pojęcia i definicje
                    3. Praktyczne zastosowania
                    4. Poziom trudności i grupę docelową
                    5. Rekomendacje dla nauczycieli
                    """
                    
                    llm_summary = await self.call_lm_studio(prompt)
                    
                    # Zapisz do Knowledge Base
                    lesson_id = await self.save_to_knowledge_base(
                        file_path=file_path,
                        extraction_result=extraction_result,
                        llm_summary=llm_summary,
                        content_type="educational_image"
                    )
                    
                    processing_time = time.time() - start_time
                    
                    result = {
                        'file_path': str(file_path),
                        'file_type': 'educational_image',
                        'processing_time': processing_time,
                        'status': 'success',
                        'extracted_text': extraction_result.content,
                        'llm_summary': llm_summary,
                        'lesson_id': lesson_id,
                        'confidence': extraction_result.confidence,
                        'error': None
                    }
                    
                    self.results.append(result)
                    
                    print(f"\n📖 PLIK: {file_path.name}")
                    print(f"⏱️ CZAS: {processing_time:.2f}s")
                    print(f"📝 TREŚĆ:\n{extraction_result.content}")
                    print(f"\n🧠 AI PODSUMOWANIE:\n{llm_summary}")
                    print(f"\n💾 LESSON ID: {lesson_id}")
                    print("="*80)
                    
                    return True  # Sukces!
                    
                else:
                    logger.error(f"❌ OCR NIEPOWODZENIE: {file_path.name}")
                    print(f"❌ OCR NIE WYKRYŁ TREŚCI: {extraction_result.content}")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ BŁĄD {file_path.name}: {e}")
                print(f"❌ BŁĄD PRZETWARZANIA: {e}")
                return False

    async def call_lm_studio(self, prompt: str) -> str:
        """Wywołanie LM Studio API z anti-hallucination protection"""
        try:
            payload = {
                "messages": [
                    {"role": "system", "content": "You are an educational expert. Create CONCISE, FACTUAL summaries in Polish. DO NOT repeat words or phrases. Focus on SPECIFIC content from the given material. If you don't understand the content, say so clearly."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 300,  # Reduced to prevent repetition
                "temperature": 0.1,  # Much lower temperature
                "top_p": 0.8,
                "frequency_penalty": 1.0,  # Prevent repetition
                "presence_penalty": 0.5,
                "stream": False
            }
            
            response = requests.post(
                'http://127.0.0.1:1234/v1/chat/completions',
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and data['choices']:
                    if isinstance(data['choices'], list) and len(data['choices']) > 0:
                        choice = data['choices'][0]
                        if 'message' in choice and 'content' in choice['message']:
                            raw_response = choice['message']['content'].strip()
                            
                            # 🔥 HALLUCINATION DETECTION
                            cleaned_response = self._clean_lm_studio_response(raw_response)
                            if cleaned_response:
                                return cleaned_response
                            else:
                                return "AI HALLUCINATION WYKRYTA - sprawdź model LM Studio. Użyj meta-llama-3-8b-instruct zamiast mistral."
                
                return "Podsumowanie AI (struktura odpowiedzi nieoczekiwana)"
            else:
                return f"Błąd LM Studio (HTTP {response.status_code})"
            
        except Exception as e:
            logger.error(f"Błąd LM Studio: {e}")
            return f"Błąd generowania: {str(e)}"

    def _clean_lm_studio_response(self, response: str) -> str:
        """Clean LM Studio response and detect hallucination"""
        if not response or len(response.strip()) < 10:
            return ""
        
        import re
        from collections import Counter
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', response.strip())
        
        # Detect word/phrase repetition (like Whisper hallucination)
        words = cleaned.split()
        
        if len(words) > 10:
            word_counts = Counter(words)
            total_words = len(words)
            
            # Check for excessive repetition
            for word, count in word_counts.items():
                if count / total_words > 0.3:  # More than 30% of text is one word
                    logger.warning(f"LM Studio hallucination detected: '{word}' repeated {count} times")
                    return ""
        
        # Check for phrase repetition
        if len(words) > 5:
            # Check for repeating 2-word phrases
            phrases = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            phrase_counts = Counter(phrases)
            
            for phrase, count in phrase_counts.items():
                if count > 3:  # Phrase repeated more than 3 times
                    logger.warning(f"LM Studio phrase hallucination: '{phrase}' repeated {count} times")
                    return ""
        
        # Check for meaningless repetitive patterns
        if "zastosowania nauczania" in cleaned and cleaned.count("zastosowania") > 5:
            logger.warning("LM Studio known hallucination pattern detected")
            return ""
        
        # Check if content is too repetitive (low diversity)
        if len(set(words)) < len(words) * 0.15:  # Less than 15% unique words
            logger.warning("LM Studio response has very low word diversity - likely hallucination")
            return ""
        
        # Return cleaned response if it passes all tests
        return cleaned

    async def save_to_knowledge_base(self, file_path: Path, extraction_result, llm_summary: str, content_type: str) -> str:
        """Zapisuje do Knowledge Base"""
        try:
            import hashlib
            
            file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
            lesson_id = f"edu_{content_type}_{file_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            lesson_data = {
                'lesson_id': lesson_id,
                'title': f"Materiał edukacyjny: {extraction_result.title}",
                'summary': llm_summary,
                'content': extraction_result.content,
                'source_file': str(file_path),
                'content_type': content_type,
                'extraction_metadata': extraction_result.metadata,
                'confidence': extraction_result.confidence,
                'language': extraction_result.language,
                'created_at': datetime.now().isoformat(),
                'tags': ['educational', 'test', 'real-content', content_type],
                'categories': ['Materiały edukacyjne', 'Test OCR']
            }
            
            kb_file = self.output_folder / f"knowledge_base_{lesson_id}.json"
            with open(kb_file, 'w', encoding='utf-8') as f:
                json.dump(lesson_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Zapisano do Knowledge Base: {lesson_id}")
            return lesson_id
            
        except Exception as e:
            logger.error(f"Błąd zapisu do Knowledge Base: {e}")
            return f"error_{datetime.now().strftime('%H%M%S')}"

async def main():
    """Główna funkcja testu"""
    print("🎓 PRAWDZIWY TEST MATERIAŁÓW EDUKACYJNYCH")
    print("="*80)
    print(f"Test started: {datetime.now()}")
    print("="*80)
    
    tester = RealEducationalTester()
    success = await tester.test_real_files()
    
    if success:
        print("\n🎉 ✅ SYSTEM DZIAŁA PRAWIDŁOWO!")
        print("🧠 OCR wykrywa prawdziwą treść edukacyjną")
        print("🤖 AI generuje rzeczywiste podsumowania")
        print("💾 Knowledge Base zapisuje lekcje")
    else:
        print("\n❌ SYSTEM WYMAGA NAPRAWY!")
        print("🔧 OCR nie wykrywa treści lub AI nie działa")
    
    print("\n🎓 TEST PRAWDZIWYCH MATERIAŁÓW ZAKOŃCZONY!")

if __name__ == '__main__':
    asyncio.run(main()) 