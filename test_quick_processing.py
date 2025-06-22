#!/usr/bin/env python3
"""Quick Test - 3 pliki aby sprawdzić naprawki"""

import os
import sys
import random
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
        logging.FileHandler('logs/quick_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QuickTestProcessor:
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

    def monitor_gpu_memory(self) -> Dict[str, Any]:
        """Monitor GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'free_gb': total - reserved,
                'total_gb': total,
                'usage_percent': (reserved / total) * 100
            }
        return {
            'allocated_gb': 0,
            'reserved_gb': 0,
            'free_gb': 0,
            'total_gb': 0,
            'usage_percent': 0
        }

    async def save_to_knowledge_base(self, file_path: Path, extraction_result, llm_summary: str, content_type: str) -> str:
        """Zapisuje wyniki do Knowledge Base"""
        try:
            # Generowanie unikalnego ID lekcji
            from datetime import datetime
            import hashlib
            
            file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
            lesson_id = f"{content_type}_{file_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Przygotowanie danych lekcji
            lesson_data = {
                'lesson_id': lesson_id,
                'title': extraction_result.title,
                'summary': llm_summary[:500],  # Ograniczenie długości
                'content': extraction_result.content,
                'source_file': str(file_path),
                'content_type': content_type,
                'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                'extraction_metadata': extraction_result.metadata,
                'confidence': extraction_result.confidence,
                'language': extraction_result.language,
                'created_at': datetime.now().isoformat(),
                'tags': self._generate_tags(extraction_result, content_type),
                'categories': self._generate_categories(extraction_result, content_type)
            }
            
            # Zapisanie do pliku (jako backup) i bazy
            kb_file = self.output_folder / f"knowledge_base_{lesson_id}.json"
            with open(kb_file, 'w', encoding='utf-8') as f:
                json.dump(lesson_data, f, indent=2, ensure_ascii=False)
            
            # Próba zapisu do rzeczywistej bazy Knowledge Base
            try:
                if hasattr(self, 'knowledge_base') and self.knowledge_base:
                    # Dodanie lekcji do Knowledge Base
                    self.knowledge_base.add_lesson(
                        lesson_id=lesson_id,
                        title=extraction_result.title,
                        summary=llm_summary[:500],
                        content_path=str(kb_file),
                        tags=lesson_data['tags'],
                        categories=lesson_data['categories'],
                        difficulty='intermediate',  # Default
                        duration=10,  # Estimated 10 minutes
                        topics_count=1,
                        metadata=lesson_data['extraction_metadata']
                    )
                    logger.info(f"   ✅ Dodano do Knowledge Base database: {lesson_id}")
            except Exception as kb_error:
                logger.warning(f"   ⚠️ Błąd Knowledge Base database, zapisano tylko do pliku: {kb_error}")
            
            return lesson_id
            
        except Exception as e:
            logger.error(f"Błąd zapisu do Knowledge Base: {e}")
            from datetime import datetime
            return f"error_{datetime.now().strftime('%H%M%S')}"
    
    def _generate_tags(self, extraction_result, content_type: str) -> List[str]:
        """Generuje tagi na podstawie treści"""
        tags = [content_type, "auto-extracted", "quick-test"]
        
        content = extraction_result.content.lower()
        
        # Dodaj tagi na podstawie treści
        tag_keywords = {
            "matematyka": ['równanie', 'funkcja', 'wykres', 'liczba', 'algebra', 'geometria'],
            "programowanie": ['kod', 'python', 'javascript', 'function', 'class', 'algorytm', 'script'],
            "nauka": ['eksperyment', 'badanie', 'teoria', 'hipoteza', 'analiza', 'research'],
            "język": ['gramatyka', 'słownictwo', 'tekst', 'język', 'literatura', 'tłumaczenie'],
            "historia": ['data', 'wydarzenie', 'historia', 'chronologia', 'period', 'century'],
            "fizyka": ['siła', 'energia', 'masa', 'prędkość', 'fizyka', 'physics', 'force'],
            "chemia": ['reakcja', 'związek', 'pierwiastek', 'chemia', 'molekuła', 'chemistry'],
            "edukacja": ['lekcja', 'uczenie', 'nauka', 'student', 'nauczyciel', 'szkoła']
        }
        
        for tag, keywords in tag_keywords.items():
            if any(keyword in content for keyword in keywords):
                tags.append(tag)
        
        # Dodaj tagi na podstawie metadanych
        if extraction_result.confidence > 0.8:
            tags.append("high-confidence")
        elif extraction_result.confidence > 0.5:
            tags.append("medium-confidence")
        else:
            tags.append("low-confidence")
        
        return tags
    
    def _generate_categories(self, extraction_result, content_type: str) -> List[str]:
        """Generuje kategorie na podstawie treści"""
        categories = []
        
        content = extraction_result.content.lower()
        
        # Główne kategorie edukacyjne
        if any(word in content for word in ['matematyka', 'liczba', 'równanie', 'funkcja', 'algebra']):
            categories.append("Matematyka")
        elif any(word in content for word in ['kod', 'python', 'programowanie', 'algorytm', 'script']):
            categories.append("Informatyka")
        elif any(word in content for word in ['język', 'gramatyka', 'tekst', 'literatura', 'słowo']):
            categories.append("Języki")
        elif any(word in content for word in ['fizyka', 'siła', 'energia', 'masa', 'physics']):
            categories.append("Fizyka")
        elif any(word in content for word in ['chemia', 'reakcja', 'związek', 'chemistry']):
            categories.append("Chemia")
        elif any(word in content for word in ['historia', 'wydarzenie', 'data', 'period']):
            categories.append("Historia")
        elif any(word in content for word in ['biologia', 'organizm', 'komórka', 'dna']):
            categories.append("Biologia")
        else:
            categories.append("Ogólne")
        
        # Kategorie na podstawie typu contentu
        if content_type == "video":
            categories.append("Materiały wideo")
        elif content_type == "image":
            categories.append("Materiały wizualne")
        elif content_type == "audio":
            categories.append("Materiały audio")
        
        return categories if categories else ["Nieskategoryzowane"]

    def select_test_files(self) -> Dict[str, List[Path]]:
        """Wybiera 3 małe pliki do testu"""
        logger.info("🎲 WYBIERANIE 3 MAŁYCH PLIKÓW DO TESTU")
        print("="*50)
        
        all_files = list(self.media_folder.rglob("*"))
        all_files = [f for f in all_files if f.is_file()]
        
        # Filtruj małe pliki (< 5MB)
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'}
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        
        small_videos = [f for f in all_files 
                       if f.suffix.lower() in video_extensions 
                       and f.stat().st_size < 5 * 1024 * 1024]  # < 5MB
        
        small_images = [f for f in all_files 
                       if f.suffix.lower() in image_extensions 
                       and f.stat().st_size < 1 * 1024 * 1024]  # < 1MB
        
        selected_files = {'video': [], 'image': []}
        
        # Wybierz 2 małe video + 1 obraz
        if small_videos:
            selected_files['video'] = random.sample(small_videos, min(2, len(small_videos)))
        if small_images:
            selected_files['image'] = random.sample(small_images, min(1, len(small_images)))
        
        logger.info(f"✅ Wybrano {len(selected_files['video'])} małych video")
        logger.info(f"✅ Wybrano {len(selected_files['image'])} małych obrazów")
        
        print("\n📋 WYBRANE PLIKI DO TESTU:")
        print("="*50)
        
        for i, file_path in enumerate(selected_files['video'], 1):
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"{i}. [VIDEO] {file_path.name} ({size_mb:.1f} MB)")
        
        for i, file_path in enumerate(selected_files['image'], len(selected_files['video']) + 1):
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"{i}. [IMAGE] {file_path.name} ({size_mb:.1f} MB)")
        
        return selected_files

    async def call_lm_studio(self, prompt: str) -> str:
        """Wywołanie LM Studio API z naprawkami"""
        try:
            payload = {
                "messages": [
                    {"role": "system", "content": "Jesteś asystentem edukacyjnym. Analizujesz treści edukacyjne i tworzysz krótkie podsumowania w języku polskim."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 300,
                "temperature": 0.3,
                "stream": False
            }
            
            response = requests.post(
                'http://127.0.0.1:1234/v1/chat/completions',
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and data['choices']:
                    if isinstance(data['choices'], list) and len(data['choices']) > 0:
                        choice = data['choices'][0]
                        if 'message' in choice and 'content' in choice['message']:
                            return choice['message']['content'].strip()
                
                return "Podsumowanie AI (struktura odpowiedzi nieoczekiwana)"
            else:
                return f"Błąd LM Studio (HTTP {response.status_code})"
            
        except Exception as e:
            logger.error(f"Błąd LM Studio: {e}")
            return f"Błąd generowania: {str(e)}"

    async def process_file(self, file_path: Path, file_type: str) -> Dict[str, Any]:
        """Przetwarzanie pojedynczego pliku - NAPRAWIONA WERSJA"""
        logger.info(f"🔄 Przetwarzanie {file_type}: {file_path.name}")
        
        start_time = time.time()
        gpu_before = self.monitor_gpu_memory()
        
        try:
            if self.real_processing:
                if file_type == 'video':
                    # VIDEO PROCESSING z długimi timeouts
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    
                    if file_size_mb < 10:
                        timeout_minutes = 15  # 15 minut dla małych plików
                    elif file_size_mb < 50:
                        timeout_minutes = 25  # 25 minut dla średnich
                    elif file_size_mb < 200:
                        timeout_minutes = 40  # 40 minut dla dużych
                    else:
                        timeout_minutes = 60  # 60 minut dla bardzo dużych
                    
                    logger.info(f"   ⏰ Video timeout: {timeout_minutes} minut dla {file_size_mb:.1f}MB")
                    
                    extraction_result = await asyncio.wait_for(
                        self.video_extractor.extract(str(file_path)),
                        timeout=timeout_minutes * 60
                    )
                    
                elif file_type == 'image':
                    # IMAGE PROCESSING z poprawkami OCR
                    logger.info(f"   🖼️ Uruchamianie zaawansowanego OCR...")
                    
                    # Ustaw PATH dla Tesseract
                    import pytesseract
                    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                    
                    extraction_result = await asyncio.wait_for(
                        self.image_extractor.extract(str(file_path)),
                        timeout=600  # 10 minut dla obrazów (complex OCR)
                    )
                    
                else:
                    raise ValueError(f"Nieobsługiwany typ: {file_type}")
                
                # POPRAWKA: ExtractedContent to dataclass, nie dictionary
                extracted_content = extraction_result.content
                
                if extracted_content and len(extracted_content.strip()) > 5:
                    # Enhanced LM Studio prompt
                    if file_type == 'video':
                        prompt = f"""
                        ANALIZA VIDEO EDUKACYJNEGO:

                        Tytuł: {extraction_result.title}
                        Transkrypcja: {extracted_content[:1500]}
                        Metadane: {extraction_result.metadata}

                        Stwórz strukturalne podsumowanie (max 400 słów) w języku polskim:
                        1. Główne tematy omawiane w video
                        2. Kluczowe pojęcia i definicje
                        3. Praktyczne zastosowania
                        4. Rekomendacje dla uczniów
                        """
                    else:  # image
                        prompt = f"""
                        ANALIZA OBRAZU/TEKSTU EDUKACYJNEGO:

                        Tytuł: {extraction_result.title}
                        Wyodrębniony tekst: {extracted_content}
                        Metadane: {extraction_result.metadata}

                        Stwórz strukturalne podsumowanie (max 250 słów) w języku polskim:
                        1. Rodzaj materiału (diagram, tekst, schemat, etc.)
                        2. Główne informacje zawarte
                        3. Kontekst edukacyjny
                        4. Możliwe zastosowania w nauce
                        """
                    
                    logger.info(f"   🧠 Generowanie AI podsumowania...")
                    llm_summary = await asyncio.wait_for(
                        self.call_lm_studio(prompt),
                        timeout=90  # 1.5 minuty na LLM
                    )
                    
                    # 🔥 ZAPISZ DO KNOWLEDGE BASE
                    try:
                        lesson_id = await self.save_to_knowledge_base(
                            file_path=file_path,
                            extraction_result=extraction_result,
                            llm_summary=llm_summary,
                            content_type=file_type
                        )
                        logger.info(f"   💾 Zapisano do Knowledge Base: {lesson_id}")
                    except Exception as e:
                        logger.error(f"   ❌ Błąd zapisu do KB: {e}")
                        
                else:
                    if file_type == 'image':
                        extracted_content = f"OCR nie wykrył tekstu w {file_path.name} (możliwe zdjęcie bez tekstu)"
                        llm_summary = "Brak tekstu do analizy - prawdopodobnie materiał wizualny bez treści tekstowych"
                    else:
                        extracted_content = f"Brak audio/transkrypcji w {file_path.name}"
                        llm_summary = "Brak treści do podsumowania"
                    
            else:
                extracted_content = f"Fallback: {file_path.name} (Jarvis niedostępny)"
                llm_summary = "Fallback processing"
            
            gpu_after = self.monitor_gpu_memory()
            processing_time = time.time() - start_time
            
            result = {
                'file_path': str(file_path),
                'file_type': file_type,
                'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                'processing_time': processing_time,
                'status': 'success',
                'extracted_text': extracted_content[:500],
                'llm_summary': llm_summary[:500],
                'gpu_memory_before': gpu_before,
                'gpu_memory_after': gpu_after,
                'error': None
            }
            
            logger.info(f"✅ {file_type} przetworzony: {processing_time:.2f}s")
            return result
            
        except asyncio.TimeoutError:
            processing_time = time.time() - start_time
            timeout_msg = f"⏰ Timeout dla {file_path.name} (>{timeout_minutes if file_type == 'video' else '10'} minut)"
            logger.error(timeout_msg)
            
            return {
                'file_path': str(file_path),
                'file_type': file_type,
                'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                'processing_time': processing_time,
                'status': 'timeout',
                'extracted_text': timeout_msg,
                'llm_summary': '',
                'gpu_memory_before': gpu_before,
                'gpu_memory_after': self.monitor_gpu_memory(),
                'error': 'timeout'
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Błąd {file_path.name}: {e}")
            
            return {
                'file_path': str(file_path),
                'file_type': file_type,
                'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                'processing_time': processing_time,
                'status': 'failed',
                'extracted_text': f'Błąd: {str(e)}',
                'llm_summary': '',
                'gpu_memory_before': gpu_before,
                'gpu_memory_after': self.monitor_gpu_memory(),
                'error': str(e)
            }

    async def process_all_files(self, selected_files: Dict[str, List[Path]]):
        """Przetwarzanie wszystkich plików"""
        logger.info("🚀 ROZPOCZĘCIE SZYBKIEGO TESTU")
        print("="*50)
        
        # Flatten all files
        all_files = []
        for file_path in selected_files['video']:
            all_files.append((file_path, 'video'))
        for file_path in selected_files['image']:
            all_files.append((file_path, 'image'))
        
        # Process sequentially for stability
        for file_path, file_type in all_files:
            result = await self.process_file(file_path, file_type)
            self.results.append(result)
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate_report(self):
        """Generowanie raportu"""
        logger.info("📊 GENEROWANIE RAPORTU TESTU")
        print("="*50)
        
        # Statistics
        total_files = len(self.results)
        success_files = len([r for r in self.results if r['status'] == 'success'])
        failed_files = len([r for r in self.results if r['status'] != 'success'])
        avg_time = sum(r['processing_time'] for r in self.results) / total_files if total_files > 0 else 0
        
        # Create report
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'QUICK_TEST',
            'processing_type': 'REAL' if self.real_processing else 'FALLBACK',
            'jarvis_available': JARVIS_AVAILABLE,
            'statistics': {
                'total_files': total_files,
                'success_files': success_files,
                'failed_files': failed_files,
                'success_rate': (success_files / total_files) * 100 if total_files > 0 else 0,
                'average_processing_time': avg_time
            },
            'results': self.results
        }
        
        # Save report
        with open('output/quick_test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 PODSUMOWANIE SZYBKIEGO TESTU")
        print("="*60)
        print(f"📊 Typ: {'PRAWDZIWE' if self.real_processing else 'FALLBACK'}")
        print(f"📊 Jarvis: {'DOSTĘPNY' if JARVIS_AVAILABLE else 'NIEDOSTĘPNY'}")
        print(f"📊 Pliki: {success_files}/{total_files} sukces")
        print(f"📊 Średni czas: {avg_time:.2f}s")
        
        print(f"\n📋 WYNIKI:")
        print("="*60)
        for i, result in enumerate(self.results, 1):
            status = "✅" if result['status'] == 'success' else "❌"
            print(f"{i}. {status} [{result['file_type'].upper()}] {Path(result['file_path']).name}")
            print(f"   Czas: {result['processing_time']:.2f}s | Status: {result['status']}")
            if result['extracted_text']:
                print(f"   Treść: {result['extracted_text'][:80]}...")
            if result['llm_summary']:
                print(f"   AI: {result['llm_summary'][:80]}...")
            print()
        
        logger.info("📋 Raport zapisany w: output/quick_test_report.json")
        return report

async def main():
    """Główna funkcja testu"""
    print("🧪 JARVIS EDU QUICK TEST - 3 PLIKI")
    print("="*60)
    print(f"Test started: {datetime.now()}")
    print("="*60)
    
    processor = QuickTestProcessor()
    selected_files = processor.select_test_files()
    
    if not selected_files['video'] and not selected_files['image']:
        print("❌ Brak małych plików do testu!")
        return
    
    await processor.process_all_files(selected_files)
    report = processor.generate_report()
    
    print("\n🎉 QUICK TEST COMPLETED!")

if __name__ == '__main__':
    asyncio.run(main()) 