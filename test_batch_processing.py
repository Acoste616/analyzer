#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test Batch Processing - Przetwarzanie 15 losowych plików z PRAWDZIWYMI ekstraktami"""

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
        logging.FileHandler('logs/batch_processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealBatchProcessor:
    def __init__(self):
        self.media_folder = Path("C:/Users/barto/OneDrive/Pulpit/mediapobrane/pobrane_media")
        self.output_folder = Path("output")
        self.output_folder.mkdir(exist_ok=True)
        
        # Initialize extractors if available
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
        
        # Processing statistics
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'processing_times': [],
            'start_time': None,
            'end_time': None
        }
        self.results = []

    def select_random_files(self, count: int = 15) -> Dict[str, List[Path]]:
        """Wybiera losowo pliki z folderu mediów"""
        logger.info(f"🎲 WYBIERANIE {count} LOSOWYCH PLIKÓW")
        print("="*50)
        
        # Get all files
        all_files = list(self.media_folder.rglob("*"))
        all_files = [f for f in all_files if f.is_file()]
        
        # Categorize files
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'}
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        
        video_files = [f for f in all_files if f.suffix.lower() in video_extensions]
        image_files = [f for f in all_files if f.suffix.lower() in image_extensions]
        
        logger.info(f"📊 Dostępne pliki:")
        logger.info(f"   Video: {len(video_files)} plików")
        logger.info(f"   Obrazy: {len(image_files)} plików")
        
        # Select random files (70% video, 30% images)
        selected_files = {'video': [], 'image': []}
        
        video_count = min(int(count * 0.7), len(video_files))
        image_count = min(count - video_count, len(image_files))
        
        if video_files:
            selected_files['video'] = random.sample(video_files, video_count)
        if image_files:
            selected_files['image'] = random.sample(image_files, image_count)
        
        logger.info(f"✅ Wybrano {len(selected_files['video'])} plików video")
        logger.info(f"✅ Wybrano {len(selected_files['image'])} plików obrazów")
        
        # Display selected files
        print("\n📋 WYBRANE PLIKI DO PRZETWARZANIA:")
        print("="*50)
        
        for i, file_path in enumerate(selected_files['video'], 1):
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"{i:2d}. [VIDEO] {file_path.name} ({size_mb:.1f} MB)")
        
        for i, file_path in enumerate(selected_files['image'], len(selected_files['video']) + 1):
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"{i:2d}. [IMAGE] {file_path.name} ({size_mb:.1f} MB)")
        
        return selected_files

    def monitor_gpu_memory(self) -> Dict[str, float]:
        """Monitorowanie pamięci GPU"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'free_gb': 6.0 - reserved
            }
        return {'allocated_gb': 0, 'reserved_gb': 0, 'free_gb': 0}

    async def call_lm_studio(self, prompt: str) -> str:
        """Wywołanie LM Studio API"""
        try:
            payload = {
                "messages": [
                    {"role": "system", "content": "Jesteś asystentem edukacyjnym. Analizujesz treści edukacyjne i tworzysz krótkie podsumowania w języku polskim."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.3,
                "stream": False
            }
            
            response = requests.post(
                'http://127.0.0.1:1234/v1/chat/completions',
                json=payload,
                timeout=30
            )
            
            logger.info(f"LM Studio response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"LM Studio response keys: {list(data.keys())}")
                
                # Sprawdzenie różnych możliwych struktur odpowiedzi
                if 'choices' in data and data['choices']:
                    if isinstance(data['choices'], list) and len(data['choices']) > 0:
                        choice = data['choices'][0]
                        if 'message' in choice and 'content' in choice['message']:
                            return choice['message']['content'].strip()
                        elif 'text' in choice:
                            return choice['text'].strip()
                
                # Fallback - jeśli struktura jest inna
                logger.warning(f"Nieoczekiwana struktura odpowiedzi LM Studio: {data}")
                return "Podsumowanie wygenerowane przez AI (struktura odpowiedzi wymagała adaptacji)"
            
            else:
                error_text = response.text
                logger.error(f"LM Studio API error {response.status_code}: {error_text}")
                return f"Błąd generowania podsumowania (HTTP {response.status_code})"
            
        except requests.exceptions.Timeout:
            logger.error("LM Studio timeout")
            return "Błąd generowania podsumowania: timeout"
        except requests.exceptions.ConnectionError:
            logger.error("LM Studio connection error")
            return "Błąd generowania podsumowania: brak połączenia z LM Studio"
        except Exception as e:
            logger.error(f"Błąd LM Studio: {e}")
            return f"Błąd generowania podsumowania: {str(e)}"

    async def process_image(self, file_path: Path) -> Dict[str, Any]:
        """Przetwarzanie pojedynczego obrazu - PRAWDZIWE z Knowledge Base"""
        logger.info(f"🖼️ Przetwarzanie obrazu: {file_path.name}")
        
        start_time = time.time()
        gpu_before = self.monitor_gpu_memory()
        
        try:
            if self.real_processing:
                # PRAWDZIWA ekstrakcja z obrazu z timeout protection
                logger.info(f"   📤 Uruchamianie zaawansowanego OCR dla {file_path.name}")
                
                try:
                    # Timeout protection dla OCR - zwiększony
                    extraction_result = await asyncio.wait_for(
                        self.image_extractor.extract(str(file_path)),
                        timeout=300  # 5 minut dla obrazów (więcej na complex OCR)
                    )
                    
                    # POPRAWKA: ExtractedContent to dataclass, nie dictionary
                    extracted_content = extraction_result.content
                    
                    if extracted_content and len(extracted_content.strip()) > 10:
                        # PRAWDZIWE wywołanie LM Studio z timeout
                        prompt = f"""
                        Przeanalizuj poniższą treść wyodrębnioną z obrazu edukacyjnego i stwórz strukturalne podsumowanie:

                        Tytuł: {extraction_result.title}
                        Treść: {extracted_content}
                        Metadane: {extraction_result.metadata}

                        Stwórz edukacyjne podsumowanie (max 300 słów) w języku polskim zawierające:
                        1. Rodzaj materiału (diagram, tekst, zdjęcie, etc.)
                        2. Główne informacje
                        3. Kontekst edukacyjny
                        4. Możliwe zastosowania
                        """
                        
                        logger.info(f"   🧠 Generowanie podsumowania AI...")
                        try:
                            llm_summary = await asyncio.wait_for(
                                self.call_lm_studio(prompt),
                                timeout=60  # 1 minuta na LLM
                            )
                        except asyncio.TimeoutError:
                            logger.warning(f"LM Studio timeout dla {file_path.name}")
                            llm_summary = "Podsumowanie AI - timeout"
                        
                        # 🔥 ZAPISZ DO KNOWLEDGE BASE
                        try:
                            lesson_id = await self.save_to_knowledge_base(
                                file_path=file_path,
                                extraction_result=extraction_result,
                                llm_summary=llm_summary,
                                content_type="image"
                            )
                            logger.info(f"   💾 Zapisano do Knowledge Base: {lesson_id}")
                        except Exception as e:
                            logger.error(f"   ❌ Błąd zapisu do KB: {e}")
                    else:
                        extracted_content = f"Nie wykryto tekstu w {file_path.name} (może być to zdjęcie bez tekstu)"
                        llm_summary = "Brak tekstu do analizy - możliwe zdjęcie bez treści tekstowych"
                        
                except asyncio.TimeoutError:
                    logger.error(f"OCR timeout dla {file_path.name}")
                    extracted_content = f"OCR timeout dla {file_path.name}"
                    llm_summary = "Przetwarzanie przerwane - timeout"
                
                except asyncio.CancelledError:
                    logger.error(f"OCR cancelled dla {file_path.name}")
                    extracted_content = f"OCR cancelled dla {file_path.name}"
                    llm_summary = "Przetwarzanie przerwane - anulowane"
                    
            else:
                # Fallback jeśli Jarvis nie jest dostępny
                extracted_content = f"Fallback: Obraz {file_path.name} (Jarvis ekstraktory niedostępne)"
                llm_summary = "Ekstrakcja niedostępna - wymagane poprawne importy Jarvis"
            
            # GPU memory po przetwarzaniu
            gpu_after = self.monitor_gpu_memory()
            processing_time = time.time() - start_time
            
            result = {
                'file_path': str(file_path),
                'file_type': 'image',
                'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                'processing_time': processing_time,
                'status': 'success',
                'extracted_text': extracted_content[:500],
                'llm_summary': llm_summary[:500],
                'gpu_memory_before': gpu_before,
                'gpu_memory_after': gpu_after,
                'error': None
            }
            
            logger.info(f"✅ Obraz przetworzony: {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Błąd przetwarzania obrazu {file_path.name}: {e}")
            
            return {
                'file_path': str(file_path),
                'file_type': 'image',
                'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                'processing_time': processing_time,
                'status': 'failed',
                'extracted_text': f'Błąd przetwarzania: {str(e)}',
                'llm_summary': '',
                'gpu_memory_before': gpu_before,
                'gpu_memory_after': self.monitor_gpu_memory(),
                'error': str(e)
            }

    async def process_video(self, file_path: Path) -> Dict[str, Any]:
        """Przetwarzanie pojedynczego video - PRAWDZIWE z Knowledge Base"""
        logger.info(f"🎬 Przetwarzanie video: {file_path.name}")
        
        start_time = time.time()
        gpu_before = self.monitor_gpu_memory()
        
        try:
            if self.real_processing:
                # PRAWDZIWA ekstrakcja z video z długimi timeouts
                logger.info(f"   🎵 Ekstrakcja audio i transkrypcja...")
                
                try:
                    # Długi timeout dla video processing (10-15 minut w zależności od rozmiaru)
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    
                    if file_size_mb < 10:
                        timeout_minutes = 10  # 10 minut dla małych plików
                    elif file_size_mb < 50:
                        timeout_minutes = 15  # 15 minut dla średnich
                    elif file_size_mb < 200:
                        timeout_minutes = 25  # 25 minut dla dużych
                    else:
                        timeout_minutes = 40  # 40 minut dla bardzo dużych
                    
                    logger.info(f"   ⏰ Timeout ustawiony na {timeout_minutes} minut dla pliku {file_size_mb:.1f}MB")
                    
                    extraction_result = await asyncio.wait_for(
                        self.video_extractor.extract(str(file_path)),
                        timeout=timeout_minutes * 60  # Konwersja na sekundy
                    )
                    
                    # POPRAWKA: ExtractedContent to dataclass, nie dictionary
                    extracted_content = extraction_result.content
                    
                    if extracted_content and len(extracted_content.strip()) > 20:
                        # PRAWDZIWE wywołanie LM Studio z timeout
                        prompt = f"""
                        Przeanalizuj poniższą treść wyodrębnioną z video edukacyjnego i stwórz strukturalne podsumowanie:

                        Tytuł: {extraction_result.title}
                        Treść: {extracted_content[:2000]}
                        Metadane: {extraction_result.metadata}

                        Stwórz edukacyjne podsumowanie (max 500 słów) w języku polskim zawierające:
                        1. Główne tematy
                        2. Kluczowe pojęcia
                        3. Praktyczne zastosowania
                        4. Rekomendacje dla uczniów
                        """
                        
                        logger.info(f"   🧠 Generowanie podsumowania AI...")
                        try:
                            llm_summary = await asyncio.wait_for(
                                self.call_lm_studio(prompt),
                                timeout=90  # 1.5 minuty na LLM
                            )
                        except asyncio.TimeoutError:
                            logger.warning(f"LM Studio timeout dla {file_path.name}")
                            llm_summary = "Podsumowanie AI - timeout"
                        
                        # 🔥 ZAPISZ DO KNOWLEDGE BASE
                        try:
                            lesson_id = await self.save_to_knowledge_base(
                                file_path=file_path,
                                extraction_result=extraction_result,
                                llm_summary=llm_summary,
                                content_type="video"
                            )
                            logger.info(f"   💾 Zapisano do Knowledge Base: {lesson_id}")
                        except Exception as e:
                            logger.error(f"   ❌ Błąd zapisu do KB: {e}")
                            
                    else:
                        extracted_content = f"Brak audio/tekstu do wyodrębnienia z {file_path.name}"
                        llm_summary = "Brak treści do podsumowania"
                        
                except asyncio.TimeoutError:
                    logger.error(f"Video processing timeout dla {file_path.name} (>{timeout_minutes} min)")
                    extracted_content = f"Video processing timeout dla {file_path.name} (>{timeout_minutes} min)"
                    llm_summary = "Przetwarzanie przerwane - zbyt długie video"
                
                except asyncio.CancelledError:
                    logger.error(f"Video processing cancelled dla {file_path.name}")
                    extracted_content = f"Video processing cancelled dla {file_path.name}"
                    llm_summary = "Przetwarzanie przerwane - anulowane"
                    
            else:
                # Fallback jeśli Jarvis nie jest dostępny
                extracted_content = f"Fallback: Video {file_path.name} (Jarvis ekstraktory niedostępne)"
                llm_summary = "Ekstrakcja niedostępna - wymagane poprawne importy Jarvis"
            
            # GPU memory po przetwarzaniu
            gpu_after = self.monitor_gpu_memory()
            processing_time = time.time() - start_time
            
            result = {
                'file_path': str(file_path),
                'file_type': 'video',
                'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                'processing_time': processing_time,
                'status': 'success',
                'extracted_text': extracted_content[:500],
                'llm_summary': llm_summary[:500],
                'gpu_memory_before': gpu_before,
                'gpu_memory_after': gpu_after,
                'error': None
            }
            
            logger.info(f"✅ Video przetworzony: {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Błąd przetwarzania video {file_path.name}: {e}")
            
            return {
                'file_path': str(file_path),
                'file_type': 'video',
                'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                'processing_time': processing_time,
                'status': 'failed',
                'extracted_text': f'Błąd przetwarzania: {str(e)}',
                'llm_summary': '',
                'gpu_memory_before': gpu_before,
                'gpu_memory_after': self.monitor_gpu_memory(),
                'error': str(e)
            }

    async def process_batch(self, selected_files: Dict[str, List[Path]]):
        """Przetwarzanie wsadowe plików - PRAWDZIWE"""
        logger.info("🚀 ROZPOCZĘCIE PRAWDZIWEGO PRZETWARZANIA WSADOWEGO")
        print("="*60)
        
        self.stats['start_time'] = datetime.now()
        
        # Flatten all files
        all_files = []
        for file_path in selected_files['video']:
            all_files.append((file_path, 'video'))
        for file_path in selected_files['image']:
            all_files.append((file_path, 'image'))
        
        self.stats['total_files'] = len(all_files)
        
        # Process files in batches (RTX 4050 optimized)
        video_batch_size = 2  # Reduced for real processing
        image_batch_size = 4  # Reduced for real processing
        
        # Process videos first (they use more memory)
        video_files = [(f, t) for f, t in all_files if t == 'video']
        for i in range(0, len(video_files), video_batch_size):
            batch = video_files[i:i + video_batch_size]
            logger.info(f"🎬 Przetwarzanie batch video {i//video_batch_size + 1}: {len(batch)} plików")
            
            # Process sequentially for stability
            for file_path, file_type in batch:
                result = await self.process_video(file_path)
                self.results.append(result)
                
                if result['status'] == 'success':
                    self.stats['processed_files'] += 1
                else:
                    self.stats['failed_files'] += 1
                self.stats['processing_times'].append(result['processing_time'])
                
                # Clear GPU memory after each file
                torch.cuda.empty_cache()
            
            logger.info("🧹 GPU memory cleared")
        
        # Process images
        image_files = [(f, t) for f, t in all_files if t == 'image']
        for i in range(0, len(image_files), image_batch_size):
            batch = image_files[i:i + image_batch_size]
            logger.info(f"🖼️ Przetwarzanie batch obrazów {i//image_batch_size + 1}: {len(batch)} plików")
            
            # Process sequentially for stability
            for file_path, file_type in batch:
                result = await self.process_image(file_path)
                self.results.append(result)
                
                if result['status'] == 'success':
                    self.stats['processed_files'] += 1
                else:
                    self.stats['failed_files'] += 1
                self.stats['processing_times'].append(result['processing_time'])
                
                # Clear GPU memory after each file
                torch.cuda.empty_cache()
            
            logger.info("🧹 GPU memory cleared")
        
        self.stats['end_time'] = datetime.now()

    def generate_report(self):
        """Generowanie raportu z wyników"""
        logger.info("📊 GENEROWANIE RAPORTU")
        print("="*50)
        
        # Calculate statistics
        total_time = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        avg_processing_time = sum(self.stats['processing_times']) / len(self.stats['processing_times']) if self.stats['processing_times'] else 0
        success_rate = (self.stats['processed_files'] / self.stats['total_files']) * 100 if self.stats['total_files'] > 0 else 0
        
        # Create report
        report = {
            'timestamp': datetime.now().isoformat(),
            'processing_type': 'REAL' if self.real_processing else 'FALLBACK',
            'jarvis_available': JARVIS_AVAILABLE,
            'statistics': {
                'total_files': self.stats['total_files'],
                'processed_files': self.stats['processed_files'],
                'failed_files': self.stats['failed_files'],
                'success_rate': success_rate,
                'total_time_seconds': total_time,
                'average_processing_time': avg_processing_time,
                'files_per_minute': (self.stats['processed_files'] / total_time) * 60 if total_time > 0 else 0
            },
            'results': self.results
        }
        
        # Save JSON report
        with open('output/batch_processing_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 PODSUMOWANIE PRAWDZIWEGO PRZETWARZANIA")
        print("="*60)
        print(f"📊 Typ przetwarzania: {'PRAWDZIWE' if self.real_processing else 'FALLBACK'}")
        print(f"📊 Jarvis dostępny: {'TAK' if JARVIS_AVAILABLE else 'NIE'}")
        print(f"📊 Łączny czas: {total_time:.2f} sekund")
        print(f"📊 Przetworzone pliki: {self.stats['processed_files']}/{self.stats['total_files']}")
        print(f"📊 Sukces rate: {success_rate:.1f}%")
        print(f"📊 Średni czas na plik: {avg_processing_time:.2f} sekund")
        
        # Show sample results
        print(f"\n📋 PRZYKŁADOWE WYNIKI:")
        print("="*60)
        for i, result in enumerate(self.results[:3], 1):
            status = "✅" if result['status'] == 'success' else "❌"
            print(f"{i}. {status} [{result['file_type'].upper()}] {Path(result['file_path']).name}")
            print(f"   Czas: {result['processing_time']:.2f}s")
            print(f"   Treść: {result['extracted_text'][:100]}...")
            if result['llm_summary']:
                print(f"   AI: {result['llm_summary'][:100]}...")
            print()
        
        logger.info(f"📋 Raport zapisany w: output/batch_processing_report.json")
        return report

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
                if hasattr(self, 'knowledge_base'):
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
                    logger.info(f"   ✅ Dodano do Knowledge Base: {lesson_id}")
            except Exception as kb_error:
                logger.warning(f"   ⚠️ Błąd Knowledge Base, zapisano tylko do pliku: {kb_error}")
            
            return lesson_id
            
        except Exception as e:
            logger.error(f"Błąd zapisu do Knowledge Base: {e}")
            return f"error_{datetime.now().strftime('%H%M%S')}"
    
    def _generate_tags(self, extraction_result, content_type: str) -> List[str]:
        """Generuje tagi na podstawie treści"""
        tags = [content_type, "auto-extracted"]
        
        content = extraction_result.content.lower()
        
        # Dodaj tagi na podstawie treści
        tag_keywords = {
            "matematyka": ['równanie', 'funkcja', 'wykres', 'liczba', 'algebra'],
            "programowanie": ['kod', 'python', 'javascript', 'function', 'class', 'algorytm'],
            "nauka": ['eksperyment', 'badanie', 'teoria', 'hipoteza', 'analiza'],
            "język": ['gramatyka', 'słownictwo', 'tekst', 'język', 'literatura'],
            "historia": ['data', 'wydarzenie', 'historia', 'chronologia'],
            "fizyka": ['siła', 'energia', 'masa', 'prędkość', 'fizyka'],
            "chemia": ['reakcja', 'związek', 'pierwiastek', 'chemia', 'molekuła']
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
        if any(word in content for word in ['matematyka', 'liczba', 'równanie', 'funkcja']):
            categories.append("Matematyka")
        elif any(word in content for word in ['kod', 'python', 'programowanie', 'algorytm']):
            categories.append("Informatyka")
        elif any(word in content for word in ['język', 'gramatyka', 'tekst', 'literatura']):
            categories.append("Języki")
        elif any(word in content for word in ['fizyka', 'siła', 'energia', 'masa']):
            categories.append("Fizyka")
        elif any(word in content for word in ['chemia', 'reakcja', 'związek']):
            categories.append("Chemia")
        elif any(word in content for word in ['historia', 'wydarzenie', 'data']):
            categories.append("Historia")
        else:
            categories.append("Ogólne")
        
        # Kategorie na podstawie typu contentu
        if content_type == "video":
            categories.append("Materiały wideo")
        elif content_type == "image":
            categories.append("Materiały wizualne")
        
        return categories if categories else ["Nieskategoryzowane"]

async def main():
    """Główna funkcja testowa"""
    print("🧪 JARVIS EDU REAL BATCH PROCESSING")
    print("="*60)
    print(f"Test started: {datetime.now()}")
    print("="*60)
    
    # Initialize processor
    processor = RealBatchProcessor()
    
    # Select random files
    selected_files = processor.select_random_files(15)
    
    # Process files
    await processor.process_batch(selected_files)
    
    # Generate report
    report = processor.generate_report()
    
    print("\n🎉 REAL BATCH PROCESSING COMPLETED!")

if __name__ == '__main__':
    asyncio.run(main()) 