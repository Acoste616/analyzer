import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import hashlib
import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import torch

from ..extractors import (
    EnhancedImageExtractor,
    EnhancedVideoExtractor, 
    EnhancedAudioExtractor
)
from ..processors.lm_studio import LMStudioProcessor
from ..storage.database import ProcessingQueue, FileStatus
from ..storage.knowledge_base import KnowledgeBase
from ..core.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ProcessingTask:
    file_path: Path
    file_type: str
    priority: int  # 1-highest, 5-lowest
    retry_count: int = 0
    last_error: Optional[str] = None

class ContinuousProcessor:
    """
    Główny procesor do ciągłego przetwarzania mediów.
    Optymalizowany pod długie video i batch processing obrazów.
    """
    
    def __init__(self, watch_folder: str, output_folder: str = "output"):
        self.watch_folder = Path(watch_folder)
        self.output_folder = Path(output_folder)
        self.processing_queue = ProcessingQueue()
        self.knowledge_base = KnowledgeBase()
        
        # Ekstraktory
        self.image_extractor = EnhancedImageExtractor()
        self.video_extractor = EnhancedVideoExtractor() 
        self.audio_extractor = EnhancedAudioExtractor()
        self.llm_processor = LMStudioProcessor()
        
        # Stan procesora
        self.processed_hashes: Set[str] = set()
        self.processing_tasks: asyncio.Queue[ProcessingTask] = asyncio.Queue()
        self.active_tasks: Dict[str, asyncio.Task] = {}
        
        # Konfiguracja
        self.max_concurrent_images = 5  # Obrazy są szybkie
        self.max_concurrent_videos = 2  # Video wymaga więcej zasobów
        self.scan_interval = 60  # Skanuj co minutę
        self.gpu_memory_threshold = 0.8  # 80% GPU memory
        
        # Thread pool dla IO operations
        self.io_executor = ThreadPoolExecutor(max_workers=4)
        
        self._running = False
        self.start_time = datetime.now()
        self.processed_today = 0
        self._load_processed_hashes()
    
    def _load_processed_hashes(self):
        """Załaduj już przetworzone pliki z cache"""
        cache_file = self.output_folder / "processed_files_cache.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                self.processed_hashes = set(data.get('hashes', []))
                logger.info(f"Loaded {len(self.processed_hashes)} processed files from cache")
    
    def _save_processed_hashes(self):
        """Zapisz cache przetworzonych plików"""
        cache_file = self.output_folder / "processed_files_cache.json"
        with open(cache_file, 'w') as f:
            json.dump({
                'hashes': list(self.processed_hashes),
                'last_update': datetime.now().isoformat()
            }, f)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Oblicz hash pliku dla deduplicacji"""
        hasher = hashlib.md5()
        # Use file stats for quick hash (size + mtime)
        stat = file_path.stat()
        hasher.update(f"{file_path.name}_{stat.st_size}_{stat.st_mtime}".encode())
        return hasher.hexdigest()
    
    async def start(self):
        """Uruchom continuous processor"""
        self._running = True
        logger.info(f"Starting continuous processor for: {self.watch_folder}")
        
        # Uruchom zadania
        tasks = [
            asyncio.create_task(self._folder_scanner()),
            asyncio.create_task(self._task_processor()),
            asyncio.create_task(self._gpu_monitor()),
            asyncio.create_task(self._periodic_cleanup()),
            asyncio.create_task(self._state_monitor())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Continuous processor stopped")
        finally:
            self._save_processed_hashes()
    
    async def _folder_scanner(self):
        """Skanuj folder w poszukiwaniu nowych plików"""
        while self._running:
            try:
                await self._scan_folder()
                await asyncio.sleep(self.scan_interval)
            except Exception as e:
                logger.error(f"Folder scanner error: {e}")
                await asyncio.sleep(5)
    
    async def _scan_folder(self):
        """Pojedyncze skanowanie folderu"""
        logger.debug("Scanning folder for new files...")
        
        # Wspierane rozszerzenia
        image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
        audio_exts = {'.mp3', '.wav', '.m4a', '.flac', '.ogg'}
        
        new_files = 0
        
        for file_path in self.watch_folder.rglob("*"):
            if not file_path.is_file():
                continue
                
            ext = file_path.suffix.lower()
            file_hash = self._calculate_file_hash(file_path)
            
            # Sprawdź czy już przetworzony
            if file_hash in self.processed_hashes:
                continue
            
            # Określ typ i priorytet
            if ext in image_exts:
                file_type = 'image'
                priority = 2  # Obrazy mają wyższy priorytet (szybsze)
            elif ext in video_exts:
                file_type = 'video'
                # Priorytet na podstawie rozmiaru
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb < 100:
                    priority = 3
                elif size_mb < 500:
                    priority = 4
                else:
                    priority = 5  # Duże video najniższy priorytet
            elif ext in audio_exts:
                file_type = 'audio'
                priority = 3
            else:
                continue
            
            # Dodaj do kolejki
            task = ProcessingTask(
                file_path=file_path,
                file_type=file_type,
                priority=priority
            )
            
            await self.processing_tasks.put(task)
            new_files += 1
            logger.info(f"Added to queue: {file_path.name} (priority: {priority})")
        
        if new_files > 0:
            logger.info(f"Found {new_files} new files to process")
    
    async def _task_processor(self):
        """Przetwarzaj zadania z kolejki"""
        while self._running:
            try:
                # Pobierz zadanie z kolejki (priorytetowo)
                task = await self._get_next_task()
                if not task:
                    await asyncio.sleep(1)
                    continue
                
                # Sprawdź czy możemy przetworzyć
                if await self._can_process_task(task):
                    # Utwórz zadanie asynchroniczne
                    process_task = asyncio.create_task(
                        self._process_single_file(task)
                    )
                    self.active_tasks[str(task.file_path)] = process_task
                else:
                    # Wróć do kolejki jeśli nie możemy przetworzyć
                    await self.processing_tasks.put(task)
                    await asyncio.sleep(5)
                    
            except Exception as e:
                logger.error(f"Task processor error: {e}")
                await asyncio.sleep(1)
    
    async def _get_next_task(self) -> Optional[ProcessingTask]:
        """Pobierz następne zadanie z priorytetem"""
        if self.processing_tasks.empty():
            return None
        
        # Pobierz wszystkie zadania i posortuj po priorytecie
        tasks = []
        while not self.processing_tasks.empty():
            try:
                task = self.processing_tasks.get_nowait()
                tasks.append(task)
            except asyncio.QueueEmpty:
                break
        
        if not tasks:
            return None
        
        # Sortuj po priorytecie (1 = najwyższy)
        tasks.sort(key=lambda t: t.priority)
        
        # Wróć resztę do kolejki
        for task in tasks[1:]:
            await self.processing_tasks.put(task)
        
        return tasks[0]
    
    async def _can_process_task(self, task: ProcessingTask) -> bool:
        """Sprawdź czy możemy przetworzyć zadanie"""
        # Sprawdź limity współbieżności
        active_images = sum(1 for t in self.active_tasks.values() 
                          if not t.done() and 'image' in str(t))
        active_videos = sum(1 for t in self.active_tasks.values() 
                          if not t.done() and 'video' in str(t))
        
        if task.file_type == 'image' and active_images >= self.max_concurrent_images:
            return False
        if task.file_type == 'video' and active_videos >= self.max_concurrent_videos:
            return False
        
        # Sprawdź GPU memory
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_reserved() / torch.cuda.get_device_properties(0).total_memory
            if gpu_usage > self.gpu_memory_threshold:
                logger.warning(f"GPU memory usage too high: {gpu_usage:.1%}")
                return False
        
        return True
    
    async def _process_single_file(self, task: ProcessingTask):
        """Przetwórz pojedynczy plik"""
        file_path = task.file_path
        file_hash = self._calculate_file_hash(file_path)
        
        try:
            logger.info(f"Processing {task.file_type}: {file_path.name}")
            start_time = datetime.now()
            
            # Wybierz ekstraktor
            if task.file_type == 'image':
                extractor = self.image_extractor
            elif task.file_type == 'video':
                extractor = self.video_extractor
            else:
                extractor = self.audio_extractor
            
            # Ustaw timeout na podstawie typu i rozmiaru
            timeout = self._calculate_timeout(task)
            
            # Ekstraktuj treść z timeoutem
            try:
                extraction_result = await asyncio.wait_for(
                    extractor.extract(str(file_path)),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.error(f"Timeout after {timeout}s for {file_path.name}")
                raise
            
            # Generuj podsumowanie AI
            if extraction_result.content and len(extraction_result.content) > 20:
                lesson_data = await self._generate_ai_lesson(
                    extraction_result, task.file_type
                )
            else:
                lesson_data = {
                    'title': extraction_result.title,
                    'summary': extraction_result.summary,
                    'tags': ['no-content', task.file_type],
                    'categories': ['Uncategorized']
                }
            
            # Zapisz do Knowledge Base
            lesson_id = await self._save_to_knowledge_base(
                file_path, extraction_result, lesson_data, task.file_type
            )
            
            # Oznacz jako przetworzony
            self.processed_hashes.add(file_hash)
            
            # Log sukcesu
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Processed {file_path.name} in {processing_time:.1f}s")
            
            # Update counters and save history
            self.processed_today += 1
            self._save_processing_history(file_path, task.file_type, processing_time, True)
            
            # Zapisz cache co 10 plików
            if len(self.processed_hashes) % 10 == 0:
                self._save_processed_hashes()
            
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")
            task.retry_count += 1
            task.last_error = str(e)
            
            # Save failed processing to history
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Retry logic
            if task.retry_count < 3:
                task.priority = min(task.priority + 1, 5)  # Obniż priorytet
                await self.processing_tasks.put(task)
                logger.info(f"Requeued {file_path.name} for retry {task.retry_count}/3")
            else:
                logger.error(f"Max retries reached for {file_path.name}")
                # Save failed processing to history (final failure)
                self._save_processing_history(file_path, task.file_type, processing_time, False)
                # Oznacz jako przetworzony żeby nie próbować w nieskończoność
                self.processed_hashes.add(file_hash)
        
        finally:
            # Usuń z aktywnych zadań
            self.active_tasks.pop(str(file_path), None)
            
            # Wyczyść GPU memory po dużych plikach
            if task.file_type == 'video' and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _calculate_timeout(self, task: ProcessingTask) -> int:
        """Oblicz timeout na podstawie typu i rozmiaru pliku"""
        size_mb = task.file_path.stat().st_size / (1024 * 1024)
        
        if task.file_type == 'image':
            # Obrazy: 1 minuta na każde 10MB, max 10 minut
            return min(int(size_mb / 10 * 60), 600)
        
        elif task.file_type == 'video':
            # Video: progresywny timeout
            if size_mb < 100:
                return 1800  # 30 minut
            elif size_mb < 500:
                return 3600  # 60 minut
            elif size_mb < 1000:
                return 5400  # 90 minut
            else:
                return 7200  # 120 minut dla bardzo dużych
        
        else:  # audio
            # Audio: 10 minut na każde 100MB
            return min(int(size_mb / 100 * 600), 3600)
    
    async def _generate_ai_lesson(self, extraction_result, file_type: str) -> Dict:
        """Generuj lekcję z AI"""
        prompt = f"""
        Przeanalizuj wyodrębnioną treść i stwórz strukturalne podsumowanie edukacyjne.
        
        Typ pliku: {file_type}
        Tytuł: {extraction_result.title}
        Treść: {extraction_result.content[:3000]}
        
        Wygeneruj w formacie JSON:
        - title: opisowy tytuł
        - summary: podsumowanie 2-3 zdania
        - main_topics: lista głównych tematów
        - tags: lista tagów (min 5)
        - categories: lista kategorii edukacyjnych
        - difficulty: poziom trudności (beginner/intermediate/advanced)
        - use_cases: zastosowania edukacyjne
        """
        
        try:
            response = await asyncio.wait_for(
                self.llm_processor.generate_lesson_from_prompt(prompt),
                timeout=60
            )
            
            # Merge z domyślnymi wartościami
            default = {
                'title': extraction_result.title,
                'summary': extraction_result.summary,
                'tags': [file_type, 'auto-processed'],
                'categories': ['General']
            }
            
            if isinstance(response, dict):
                default.update(response)
            
            return default
            
        except Exception as e:
            logger.error(f"AI lesson generation failed: {e}")
            return {
                'title': extraction_result.title,
                'summary': extraction_result.summary,
                'tags': [file_type, 'ai-failed'],
                'categories': ['General']
            }
    
    async def _save_to_knowledge_base(self, file_path: Path, extraction_result, 
                                    lesson_data: Dict, file_type: str) -> str:
        """Zapisz do knowledge base"""
        lesson_id = f"{file_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file_path.stem[:20]}"
        
        try:
            # Zapisz treść do pliku
            content_file = self.output_folder / "content" / f"{lesson_id}.txt"
            content_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(content_file, 'w', encoding='utf-8') as f:
                f.write(extraction_result.content)
            
            # Dodaj do bazy
            self.knowledge_base.add_lesson(
                lesson_id=lesson_id,
                title=lesson_data['title'],
                summary=lesson_data['summary'],
                content_path=str(content_file),
                tags=lesson_data.get('tags', []),
                categories=lesson_data.get('categories', []),
                difficulty=lesson_data.get('difficulty', 'intermediate'),
                duration=self._estimate_duration(extraction_result, file_type),
                topics_count=len(lesson_data.get('main_topics', [])),
                metadata={
                    'source_file': str(file_path),
                    'file_type': file_type,
                    'extraction_confidence': extraction_result.confidence,
                    'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                    'processed_at': datetime.now().isoformat(),
                    **extraction_result.metadata
                }
            )
            
            logger.info(f"Saved to knowledge base: {lesson_id}")
            return lesson_id
            
        except Exception as e:
            logger.error(f"Failed to save to knowledge base: {e}")
            return f"error_{datetime.now().timestamp()}"
    
    def _estimate_duration(self, extraction_result, file_type: str) -> int:
        """Estymuj czas trwania lekcji"""
        if file_type == 'video':
            # Use actual video duration if available
            duration = extraction_result.metadata.get('duration', 0)
            return int(duration / 60) if duration > 0 else 30
        
        elif file_type == 'image':
            # Based on text complexity
            word_count = len(extraction_result.content.split()) if extraction_result.content else 0
            return max(5, min(30, word_count // 50))
        
        else:  # audio
            duration = extraction_result.metadata.get('duration', 0)
            return int(duration / 60) if duration > 0 else 15
    
    async def _gpu_monitor(self):
        """Monitor GPU i zarządzaj pamięcią"""
        while self._running:
            try:
                if torch.cuda.is_available():
                    # Sprawdź użycie
                    allocated = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                    reserved = torch.cuda.memory_reserved() / torch.cuda.get_device_properties(0).total_memory
                    
                    logger.debug(f"GPU Memory: {allocated:.1%} allocated, {reserved:.1%} reserved")
                    
                    # Wyczyść jeśli za dużo
                    if reserved > 0.9:
                        logger.warning("GPU memory critical - clearing cache")
                        torch.cuda.empty_cache()
                        
                        # Zatrzymaj nowe zadania video
                        self.max_concurrent_videos = 0
                        await asyncio.sleep(10)
                        self.max_concurrent_videos = 2
                
                await asyncio.sleep(30)  # Check every 30s
                
            except Exception as e:
                logger.error(f"GPU monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _periodic_cleanup(self):
        """Okresowe czyszczenie i maintenance"""
        while self._running:
            try:
                # Co godzinę
                await asyncio.sleep(3600)
                
                # Zapisz cache
                self._save_processed_hashes()
                
                # Wyczyść stare pliki tymczasowe
                temp_dir = self.output_folder / "temp"
                if temp_dir.exists():
                    for file in temp_dir.glob("*"):
                        if file.stat().st_mtime < (datetime.now().timestamp() - 86400):
                            file.unlink()
                
                # Kompaktuj bazę danych
                self.knowledge_base.cleanup_old_entries(days=30)
                
                logger.info("Periodic cleanup completed")
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    def _save_processor_state(self):
        """Save current processor state for web dashboard"""
        try:
            # Calculate active files info
            active_files = []
            for file_path, task in self.active_tasks.items():
                if not task.done():
                    # Try to get task info from file path
                    path_obj = Path(file_path)
                    active_files.append({
                        'name': path_obj.name,
                        'type': path_obj.suffix[1:] if path_obj.suffix else 'unknown',
                        'progress': 50,  # Default progress - could be enhanced
                        'elapsed_time': int((datetime.now() - self.start_time).total_seconds())
                    })
            
            # Calculate uptime
            uptime_minutes = int((datetime.now() - self.start_time).total_seconds() / 60)
            
            state = {
                'active_tasks': len(self.active_tasks),
                'queued_tasks': self.processing_tasks.qsize(),
                'processed_today': self.processed_today,
                'uptime_minutes': uptime_minutes,
                'active_files': active_files,
                'running': self._running,
                'last_update': datetime.now().isoformat(),
                'start_time': self.start_time.isoformat(),
                'max_concurrent_images': self.max_concurrent_images,
                'max_concurrent_videos': self.max_concurrent_videos,
                'total_processed': len(self.processed_hashes)
            }
            
            state_file = self.output_folder / "processor_state.json"
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save processor state: {e}")
    
    def _save_processing_history(self, file_path: Path, file_type: str, 
                               processing_time: float, success: bool):
        """Save processing history for charts"""
        try:
            history_file = self.output_folder / "processing_history.json"
            
            # Load existing history
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            else:
                history = {'entries': []}
            
            # Add new entry
            entry = {
                'timestamp': datetime.now().isoformat(),
                'file_name': file_path.name,
                'file_type': file_type,
                'processing_time': processing_time,
                'success': success,
                'file_size_mb': file_path.stat().st_size / (1024 * 1024)
            }
            
            history['entries'].append(entry)
            
            # Keep only last 1000 entries
            if len(history['entries']) > 1000:
                history['entries'] = history['entries'][-1000:]
            
            # Save updated history
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save processing history: {e}")
    
    async def _state_monitor(self):
        """Monitor and save processor state for web dashboard"""
        while self._running:
            try:
                self._save_processor_state()
                await asyncio.sleep(5)  # Update every 5 seconds for real-time dashboard
            except Exception as e:
                logger.error(f"State monitor error: {e}")
                await asyncio.sleep(10)
    
    def stop(self):
        """Zatrzymaj processor"""
        logger.info("Stopping continuous processor...")
        self._running = False
        
        # Cancel active tasks
        for task in self.active_tasks.values():
            if not task.done():
                task.cancel()
        
        # Save state
        self._save_processed_hashes()
        
        # Shutdown executor
        self.io_executor.shutdown(wait=True) 