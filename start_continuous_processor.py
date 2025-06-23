#!/usr/bin/env python3
"""
CONTINUOUS PROCESSOR LAUNCHER
Uruchamia ciągłe monitorowanie i przetwarzanie plików multimedialnych.
"""

import asyncio
import signal
import sys
from pathlib import Path

# Dodaj ścieżkę do modułów
sys.path.append(str(Path(__file__).parent))

from jarvis_edu.pipeline.continuous_processor import ContinuousProcessor
from jarvis_edu.core.logger import get_logger

logger = get_logger(__name__)

# Konfiguracja
WATCH_FOLDER = "C:/Users/barto/OneDrive/Pulpit/mediapobrane/pobrane_media"
OUTPUT_FOLDER = "output"

class ContinuousProcessorLauncher:
    def __init__(self):
        self.processor = None
        self.running = False
    
    async def start(self):
        """Uruchom continuous processor"""
        try:
            print("🚀 STARTING CONTINUOUS PROCESSOR")
            print("="*60)
            print(f"📁 Watch folder: {WATCH_FOLDER}")
            print(f"📂 Output folder: {OUTPUT_FOLDER}")
            print("="*60)
            
            # Sprawdź czy folder istnieje
            watch_path = Path(WATCH_FOLDER)
            if not watch_path.exists():
                print(f"❌ Watch folder does not exist: {WATCH_FOLDER}")
                return
            
            # Utwórz processor
            self.processor = ContinuousProcessor(
                watch_folder=WATCH_FOLDER,
                output_folder=OUTPUT_FOLDER
            )
            
            # Ustaw handler dla graceful shutdown
            self.setup_signal_handlers()
            
            print("✅ Continuous processor initialized")
            print("🔄 Starting monitoring...")
            print("\n🛑 Press Ctrl+C to stop\n")
            
            self.running = True
            
            # Uruchom processor
            await self.processor.start()
            
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested by user")
        except Exception as e:
            logger.error(f"Continuous processor error: {e}")
            print(f"❌ Error: {e}")
        finally:
            await self.shutdown()
    
    def setup_signal_handlers(self):
        """Ustaw obsługę sygnałów dla graceful shutdown"""
        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}")
            self.running = False
            if self.processor:
                self.processor.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def shutdown(self):
        """Graceful shutdown"""
        print("\n🔄 Shutting down continuous processor...")
        
        if self.processor:
            self.processor.stop()
            print("✅ Processor stopped")
        
        print("👋 Continuous processor shut down complete")

async def main():
    """Główna funkcja"""
    launcher = ContinuousProcessorLauncher()
    await launcher.start()

if __name__ == "__main__":
    print("🤖 JARVIS EDU CONTINUOUS PROCESSOR")
    print("Continuous monitoring and processing of multimedia files")
    print(f"Python: {sys.version}")
    print("-" * 60)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1) 