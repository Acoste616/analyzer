#!/usr/bin/env python3
"""
Auto-mode runner for Jarvis EDU Extractor.
"""

import time
from pathlib import Path
from loguru import logger


class AutoModeProcessor:
    """Auto-mode processor."""
    
    def __init__(self):
        """Initialize processor."""
        self.running = False
    
    def start(self):
        """Start auto-mode."""
        logger.info("Starting auto-mode...")
        self.running = True
        
        while self.running:
            time.sleep(60)
            logger.info("Auto-mode running...")
    
    def stop(self):
        """Stop auto-mode."""
        self.running = False
        logger.info("Auto-mode stopped")


if __name__ == "__main__":
    processor = AutoModeProcessor()
    try:
        processor.start()
    except KeyboardInterrupt:
        processor.stop() 