#!/usr/bin/env python3
"""
Jarvis EDU Continuous Processing System
Auto-recovery, monitoring, and web interface
"""

import sys
import asyncio
import signal
import logging
from pathlib import Path
from datetime import datetime
import psutil
import torch
from typing import Optional

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

from jarvis_edu.pipeline.continuous_processor import ContinuousProcessor
from jarvis_edu.api.dashboard import create_dashboard_app
from jarvis_edu.core.logger import setup_logging, LogConfig

# Configure logging
setup_logging(LogConfig(
    level="INFO",
    format="human",
    file="logs/jarvis_continuous.log"
))

logger = logging.getLogger(__name__)

class JarvisRunner:
    """Main runner with auto-recovery and monitoring"""
    
    def __init__(self):
        self.processor: Optional[ContinuousProcessor] = None
        self.dashboard_task: Optional[asyncio.Task] = None
        self.monitor_task: Optional[asyncio.Task] = None
        self.running = False
        self.restart_count = 0
        self.max_restarts = 5
        
    async def start(self):
        """Start all components with monitoring"""
        self.running = True
        
        print("\n" + "="*60)
        print("🧠 JARVIS EDU CONTINUOUS PROCESSING SYSTEM")
        print("="*60)
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Watch folder: C:/Users/barto/OneDrive/Pulpit/mediapobrane/pobrane_media")
        print(f"🖥️ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
        print("="*60 + "\n")
        
        # Start components
        tasks = [
            self._run_processor(),
            self._run_dashboard(),
            self._run_system_monitor()
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            await self.cleanup()
    
    async def _run_processor(self):
        """Run processor with auto-restart"""
        while self.running and self.restart_count < self.max_restarts:
            try:
                logger.info("Starting continuous processor...")
                
                self.processor = ContinuousProcessor(
                    watch_folder="C:/Users/barto/OneDrive/Pulpit/mediapobrane/pobrane_media",
                    output_folder="output"
                )
                
                await self.processor.start()
                
            except Exception as e:
                logger.error(f"Processor crashed: {e}")
                self.restart_count += 1
                
                if self.restart_count < self.max_restarts:
                    wait_time = min(60 * self.restart_count, 300)  # Max 5 min
                    logger.info(f"Restarting in {wait_time}s (attempt {self.restart_count}/{self.max_restarts})")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Max restarts reached - processor stopped")
                    self.running = False
    
    async def _run_dashboard(self):
        """Run web dashboard"""
        try:
            import uvicorn
            from jarvis_edu.api.dashboard import create_dashboard_app
            
            app = create_dashboard_app()
            
            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=8000,
                log_level="warning"
            )
            
            server = uvicorn.Server(config)
            
            logger.info("Dashboard available at http://localhost:8000")
            await server.serve()
            
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
    
    async def _run_system_monitor(self):
        """Monitor system health"""
        while self.running:
            try:
                # CPU & Memory
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # GPU
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_reserved() / torch.cuda.get_device_properties(0).total_memory
                    gpu_temp = "N/A"  # Would need nvidia-ml-py for temperature
                else:
                    gpu_memory = 0
                    gpu_temp = "N/A"
                
                # Disk space
                disk = psutil.disk_usage('/')
                
                # Log status
                status = (
                    f"System: CPU {cpu_percent:.1f}% | "
                    f"RAM {memory.percent:.1f}% | "
                    f"GPU {gpu_memory:.1%} | "
                    f"Disk {disk.percent:.1f}%"
                )
                
                if cpu_percent > 90 or memory.percent > 90:
                    logger.warning(f"High resource usage: {status}")
                else:
                    logger.debug(status)
                
                # Check if processor is healthy
                if self.processor and hasattr(self.processor, 'active_tasks'):
                    active = len(self.processor.active_tasks)
                    queued = self.processor.processing_tasks.qsize()
                    logger.info(f"Processing: {active} active, {queued} queued")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(60)
    
    async def cleanup(self):
        """Clean shutdown"""
        self.running = False
        
        if self.processor:
            self.processor.stop()
        
        # Cancel tasks
        for task in [self.dashboard_task, self.monitor_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Shutdown complete")

async def main():
    """Main entry point"""
    runner = JarvisRunner()
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        runner.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start system
    await runner.start()

if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 10):
        print("Python 3.10+ required")
        sys.exit(1)
    
    # Run
    asyncio.run(main()) 