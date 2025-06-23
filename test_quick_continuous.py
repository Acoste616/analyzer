#!/usr/bin/env python3
"""
🚀 Quick Test for Continuous Processor
Simple test script to verify continuous processor functionality.
"""

import asyncio
import sys
from pathlib import Path
import time
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from jarvis_edu.pipeline.continuous_processor import ContinuousProcessor
from jarvis_edu.core.logger import setup_logging, LogConfig

# Setup logging
setup_logging(LogConfig(level="INFO", format="human"))

async def quick_test():
    """Quick test of continuous processor"""
    
    print("\n" + "="*50)
    print("🚀 QUICK CONTINUOUS PROCESSOR TEST")
    print("="*50)
    
    # Setup test directories
    test_dir = Path("quick_test")
    watch_dir = test_dir / "watch"
    output_dir = test_dir / "output"
    
    # Create directories
    test_dir.mkdir(exist_ok=True)
    watch_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Create a simple test file
    test_file = watch_dir / "test_document.txt"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("""
QUICK TEST DOCUMENT
==================

This is a test document for the Jarvis EDU continuous processor.

Mathematics:
- Algebra: ax + b = 0
- Calculus: dy/dx = f'(x)

Programming:
def test_function():
    return "Hello, Jarvis!"

Science:
The formula for kinetic energy is KE = 1/2 * m * v²

Created: {datetime.now().isoformat()}
        """)
    
    print(f"📁 Test setup:")
    print(f"   Watch: {watch_dir}")
    print(f"   Output: {output_dir}")
    print(f"   Test file: {test_file.name} ({test_file.stat().st_size} bytes)")
    
    # Create processor
    processor = ContinuousProcessor(
        watch_folder=str(watch_dir),
        output_folder=str(output_dir)
    )
    
    print(f"\n⚙️ Processor config:")
    print(f"   Scan interval: {processor.scan_interval}s")
    
    # Run test for 30 seconds
    print(f"\n🎯 Running test for 30 seconds...")
    print("Press Ctrl+C to stop early\n")
    
    start_time = time.time()
    
    try:
        # Start processor
        processor_task = asyncio.create_task(processor.start())
        
        # Wait for processing
        for i in range(30):
            await asyncio.sleep(1)
            if (i + 1) % 10 == 0:
                processed = len(processor.processed_hashes)
                print(f"⏳ {i+1}s - Processed: {processed} files")
        
        # Stop processor
        processor.stop()
        await asyncio.sleep(1)
        
    except KeyboardInterrupt:
        print("\n🛑 Test stopped by user")
        processor.stop()
    
    # Show results
    duration = time.time() - start_time
    processed = len(processor.processed_hashes)
    active_tasks = len(processor.active_tasks)
    
    print(f"\n📊 Results:")
    print(f"   Duration: {duration:.1f}s")
    print(f"   Files processed: {processed}")
    print(f"   Active tasks: {active_tasks}")
    
    # Check output
    content_files = list(output_dir.glob("content/*.txt"))
    knowledge_files = list(output_dir.glob("knowledge_base_*.json"))
    
    print(f"   Content files: {len(content_files)}")
    print(f"   Knowledge files: {len(knowledge_files)}")
    
    # Show sample output
    if content_files:
        print(f"\n📄 Sample content from {content_files[0].name}:")
        with open(content_files[0], 'r', encoding='utf-8') as f:
            content = f.read()[:200]
        print(f"   {content}...")
    
    # Cleanup option
    cleanup = input(f"\n🧹 Remove test directory? (y/n): ")
    if cleanup.lower() == 'y':
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)
        print("✅ Test directory cleaned up")
    else:
        print(f"📁 Test files preserved in: {test_dir}")

if __name__ == "__main__":
    asyncio.run(quick_test()) 