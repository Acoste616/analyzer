#!/usr/bin/env python3
"""
🚀 Simple Initialization Test for Continuous Processor
Tests step-by-step initialization with proper timing.
"""

import asyncio
import time
from pathlib import Path
from datetime import datetime

print("🧪 SIMPLE CONTINUOUS PROCESSOR TEST")
print("="*50)

def test_step(step_name: str, func):
    """Test individual step with timing"""
    print(f"\n{step_name}...")
    start = time.time()
    try:
        result = func()
        duration = time.time() - start
        print(f"✅ {step_name} completed in {duration:.1f}s")
        return result
    except Exception as e:
        duration = time.time() - start
        print(f"❌ {step_name} failed after {duration:.1f}s: {e}")
        return None

# Step 1: Test imports
def test_imports():
    from jarvis_edu.pipeline.continuous_processor import ContinuousProcessor
    from jarvis_edu.core.logger import setup_logging, LogConfig
    return True

# Step 2: Test processor creation
def test_processor_creation():
    from jarvis_edu.pipeline.continuous_processor import ContinuousProcessor
    processor = ContinuousProcessor("test_watch", "test_output")
    return processor

# Step 3: Test file creation
def test_file_creation():
    test_dir = Path("simple_test")
    test_dir.mkdir(exist_ok=True)
    
    test_file = test_dir / "sample.txt"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("Simple test content for Jarvis EDU processor.")
    
    return test_file

# Step 4: Test basic scan
def test_scan(processor):
    # Set processor to scan our test directory
    processor.watch_folder = Path("simple_test")
    # Test if processor can scan the folder
    files = list(processor.watch_folder.glob("*"))
    return len(files)

# Run tests
print(f"🕐 Starting at {datetime.now().strftime('%H:%M:%S')}")

# Step 1: Imports (should be fast)
if test_step("Step 1: Testing imports", test_imports):
    
    # Step 2: Processor creation (takes ~20-30s first time)
    print("\n⚠️ Note: Processor creation loads AI models - this may take 20-30 seconds...")
    processor = test_step("Step 2: Creating processor", test_processor_creation)
    
    if processor:
        # Step 3: File creation (should be fast)
        test_file = test_step("Step 3: Creating test file", test_file_creation)
        
        if test_file:
            # Step 4: Basic functionality test
            file_count = test_step("Step 4: Testing file scan", lambda: test_scan(processor))
            
            print(f"\n📊 RESULTS:")
            print(f"✅ All steps completed successfully!")
            print(f"📁 Test file created: {test_file}")
            print(f"🔍 Files found in scan: {file_count}")
            print(f"🕐 Total time: {time.time() - time.time():.1f}s")
            
            # Cleanup
            cleanup = input("\n🧹 Remove test files? (y/n): ")
            if cleanup.lower() == 'y':
                import shutil
                shutil.rmtree("simple_test", ignore_errors=True)
                shutil.rmtree("test_output", ignore_errors=True)
                print("✅ Cleaned up")
            else:
                print(f"📁 Test files preserved in: simple_test/")

print(f"\n🕐 Finished at {datetime.now().strftime('%H:%M:%S')}") 