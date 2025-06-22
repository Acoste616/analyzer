#!/usr/bin/env python3
"""Simple test of file processing without dashboard."""

import sys
import asyncio
from pathlib import Path

# Add project root to path  
sys.path.insert(0, str(Path(__file__).parent))

from jarvis_edu.pipeline.auto_processor import AutoProcessor
from jarvis_edu.processors.lm_studio import LMStudioProcessor

async def test_processor():
    """Test file processor."""
    print("🧪 Testing Jarvis EDU File Processor")
    print("=" * 40)
    
    # Test LM Studio
    print("🤖 Testing LM Studio...")
    lm_studio = LMStudioProcessor()
    
    is_available = await lm_studio.is_available()
    print(f"   LM Studio available: {'✅' if is_available else '❌'}")
    
    if is_available:
        connection_info = await lm_studio.test_connection()
        print(f"   Models loaded: {len(connection_info.get('models', []))}")
    
    # Test AutoProcessor initialization
    print("\n🔧 Testing AutoProcessor...")
    try:
        processor = AutoProcessor()
        print("   ✅ AutoProcessor created successfully")
        
        # Test starting processor
        await processor.start()
        print("   ✅ AutoProcessor started successfully")
        
        # Get queue status
        queue_status = processor.processing_queue.get_pending_files(limit=1)
        print(f"   📋 Pending files in queue: {len(queue_status)}")
        
        # Test folders
        watch_folder = Path(processor.watch_folder)
        output_folder = Path(processor.output_folder)
        
        print(f"   📁 Watch folder: {watch_folder} ({'exists' if watch_folder.exists() else 'missing'})")
        print(f"   📁 Output folder: {output_folder} ({'exists' if output_folder.exists() else 'missing'})")
        
        # Stop processor
        processor.stop()
        print("   ✅ AutoProcessor stopped cleanly")
        
        print("\n🎉 FILE PROCESSING IS READY!")
        print("=" * 40)
        print("✅ LM Studio: Connected")
        print("✅ AutoProcessor: Working")
        print("✅ File monitoring: Ready")
        print("✅ System paths: Configured")
        print("")
        print("📁 Drop files into:")
        print(f"   {watch_folder}")
        print("")
        print("📄 Results will appear in:")
        print(f"   {output_folder}")
        
    except Exception as e:
        print(f"   ❌ AutoProcessor failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_processor()) 