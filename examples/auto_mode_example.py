#!/usr/bin/env python3
"""
Example script showing how to use Jarvis EDU Auto-Mode.
"""

import asyncio
from pathlib import Path
from jarvis_edu.auto_mode import AutoModeProcessor
from jarvis_edu.core.auto_config import load_auto_config


async def run_auto_mode_example():
    """Example of running auto-mode programmatically."""
    
    print("🚀 Jarvis EDU Auto-Mode Example")
    print("=" * 40)
    
    try:
        # Load configuration
        config_path = "config/auto-mode.yaml"
        if not Path(config_path).exists():
            print(f"❌ Configuration file not found: {config_path}")
            print("💡 Run 'jarvis-edu init' to create sample configuration")
            return
        
        config = load_auto_config(config_path)
        
        print(f"📁 Watch folder: {config.system.watch_folder}")
        print(f"⚙️  LLM Provider: {config.llm.provider}")
        print(f"📤 Output directory: {config.output.base_dir}")
        print(f"🔄 Scan interval: {config.system.scan_interval} seconds")
        print()
        
        # Create processor
        processor = AutoModeProcessor()
        
        print("Starting auto-mode processor...")
        print("Press Ctrl+C to stop")
        print()
        
        # Start processing
        processor.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(run_auto_mode_example()) 