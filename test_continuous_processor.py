#!/usr/bin/env python3
"""
🧪 Comprehensive Test Suite for Continuous Processor
Tests the enhanced continuous processor with various file types and scenarios.
"""

import asyncio
import sys
import json
import os
import time
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import shutil

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from jarvis_edu.pipeline.continuous_processor import ContinuousProcessor
from jarvis_edu.core.logger import setup_logging, LogConfig
from jarvis_edu.core.config_loader import ConfigLoader

# Setup logging
setup_logging(LogConfig(level="DEBUG", format="human"))

class TestFilesCreator:
    """Creates test files for various scenarios"""
    
    def __init__(self, test_dir: Path):
        self.test_dir = test_dir
        self.test_dir.mkdir(exist_ok=True)
    
    def create_test_image(self, name: str, content: str = "Test Image Content") -> Path:
        """Create a test image file with OCR-readable text"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Create a simple image with text
            img = Image.new('RGB', (800, 600), color='white')
            draw = ImageDraw.Draw(img)
            
            # Try to use a font, fallback to default
            try:
                font = ImageFont.truetype("arial.ttf", 40)
            except:
                font = ImageFont.load_default()
            
            # Add educational content
            text_lines = [
                "EDUCATIONAL CONTENT",
                "Mathematics: f(x) = 2x + 3",
                "Science: E = mc²",
                "Programming: def hello():",
                "    print('Hello World')",
                content
            ]
            
            y_pos = 50
            for line in text_lines:
                draw.text((50, y_pos), line, fill='black', font=font)
                y_pos += 60
            
            file_path = self.test_dir / name
            img.save(file_path)
            return file_path
        except ImportError:
            # Fallback: create a text file instead
            print(f"⚠️ PIL not available, creating text file instead of image")
            return self.create_test_text_file(name.replace('.png', '.txt'), content)
    
    def create_test_text_file(self, name: str, content: str = "Test document content") -> Path:
        """Create a test text file"""
        file_path = self.test_dir / name
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"""
EDUCATIONAL DOCUMENT
====================

Subject: {name.split('.')[0].replace('_', ' ').title()}

Content:
{content}

This is a test document for the Jarvis EDU system.
It contains educational content for analysis.

Mathematics Example:
- Linear function: y = ax + b
- Quadratic function: y = ax² + bx + c

Programming Example:
```python
def calculate_sum(a, b):
    return a + b
```

Science Example:
The speed of light is approximately 299,792,458 m/s.

Created: {datetime.now().isoformat()}
            """)
        return file_path
    
    def create_test_video_placeholder(self, name: str) -> Path:
        """Create a placeholder for video (since we can't easily create real video)"""
        file_path = self.test_dir / name
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"""
PLACEHOLDER VIDEO FILE: {name}
===============================

This would be a video file containing:
- Educational lecture about: {name.split('.')[0].replace('_', ' ').title()}
- Duration: 15 minutes
- Language: Polish/English
- Quality: 1080p

Content Summary:
- Introduction to the topic
- Key concepts explanation
- Practical examples
- Summary and conclusions

Audio Transcript:
"Welcome to today's lesson on {name.split('.')[0].replace('_', ' ')}. 
In this video, we will explore the fundamental concepts..."

Created: {datetime.now().isoformat()}
            """)
        return file_path

class ContinuousProcessorTester:
    """Comprehensive tester for continuous processor"""
    
    def __init__(self):
        self.test_base_dir = Path("test_continuous_processor")
        self.watch_dir = self.test_base_dir / "watch"
        self.output_dir = self.test_base_dir / "output"
        self.test_files: List[Path] = []
        self.processor: ContinuousProcessor = None
        self.results: Dict[str, Any] = {}
        
    def setup_test_environment(self):
        """Setup test directories and files"""
        print("\n" + "="*70)
        print("🚀 SETTING UP TEST ENVIRONMENT")
        print("="*70)
        
        # Create directories
        self.test_base_dir.mkdir(exist_ok=True)
        self.watch_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"📁 Created test directories:")
        print(f"   Watch: {self.watch_dir}")
        print(f"   Output: {self.output_dir}")
        
        # Create test files
        file_creator = TestFilesCreator(self.watch_dir)
        
        # Create various test files
        test_files_config = [
            ("mathematics_lesson.png", "Algebra equations and solving methods"),
            ("science_physics.png", "Laws of motion and energy conservation"),
            ("programming_tutorial.png", "Python programming basics"),
            ("history_document.txt", "Ancient civilizations and their impact"),
            ("chemistry_notes.txt", "Periodic table and chemical reactions"),
            ("language_lesson.txt", "English grammar and vocabulary"),
            ("math_video.mp4", "Advanced calculus concepts"),
            ("science_video.mp4", "Biology cell structure"),
            ("programming_video.mp4", "Object-oriented programming"),
        ]
        
        print(f"\n📝 Creating test files:")
        for filename, content in test_files_config:
            if filename.endswith('.png'):
                file_path = file_creator.create_test_image(filename, content)
            elif filename.endswith('.txt'):
                file_path = file_creator.create_test_text_file(filename, content)
            elif filename.endswith('.mp4'):
                file_path = file_creator.create_test_video_placeholder(filename)
            
            self.test_files.append(file_path)
            print(f"   ✅ {filename} ({file_path.stat().st_size} bytes)")
        
        print(f"\n📊 Total test files created: {len(self.test_files)}")
    
    def create_processor(self) -> ContinuousProcessor:
        """Create and configure the continuous processor"""
        print("\n" + "="*70)
        print("⚙️  CONFIGURING CONTINUOUS PROCESSOR")
        print("="*70)
        
        # Create processor with basic configuration
        processor = ContinuousProcessor(
            watch_folder=str(self.watch_dir),
            output_folder=str(self.output_dir)
        )
        
        print(f"📁 Watch folder: {self.watch_dir}")
        print(f"📤 Output folder: {self.output_dir}")
        print(f"⏱️  Scan interval: {processor.scan_interval}s")
        
        return processor
    
    async def run_processing_test(self, duration: int = 60):
        """Run the continuous processor for a specified duration"""
        print("\n" + "="*70)
        print(f"🎯 RUNNING CONTINUOUS PROCESSOR TEST ({duration}s)")
        print("="*70)
        
        self.processor = self.create_processor()
        
        start_time = time.time()
        print(f"⏰ Test started at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"🎯 Test duration: {duration} seconds")
        print(f"📊 Files to process: {len(self.test_files)}")
        print("\nPress Ctrl+C to stop early\n")
        
        try:
            # Start processor in background
            processor_task = asyncio.create_task(self.processor.start())
            
            # Monitor progress
            for i in range(duration):
                await asyncio.sleep(1)
                
                # Show progress every 10 seconds
                if (i + 1) % 10 == 0:
                    processed = len(self.processor.processed_hashes)
                    errors = len(self.processor.error_files)
                    print(f"⏳ {i+1}s - Processed: {processed}, Errors: {errors}")
            
            # Stop processor
            self.processor.stop()
            await asyncio.sleep(2)  # Give time for cleanup
            
        except KeyboardInterrupt:
            print("\n🛑 Test stopped by user")
            self.processor.stop()
        
        end_time = time.time()
        test_duration = end_time - start_time
        
        print(f"\n⏰ Test completed in {test_duration:.1f} seconds")
        
        # Collect results
        self.results = {
            'test_duration': test_duration,
            'files_processed': len(self.processor.processed_hashes),
            'files_with_errors': len(self.processor.error_files),
            'total_files': len(self.test_files),
            'processing_rate': len(self.processor.processed_hashes) / (test_duration / 60) if test_duration > 0 else 0
        }
    
    def analyze_results(self):
        """Analyze and display test results"""
        print("\n" + "="*70)
        print("📊 TEST RESULTS ANALYSIS")
        print("="*70)
        
        # Basic statistics
        print(f"⏱️  Test Duration: {self.results['test_duration']:.1f} seconds")
        print(f"📁 Total Files: {self.results['total_files']}")
        print(f"✅ Files Processed: {self.results['files_processed']}")
        print(f"❌ Files with Errors: {self.results['files_with_errors']}")
        print(f"📊 Processing Rate: {self.results['processing_rate']:.1f} files/minute")
        
        # Success rate
        if self.results['total_files'] > 0:
            success_rate = (self.results['files_processed'] / self.results['total_files']) * 100
            print(f"🎯 Success Rate: {success_rate:.1f}%")
        
        # Check output files
        output_files = {
            'content': list(self.output_dir.glob("content/*.txt")),
            'knowledge_base': list(self.output_dir.glob("knowledge_base_*.json")),
            'logs': list(self.output_dir.glob("logs/*.log")),
            'reports': list(self.output_dir.glob("reports/*.json")),
        }
        
        print(f"\n📂 Output Files Generated:")
        for category, files in output_files.items():
            print(f"   {category.title()}: {len(files)} files")
        
        # Show sample content
        if output_files['content']:
            print(f"\n📄 Sample Content (from {output_files['content'][0].name}):")
            try:
                with open(output_files['content'][0], 'r', encoding='utf-8') as f:
                    content = f.read()[:300]
                print(f"   {content}...")
            except Exception as e:
                print(f"   ❌ Error reading sample: {e}")
        
        # Performance analysis
        print(f"\n⚡ Performance Analysis:")
        if self.results['processing_rate'] > 5:
            print("   🚀 Excellent performance (>5 files/min)")
        elif self.results['processing_rate'] > 2:
            print("   ✅ Good performance (2-5 files/min)")
        elif self.results['processing_rate'] > 1:
            print("   ⚠️ Moderate performance (1-2 files/min)")
        else:
            print("   🐌 Slow performance (<1 file/min)")
        
        return self.results
    
    def save_test_report(self):
        """Save detailed test report"""
        report = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'test_duration': self.results['test_duration'],
                'total_files': self.results['total_files'],
            },
            'results': self.results,
            'processor_config': {
                'watch_folder': str(self.watch_dir),
                'output_folder': str(self.output_dir),
            },
            'test_files': [str(f) for f in self.test_files],
        }
        
        report_file = self.test_base_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Test report saved: {report_file}")
        return report_file
    
    def cleanup(self, ask_user: bool = True):
        """Clean up test files and directories"""
        if ask_user:
            response = input(f"\n🧹 Clean up test directory '{self.test_base_dir}'? (y/n): ")
            if response.lower() != 'y':
                print("Test files preserved for manual inspection")
                return
        
        try:
            shutil.rmtree(self.test_base_dir, ignore_errors=True)
            print("✅ Test directory cleaned up")
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")

async def main():
    """Main test execution"""
    print("🧪 JARVIS EDU CONTINUOUS PROCESSOR TEST SUITE")
    print("=" * 70)
    
    tester = ContinuousProcessorTester()
    
    try:
        # Setup test environment
        tester.setup_test_environment()
        
        # Ask user for test duration
        duration_input = input("\n⏱️ Test duration in seconds (default 60): ").strip()
        duration = int(duration_input) if duration_input.isdigit() else 60
        
        # Run the test
        await tester.run_processing_test(duration)
        
        # Analyze results
        tester.analyze_results()
        
        # Save report
        tester.save_test_report()
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        tester.cleanup(ask_user=True)

if __name__ == "__main__":
    asyncio.run(main()) 