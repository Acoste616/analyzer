# 🧪 Continuous Processor Testing Guide

## Overview

This directory contains comprehensive testing tools for the Jarvis EDU Continuous Processor system. The tests verify functionality, performance, and reliability of the enhanced continuous processor.

## Test Files

### 1. `test_continuous_processor.py` - Full Test Suite
**Comprehensive testing with multiple file types and scenarios**

**Features:**
- Creates test images, documents, and video placeholders
- Tests various file types (PNG, TXT, MP4)
- Performance monitoring and analysis
- Detailed reporting with JSON output
- Error handling and recovery testing
- Automatic cleanup options

**Usage:**
```bash
python test_continuous_processor.py
```

**Test Scenarios:**
- Mathematics lesson images with OCR content
- Science documents with formulas
- Programming tutorials and code examples
- Educational video placeholders
- Dynamic file addition during processing
- Error recovery and timeout handling

### 2. `test_quick_continuous.py` - Quick Test
**Simple 30-second test for basic functionality verification**

**Features:**
- Single test document
- Quick setup and execution
- Basic processor verification
- Simple output analysis

**Usage:**
```bash
python test_quick_continuous.py
```

## Prerequisites

### System Requirements
- Python 3.10+
- All Jarvis EDU dependencies installed
- PIL/Pillow for image creation (optional - fallback to text files)
- At least 100MB free disk space for test files

### Installation
```bash
# Install dependencies
pip install -e .

# Optional: Install PIL for image testing
pip install Pillow
```

## Test Execution

### Quick Test (Recommended for first-time users)
```bash
# Run the quick test
python test_quick_continuous.py

# Expected output:
# - Test setup information
# - 30-second processing run
# - Results summary
# - Cleanup option
```

### Full Test Suite
```bash
# Run comprehensive tests
python test_continuous_processor.py

# Interactive prompts:
# - Test duration (default: 60 seconds)
# - Cleanup preference (y/n)

# Expected output:
# - Test environment setup
# - File creation summary
# - Processing progress updates
# - Detailed results analysis
# - Performance recommendations
# - JSON test report
```

## Test Output Structure

### Directory Structure
```
test_continuous_processor/
├── watch/                    # Input files
│   ├── mathematics_lesson.png
│   ├── science_physics.png
│   ├── programming_tutorial.png
│   ├── history_document.txt
│   ├── chemistry_notes.txt
│   ├── language_lesson.txt
│   ├── math_video.mp4
│   ├── science_video.mp4
│   └── programming_video.mp4
└── output/                   # Generated content
    ├── content/              # Extracted text content
    ├── knowledge_base_*.json # Structured knowledge
    ├── logs/                 # Processing logs
    └── reports/              # Analysis reports
```

### Test Report Format
```json
{
  "test_info": {
    "timestamp": "2024-12-22T10:30:00",
    "test_duration": 60.5,
    "total_files": 9
  },
  "results": {
    "files_processed": 8,
    "files_with_errors": 1,
    "processing_rate": 7.9
  },
  "processor_config": {
    "watch_folder": "test_continuous_processor/watch",
    "output_folder": "test_continuous_processor/output"
  },
  "test_files": ["..."]
}
```

## Performance Benchmarks

### Expected Performance
- **Excellent**: >5 files/minute
- **Good**: 2-5 files/minute  
- **Moderate**: 1-2 files/minute
- **Slow**: <1 file/minute

### Success Rate Targets
- **Excellent**: >90% success rate
- **Good**: 80-90% success rate
- **Acceptable**: 70-80% success rate
- **Needs Investigation**: <70% success rate

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'jarvis_edu'
# Solution: Install project in development mode
pip install -e .
```

#### 2. PIL/Pillow Missing
```bash
# Warning: PIL not available, creating text file instead of image
# Solution: Install Pillow (optional)
pip install Pillow
```

#### 3. Permission Errors
```bash
# Error: PermissionError: Access denied
# Solution: Run with proper permissions or change test directory
# Windows: Run as Administrator
# Linux/Mac: Check directory permissions
```

#### 4. Long Video Processing Timeouts
```bash
# Expected behavior: Videos >2 hours use sampling
# Check logs for: "Using intelligent sampling for long video"
```

### Log Analysis

#### Normal Processing Logs
```
INFO - 🔄 Processing file: mathematics_lesson.png
INFO - 📤 Running advanced OCR for mathematics_lesson.png
INFO - ✅ Successfully processed: mathematics_lesson.png
```

#### Error Logs
```
ERROR - ❌ Processing failed: science_video.mp4: Timeout
ERROR - 🔄 Retrying with reduced timeout: science_video.mp4
```

## Test Customization

### Custom Test Duration
```python
# In test_continuous_processor.py
# Modify the duration input or set directly:
duration = 120  # 2 minutes
```

### Custom File Types
```python
# Add custom test files to test_files_config:
test_files_config = [
    ("custom_document.pdf", "Custom PDF content"),
    ("custom_audio.mp3", "Audio transcription test"),
    # ... existing files
]
```

### Custom Processor Settings
```python
# Modify processor settings:
processor.scan_interval = 5  # Scan every 5 seconds
processor.max_concurrent_images = 3
processor.max_concurrent_videos = 2
```

## Integration with CI/CD

### GitHub Actions Example
```yaml
name: Test Continuous Processor
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -e .
      - name: Run quick test
        run: python test_quick_continuous.py
      - name: Run full test suite
        run: python test_continuous_processor.py
```

## Best Practices

### Before Testing
1. **Clean Environment**: Remove previous test directories
2. **Check Dependencies**: Ensure all packages are installed
3. **Free Resources**: Close other heavy applications
4. **Check Disk Space**: Ensure sufficient storage

### During Testing
1. **Monitor Resources**: Watch CPU/Memory usage
2. **Check Logs**: Review processing logs for errors
3. **Network Stability**: Ensure stable connection for LM Studio
4. **Don't Interrupt**: Let tests complete naturally

### After Testing
1. **Review Reports**: Analyze JSON test reports
2. **Check Output Quality**: Verify extracted content
3. **Performance Analysis**: Compare with benchmarks
4. **Error Investigation**: Analyze failed files
5. **Cleanup**: Remove test directories if needed

## Advanced Testing

### Load Testing
```python
# Create many files for stress testing
for i in range(100):
    create_test_file(f"load_test_{i}.txt")
```

### Memory Testing
```python
# Monitor memory usage during processing
import psutil
process = psutil.Process()
memory_usage = process.memory_info().rss / 1024 / 1024  # MB
```

### Concurrent Testing
```python
# Run multiple processors simultaneously
processors = [
    ContinuousProcessor(watch_folder=f"test_{i}") 
    for i in range(3)
]
```

## Support

For issues with testing:
1. Check the troubleshooting section above
2. Review the main project documentation
3. Check processor logs in `logs/` directory
4. Verify system requirements and dependencies
5. Test with the quick test first before full suite

## Contributing

To improve the test suite:
1. Add new test scenarios to `test_files_config`
2. Implement additional file type creators
3. Add performance benchmarks
4. Improve error detection and reporting
5. Add integration tests with real media files 