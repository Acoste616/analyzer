#!/usr/bin/env python3
"""Test OCR quality on sample files"""

import asyncio
from pathlib import Path
from PIL import Image

async def test_ocr_quality():
    from jarvis_edu.extractors.enhanced_image_extractor import EnhancedImageExtractor
    
    extractor = EnhancedImageExtractor()
    media_folder = Path("C:/Users/barto/OneDrive/Pulpit/mediapobrane/pobrane_media")
    
    # Test pierwszych 5 obrazów
    image_files = list(media_folder.glob("*.jpg"))[:5]
    
    for i, image_path in enumerate(image_files):
        print(f"\n{'='*60}")
        print(f"Testing {i+1}/{len(image_files)}: {image_path.name}")
        
        # Sprawdź rozmiar i właściwości obrazu
        img = Image.open(image_path)
        print(f"Image size: {img.size}")
        print(f"Image mode: {img.mode}")
        
        # Ekstraktuj tekst
        result = await extractor.extract(str(image_path))
        
        print(f"Extracted text length: {len(result.content)}")
        print(f"Confidence: {result.confidence}")
        
        if len(result.content) > 50:
            print(f"Text preview: {result.content[:200]}...")
        else:
            print(f"Full text: {result.content}")
        
        # Analiza metadanych
        metadata = result.metadata.get('content_analysis', {})
        print(f"Has text regions: {metadata.get('text_regions', 0)}")
        print(f"Content type: {metadata.get('content_type', 'unknown')}")

if __name__ == "__main__":
    asyncio.run(test_ocr_quality()) 