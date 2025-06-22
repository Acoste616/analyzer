#!/usr/bin/env python3
"""Integration test for fixed production processor"""

import asyncio
from pathlib import Path

async def test_fixed_processor():
    from real_production_processor import RealProductionProcessor
    
    processor = RealProductionProcessor()
    
    # Test with single file
    test_files = list(processor.media_folder.glob("*.jpg"))[:1]
    
    if test_files:
        test_file = test_files[0]
        print(f"Testing with: {test_file.name}")
        
        # Test extraction
        extracted = await processor.extract_real_content(test_file)
        print(f"Extracted content length: {len(extracted['content']) if extracted else 0}")
        
        # Test analysis
        if extracted:
            analysis = processor.analyze_extracted_content(extracted, test_file)
            print(f"Category: {analysis['primary_category']} → {analysis['primary_subcategory']}")
            print(f"Confidence: {analysis['confidence']}%")
            
            # Test AI
            ai_response = await processor.try_ai_enhancement(extracted['content'][:500], analysis)
            print(f"AI Response: {ai_response}")
        
        return True
    
    return False

if __name__ == "__main__":
    success = asyncio.run(test_fixed_processor())
    print(f"\nTest {'PASSED ✅' if success else 'FAILED ❌'}") 