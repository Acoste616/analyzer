# -*- coding: utf-8 -*-
"""REAL PRODUCTION PROCESSOR - Integruje działające ekstraktory Jarvis z systemem tagowania"""

import asyncio
import time
import json
from pathlib import Path
from datetime import datetime
import requests
import re
from typing import Dict, Any, List, Optional

print("🔧 REAL PRODUCTION PROCESSOR - Using Jarvis Extractors")

# Jarvis imports
try:
    from jarvis_edu.extractors.enhanced_image_extractor import EnhancedImageExtractor
    from jarvis_edu.extractors.enhanced_video_extractor import EnhancedVideoExtractor
    from jarvis_edu.extractors.enhanced_audio_extractor import EnhancedAudioExtractor
    JARVIS_AVAILABLE = True
    print("✅ Jarvis extractors imported successfully")
except ImportError as e:
    print(f"❌ Jarvis import error: {e}")
    JARVIS_AVAILABLE = False


class RealProductionProcessor:
    """Production processor using actual Jarvis content extraction"""
    
    def __init__(self):
        self.media_folder = Path("C:/Users/barto/OneDrive/Pulpit/mediapobrane/pobrane_media")
        self.output_folder = Path("output")
        self.output_folder.mkdir(exist_ok=True)
        
        # Initialize Jarvis extractors
        if JARVIS_AVAILABLE:
            self.image_extractor = EnhancedImageExtractor()
            self.video_extractor = EnhancedVideoExtractor()
            self.audio_extractor = EnhancedAudioExtractor()
            print("✅ Jarvis extractors initialized")
        else:
            print("❌ Jarvis not available")
            return
        
        # Enhanced tagging system for REAL content
        self.tag_system = {
            "AI": {
                "Claude": ["claude", "anthropic", "claude ai", "claude-3"],
                "GPT": ["gpt", "chatgpt", "openai", "gpt-4", "chat"],
                "Tools": ["cursor", "copilot", "midjourney", "ai tools", "artificial", "intelligence"]
            },
            "Development": {
                "Cursor": ["cursor ide", "cursor editor", "ai coding", "code"],
                "Programming": ["python", "javascript", "programming", "kod", "funkcja", "class", "def", "code", "script", "function", "variable"],
                "Web": ["html", "css", "react", "vue", "angular", "web", "website", "frontend", "backend"]
            },
            "Business": {
                "Marketing": ["marketing", "seo", "reklama", "social media", "brand", "marka"],
                "Krypto": ["bitcoin", "ethereum", "crypto", "blockchain", "defi", "btc", "eth"],
                "Investing": ["inwestycje", "stocks", "portfolio", "trading", "giełda", "akcje"]
            },
            "Content": {
                "Memes": ["meme", "humor", "funny", "śmieszne", "lol", "xd"],
                "Social": ["tiktok", "instagram", "youtube", "twitter", "facebook", "social"],
                "Gaming": ["gaming", "games", "gry", "twitch", "steam", "playstation", "xbox"]
            },
            "Education": {
                "Math": ["matematyka", "równanie", "funkcja", "algebra", "geometria", "liczba", "obliczenia"],
                "Science": ["fizyka", "chemia", "biologia", "nauka", "eksperyment", "atom", "cząsteczka"],
                "Tutorials": ["tutorial", "poradnik", "jak", "kurs", "lekcja", "how to", "guide"]
            },
            "General": {
                "Text": ["text", "tekst", "słowa", "words", "zdanie", "sentence"],
                "Image": ["obraz", "zdjęcie", "photo", "picture", "grafika", "image"],
                "Document": ["dokument", "document", "plik", "file", "pdf", "doc"]
            }
        }
        
        self.results = []
        
        # Batch processing settings
        self.batch_size = 5
        self.max_concurrent = 3
        
        # Cache for processed files
        self.processed_cache = set()
        self._load_processed_cache()

    def _load_processed_cache(self):
        """Load cache of already processed files"""
        cache_file = self.output_folder / "processed_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.processed_cache = set(json.load(f))
                print(f"📋 Loaded {len(self.processed_cache)} files from cache")
            except Exception as e:
                print(f"⚠️ Failed to load cache: {e}")
                self.processed_cache = set()

    def get_sample_files(self, max_files=20):
        """Get diverse sample of files"""
        print(f"📁 Scanning: {self.media_folder}")
        
        if not self.media_folder.exists():
            print("❌ Media folder not found!")
            return []
        
        # Find files by type with better distribution
        files_by_type = {
            'images': list(self.media_folder.rglob("*.jpg")) + list(self.media_folder.rglob("*.png")),
            'videos': list(self.media_folder.rglob("*.mp4")) + list(self.media_folder.rglob("*.mov")),
            'audio': list(self.media_folder.rglob("*.mp3")) + list(self.media_folder.rglob("*.wav"))
        }
        
        # Take balanced sample
        sample_files = []
        files_per_type = max_files // 3
        
        for file_type, files in files_by_type.items():
            if files:
                sample_files.extend(files[:files_per_type])
        
        print(f"📊 Selected {len(sample_files)} files for processing")
        for file_type, files in files_by_type.items():
            type_count = min(len(files), files_per_type)
            if type_count > 0:
                print(f"   📄 {file_type}: {type_count} files")
        
        return sample_files[:max_files]

    async def extract_real_content(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Extract REAL content using Jarvis extractors"""
        if not JARVIS_AVAILABLE:
            return None
            
        file_ext = file_path.suffix.lower()
        
        try:
            if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                print(f"   🖼️ Extracting image content...")
                extraction_result = await self.image_extractor.extract(str(file_path))
                return {
                    'content': extraction_result.content,
                    'confidence': extraction_result.confidence,
                    'metadata': extraction_result.metadata,
                    'type': 'image',
                    'title': extraction_result.title,
                    'summary': extraction_result.summary
                }
                
            elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
                print(f"   🎬 Extracting video content...")
                extraction_result = await self.video_extractor.extract(str(file_path))
                return {
                    'content': extraction_result.content,
                    'confidence': extraction_result.confidence,
                    'metadata': extraction_result.metadata,
                    'type': 'video',
                    'title': extraction_result.title,
                    'summary': extraction_result.summary
                }
                
            elif file_ext in ['.mp3', '.wav', '.m4a']:
                print(f"   🎵 Extracting audio content...")
                extraction_result = await self.audio_extractor.extract(str(file_path))
                return {
                    'content': extraction_result.content,
                    'confidence': extraction_result.confidence,
                    'metadata': extraction_result.metadata,
                    'type': 'audio',
                    'title': extraction_result.title,
                    'summary': extraction_result.summary
                }
                
            else:
                print(f"   ❓ Unsupported file type: {file_ext}")
                return None
                
        except Exception as e:
            print(f"   ❌ Extraction error: {e}")
            return None

    def analyze_extracted_content(self, extracted_data: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
        """Analyze REAL extracted content instead of filename"""
        
        if not extracted_data or not extracted_data.get('content'):
            print(f"   ⚠️ No content extracted from {file_path.name}")
            return {
                "primary_category": "No_Content",
                "primary_subcategory": "Empty",
                "detected_categories": {},
                "matched_keywords": [],
                "confidence": 0,
                "extraction_quality": "failed"
            }
        
        # Get actual extracted text content
        content = extracted_data['content'].lower()
        print(f"   📝 Analyzing {len(content)} characters of extracted content")
        
        detected_categories = {}
        all_keywords = []
        
        # Apply tag system to REAL content
        for main_cat, subcats in self.tag_system.items():
            cat_scores = {}
            for subcat, keywords in subcats.items():
                score = 0
                matched = []
                
                for keyword in keywords:
                    if keyword.lower() in content:
                        score += len(keyword.split()) * 2
                        matched.append(keyword)
                        all_keywords.append(keyword)
                
                if score > 0:
                    cat_scores[subcat] = {"score": score, "keywords": matched}
            
            if cat_scores:
                detected_categories[main_cat] = cat_scores
        
        # Find primary category
        primary_cat = primary_sub = None
        max_score = 0
        
        for main_cat, subcats in detected_categories.items():
            for subcat, data in subcats.items():
                if data["score"] > max_score:
                    max_score = data["score"]
                    primary_cat = main_cat
                    primary_sub = subcat
        
        # Calculate confidence based on content quality and matches
        base_confidence = extracted_data.get('confidence', 0.5)
        content_confidence = min(100, max_score * 8 + base_confidence * 30)
        
        return {
            "primary_category": primary_cat or "General",
            "primary_subcategory": primary_sub or "Other", 
            "detected_categories": detected_categories,
            "matched_keywords": all_keywords,
            "confidence": content_confidence,
            "extraction_quality": "success",
            "content_length": len(content),
            "content_type": extracted_data.get('type', 'unknown'),
            "original_title": extracted_data.get('title', ''),
            "original_summary": extracted_data.get('summary', '')
        }

    def create_polish_summary(self, file_path: Path, analysis: Dict[str, Any], extracted_data: Optional[Dict[str, Any]]) -> str:
        """Create Polish summary based on REAL content"""
        
        cat = analysis.get("primary_category", "General")
        sub = analysis.get("primary_subcategory", "Other")
        confidence = analysis.get("confidence", 0)
        keywords = analysis.get("matched_keywords", [])
        
        # Polish category mapping
        polish_map = {
            "AI": "Sztuczna Inteligencja",
            "Development": "Programowanie",
            "Business": "Biznes", 
            "Content": "Treści",
            "Education": "Edukacja",
            "Claude": "Claude AI",
            "GPT": "ChatGPT",
            "Cursor": "Edytor Cursor",
            "Programming": "Programowanie",
            "Math": "Matematyka",
            "Science": "Nauki ścisłe",
            "Tutorials": "Poradniki"
        }
        
        polish_cat = polish_map.get(cat, cat)
        polish_sub = polish_map.get(sub, sub)
        
        # Determine content type
        file_ext = file_path.suffix.lower()
        if file_ext in ['.jpg', '.png', '.gif']:
            content_type = "Obraz"
        elif file_ext in ['.mp4', '.mov', '.avi']:
            content_type = "Wideo"
        elif file_ext in ['.mp3', '.wav']:
            content_type = "Audio"
        else:
            content_type = "Plik"
        
        # Smart description based on actual content
        if analysis.get("extraction_quality") == "success" and keywords:
            keyword_text = f"Tematy: {', '.join(keywords[:4])}"
        elif extracted_data and extracted_data.get('content'):
            content_preview = extracted_data['content'][:100].replace('\n', ' ')
            keyword_text = f"Treść: {content_preview}..."
        else:
            keyword_text = "Brak wykrytej treści"
        
        return f"""📋 ANALIZA RZECZYWISTEJ TREŚCI:

🎯 KATEGORIA: {polish_cat} → {polish_sub}
📁 TYP: {content_type}
📊 PEWNOŚĆ: {confidence:.0f}%
📝 PLIK: {file_path.name}
🔍 {keyword_text}

📈 JAKOŚĆ EKSTRAKCJI: {analysis.get('extraction_quality', 'unknown')}"""

    async def try_ai_enhancement(self, content: str, analysis: Dict[str, Any]) -> Optional[str]:
        """Enhanced AI summary with anti-hallucination"""
        if not content or len(content.strip()) < 20:
            return None
        
        try:
            # Prostszy prompt dla szybszej odpowiedzi
            prompt = f"""Jedno zdanie po polsku o tym materiale:
Treść: {content[:200]}
Odpowiedz TYLKO jednym krótkim zdaniem."""
            
            payload = {
                "messages": [
                    {"role": "system", "content": "Odpowiadasz TYLKO jednym krótkim zdaniem po polsku."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 50,  # Zmniejszone z 100
                "temperature": 0.1,
                "stream": False
            }
            
            # Zwiększony timeout z 10 na 30 sekund
            response = requests.post(
                "http://127.0.0.1:1234/v1/chat/completions", 
                json=payload, 
                timeout=30  # Zwiększony timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                if "choices" in data and data["choices"]:
                    ai_text = data["choices"][0]["message"]["content"].strip()
                    if self._is_valid_ai_response(ai_text):
                        return f"🤖 AI: {ai_text}"
                        
        except requests.exceptions.Timeout:
            print("   ⚠️ AI timeout - LM Studio może być przeciążone")
        except Exception as e:
            print(f"   ⚠️ AI enhancement failed: {e}")
        
        return None

    def _is_valid_ai_response(self, text: str) -> bool:
        """Check if AI response is valid (not hallucinated)"""
        if not text or len(text) < 10:
            return False
        
        words = text.split()
        if len(set(words)) < len(words) * 0.5:  # Too many repeated words
            return False
        
        # Check for known hallucination patterns
        hallucination_patterns = [
            r'(\w+\s+){3,}\1{3,}',  # Repeated phrases
            r'zastosowania\s+nauczania\s+zastosowania',  # Known pattern
        ]
        
        for pattern in hallucination_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        
        return True

    async def process_single_file(self, file_path: Path):
        """Process a single file and add to results"""
        print(f"   📁 Processing: {file_path.name}")
        start_time = time.time()
        
        # STEP 1: Extract REAL content using Jarvis
        extracted_data = await self.extract_real_content(file_path)
        
        # STEP 2: Analyze extracted content (not filename!)
        analysis = self.analyze_extracted_content(extracted_data, file_path)
        
        # STEP 3: Create Polish summary
        polish_summary = self.create_polish_summary(file_path, analysis, extracted_data)
        
        # STEP 4: Try AI enhancement
        content_for_ai = extracted_data['content'] if extracted_data else ""
        ai_enhancement = await self.try_ai_enhancement(content_for_ai[:1000], analysis)
        
        processing_time = time.time() - start_time
        
        # Build result
        result = {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_type": file_path.suffix.lower().replace(".", ""),
            "file_size_mb": file_path.stat().st_size / (1024 * 1024),
            "processing_time": processing_time,
            "extracted_content": content_for_ai[:500] + "..." if len(content_for_ai) > 500 else content_for_ai,
            "ai_summary": polish_summary,
            "ai_enhancement": ai_enhancement,
            "tagging_analysis": analysis,
            "confidence": analysis["confidence"],
            "status": "success" if analysis["extraction_quality"] == "success" else "limited"
        }
        
        self.results.append(result)
        
        # Display progress
        print(f"      ✅ {analysis['primary_category']} → {analysis['primary_subcategory']} ({analysis['confidence']:.0f}%)")
        if analysis["matched_keywords"]:
            print(f"      🏷️ Keywords: {', '.join(analysis['matched_keywords'][:3])}")
        print(f"      ⏱️ Time: {processing_time:.2f}s")

    async def process_files_batch(self, files: List[Path]):
        """Process files in batches for better performance"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_with_limit(file_path):
            async with semaphore:
                if str(file_path) not in self.processed_cache:
                    await self.process_single_file(file_path)
                    self.processed_cache.add(str(file_path))
                else:
                    print(f"   ⏭️ Skipping cached: {file_path.name}")
        
        # Process files in batches
        for i in range(0, len(files), self.batch_size):
            batch = files[i:i + self.batch_size]
            print(f"\n🔄 Processing batch {i//self.batch_size + 1}: {len(batch)} files")
            
            tasks = [process_with_limit(f) for f in batch]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Save cache after each batch
            self._save_processed_cache()

    def _save_processed_cache(self):
        """Save processed files cache"""
        try:
            cache_file = self.output_folder / "processed_cache.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.processed_cache), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Failed to save cache: {e}")

    async def process_real_files(self):
        """Main processing function using REAL content extraction"""
        print("🔧 REAL PRODUCTION PROCESSOR - ACTUAL CONTENT EXTRACTION")
        print("="*70)
        
        if not JARVIS_AVAILABLE:
            print("❌ Cannot process - Jarvis extractors not available")
            return
        
        # Get sample files
        sample_files = self.get_sample_files(20)
        
        if not sample_files:
            print("❌ No files to process!")
            return
        
        print(f"🚀 Processing {len(sample_files)} files with BATCH processing...")
        
        # Use optimized batch processing
        await self.process_files_batch(sample_files)
        
        self._generate_comprehensive_report()

    def _generate_comprehensive_report(self):
        """Generate detailed report with real content analysis"""
        categories = {}
        file_types = {}
        extraction_quality = {"success": 0, "failed": 0}
        
        for result in self.results:
            analysis = result["tagging_analysis"]
            cat = analysis["primary_category"]
            sub = analysis["primary_subcategory"]
            
            # Categories
            if cat not in categories:
                categories[cat] = {}
            categories[cat][sub] = categories[cat].get(sub, 0) + 1
            
            # File types
            file_type = result["file_type"]
            file_types[file_type] = file_types.get(file_type, 0) + 1
            
            # Extraction quality
            quality = analysis.get("extraction_quality", "failed")
            extraction_quality[quality] = extraction_quality.get(quality, 0) + 1
        
        # Calculate stats
        ai_count = sum(1 for r in self.results if r["ai_enhancement"])
        total_size = sum(r["file_size_mb"] for r in self.results)
        avg_time = sum(r["processing_time"] for r in self.results) / len(self.results) if self.results else 0
        avg_confidence = sum(r["confidence"] for r in self.results) / len(self.results) if self.results else 0
        
        # Success rate
        success_rate = extraction_quality.get("success", 0) / len(self.results) * 100 if self.results else 0
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "processor_type": "REAL_PRODUCTION_PROCESSOR_WITH_JARVIS_EXTRACTORS",
            "llm_model": "meta-llama-3-8b-instruct",
            "total_files": len(self.results),
            "successful_extractions": extraction_quality.get("success", 0),
            "extraction_success_rate": f"{success_rate:.1f}%",
            "ai_enhancements": ai_count,
            "category_statistics": categories,
            "file_type_statistics": file_types,
            "extraction_quality_stats": extraction_quality,
            "total_size_mb": total_size,
            "avg_processing_time": avg_time,
            "avg_confidence": avg_confidence,
            "results": self.results
        }
        
        report_path = self.output_folder / "real_production_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 REAL PRODUCTION PROCESSOR REPORT: {report_path}")
        print(f"\n🎯 REAL CONTENT PROCESSING SUMMARY:")
        print("="*60)
        print(f"📁 Files processed: {len(self.results)}")
        print(f"✅ Successful extractions: {extraction_quality.get('success', 0)}/{len(self.results)} ({success_rate:.1f}%)")
        print(f"📊 Average confidence: {avg_confidence:.1f}%")
        print(f"💾 Total size: {total_size:.1f} MB")
        print(f"⏱️ Average time: {avg_time:.2f}s")
        print(f"🤖 AI enhancements: {ai_count}/{len(self.results)}")
        
        print(f"\n📂 CATEGORIES (Real Content Analysis):")
        for cat, subcats in categories.items():
            total_in_cat = sum(subcats.values())
            print(f"   📁 {cat}: {total_in_cat} files")
            for sub, count in sorted(subcats.items(), key=lambda x: x[1], reverse=True):
                print(f"      └── {sub}: {count}")
        
        print(f"\n📄 FILE TYPES:")
        for ftype, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
            print(f"   {ftype.upper()}: {count}")
        
        print(f"\n🎉 REAL PRODUCTION PROCESSOR COMPLETE!")
        print("✅ Using actual Jarvis content extraction")
        print("✅ Analyzing real content instead of filenames")
        print("🔄 Dashboard will load this newest report")


async def main():
    print("🔧 REAL PRODUCTION PROCESSOR")
    print("🎯 Using actual Jarvis extractors for content analysis")
    print("="*60)
    
    processor = RealProductionProcessor()
    await processor.process_real_files()


if __name__ == "__main__":
    asyncio.run(main())
