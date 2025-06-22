# -*- coding: utf-8 -*-
import asyncio
import time
import json
from pathlib import Path
from datetime import datetime
import requests
import re
import os

print(' REAL PRODUCTION PROCESSOR')

class RealProductionProcessor:
    def __init__(self):
        self.media_folder = Path("C:/Users/barto/OneDrive/Pulpit/mediapobrane/pobrane_media")
        self.output_folder = Path("output")
        self.output_folder.mkdir(exist_ok=True)
        self.results = []
        
        # Expanded tag system for real files
        self.tag_system = {
            "AI": {
                "Claude": ["claude", "anthropic"],
                "GPT": ["gpt", "chatgpt", "openai"],
                "Tools": ["cursor", "copilot", "midjourney"]
            },
            "Business": {
                "Marketing": ["marketing", "seo", "ads"],
                "Krypto": ["bitcoin", "crypto", "ethereum"],
                "Investing": ["stocks", "portfolio", "trading"]
            },
            "Development": {
                "Cursor": ["cursor", "cursor ide"],
                "Programming": ["python", "javascript", "coding"],
                "Web": ["react", "vue", "html", "css"]
            },
            "Content": {
                "Memes": ["meme", "humor", "funny"],
                "Social": ["tiktok", "instagram", "youtube"],
                "Gaming": ["gaming", "games", "twitch"]
            }
        }

    def get_sample_files(self, max_files=30):
        print(f" Scanning: {self.media_folder}")
        
        if not self.media_folder.exists():
            print(" Media folder not found!")
            return []
        
        # Find media files
        all_files = []
        extensions = ['*.mp4', '*.avi', '*.mov', '*.jpg', '*.jpeg', '*.png', '*.gif', '*.mp3', '*.wav']
        
        for ext in extensions:
            files = list(self.media_folder.rglob(ext))
            all_files.extend(files[:5])  # Max 5 per type
            if len(all_files) >= max_files:
                break
        
        print(f" Found {len(all_files)} sample files")
        return all_files[:max_files]

    def extract_content_from_filename(self, file_path):
        filename = file_path.name
        parent_dirs = str(file_path.parent).replace(str(self.media_folder), "")
        
        # Clean filename
        clean_name = filename.replace("_", " ").replace("-", " ").replace(".", " ")
        content = f"{clean_name} {parent_dirs}"
        
        # Add file type context
        ext = file_path.suffix.lower()
        if ext in ['.mp4', '.avi', '.mov']:
            content += " video content"
        elif ext in ['.jpg', '.jpeg', '.png']:
            content += " image content"
        elif ext in ['.mp3', '.wav']:
            content += " audio content"
        
        return content

    def analyze_file(self, file_path):
        try:
            content = self.extract_content_from_filename(file_path)
            content_lower = content.lower()
            
            detected_categories = {}
            all_keywords = []
            
            # Analyze with tag system
            for main_cat, subcats in self.tag_system.items():
                cat_scores = {}
                for subcat, keywords in subcats.items():
                    score = 0
                    matched = []
                    for keyword in keywords:
                        if keyword in content_lower:
                            score += 2
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
            
            # File info
            file_size_mb = file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0
            
            return {
                "primary_category": primary_cat or "General",
                "primary_subcategory": primary_sub or "Other",
                "detected_categories": detected_categories,
                "matched_keywords": all_keywords,
                "confidence": min(100, max_score * 10),
                "extracted_content": content,
                "file_size_mb": file_size_mb,
                "file_type": file_path.suffix.lower().replace(".", "")
            }
            
        except Exception as e:
            print(f" Error: {e}")
            return None

    def polish_summary(self, file_path, analysis):
        cat = analysis.get("primary_category", "General")
        sub = analysis.get("primary_subcategory", "Other")
        confidence = analysis.get("confidence", 0)
        file_type = analysis.get("file_type", "unknown")
        
        polish_map = {
            "AI": "Sztuczna Inteligencja",
            "Business": "Biznes", 
            "Development": "Programowanie",
            "Content": "Treści",
            "Claude": "Claude AI",
            "GPT": "ChatGPT",
            "Cursor": "Edytor Cursor",
            "Marketing": "Marketing",
            "Krypto": "Kryptowaluty",
            "Memes": "Memy"
        }
        
        polish_cat = polish_map.get(cat, cat)
        polish_sub = polish_map.get(sub, sub)
        
        return f" KATEGORIA: {polish_cat}  {polish_sub}\n TYP: {file_type.upper()}\n PEWNOŚĆ: {confidence}%\n PLIK: {file_path.name}"

    async def try_ai_enhancement(self, content, filename):
        try:
            prompt = f"Po polsku krótko - co to za plik?\n\nNazwa: {filename}\nTreść: {content[:100]}"
            
            payload = {
                "messages": [
                    {"role": "system", "content": "Odpowiadasz krótko po polsku."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 30,
                "temperature": 0.1,
                "stream": False
            }
            
            response = requests.post("http://127.0.0.1:1234/v1/chat/completions", json=payload, timeout=4)
            
            if response.status_code == 200:
                data = response.json()
                if "choices" in data and data["choices"]:
                    ai_text = data["choices"][0]["message"]["content"].strip()
                    if len(ai_text) > 5 and not any(eng in ai_text.lower() for eng in ["content", "the"]):
                        return f" AI: {ai_text}"
        except:
            pass
        return None

    async def process_real_files(self):
        print(" REAL PRODUCTION - PROCESSING MEDIA FILES")
        print("="*60)
        
        sample_files = self.get_sample_files(30)
        
        if not sample_files:
            print(" No files to process!")
            return
        
        print(f" Processing {len(sample_files)} files...")
        
        for i, file_path in enumerate(sample_files, 1):
            print(f"\n Processing {i}/{len(sample_files)}: {file_path.name}")
            start_time = time.time()
            
            analysis = self.analyze_file(file_path)
            if not analysis:
                continue
            
            polish_summary = self.polish_summary(file_path, analysis)
            ai_enhancement = await self.try_ai_enhancement(analysis["extracted_content"], file_path.name)
            
            processing_time = time.time() - start_time
            
            result = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "file_type": analysis["file_type"],
                "file_size_mb": analysis["file_size_mb"],
                "processing_time": processing_time,
                "extracted_content": analysis["extracted_content"],
                "ai_summary": polish_summary,
                "ai_enhancement": ai_enhancement,
                "tagging_analysis": analysis,
                "confidence": analysis["confidence"],
                "status": "success"
            }
            
            self.results.append(result)
            
            print(f" {analysis['primary_category']}  {analysis['primary_subcategory']} ({analysis['confidence']}%)")
            if analysis["matched_keywords"]:
                print(f"    Keywords: {', '.join(analysis['matched_keywords'][:3])}")
        
        self._generate_report()

    def _generate_report(self):
        # Calculate statistics
        categories = {}
        file_types = {}
        
        for result in self.results:
            analysis = result["tagging_analysis"]
            cat = analysis["primary_category"]
            sub = analysis["primary_subcategory"]
            
            if cat not in categories:
                categories[cat] = {}
            categories[cat][sub] = categories[cat].get(sub, 0) + 1
            
            file_type = result["file_type"]
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        ai_count = sum(1 for r in self.results if r["ai_enhancement"])
        total_size = sum(r["file_size_mb"] for r in self.results)
        avg_time = sum(r["processing_time"] for r in self.results) / len(self.results) if self.results else 0
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "processor_type": "REAL_PRODUCTION_PROCESSOR",
            "llm_model": "meta-llama-3-8b-instruct",
            "total_files": len(self.results),
            "successful_extractions": len(self.results),
            "ai_enhancements": ai_count,
            "category_statistics": categories,
            "file_type_statistics": file_types,
            "total_size_mb": total_size,
            "avg_processing_time": avg_time,
            "results": self.results
        }
        
        report_path = self.output_folder / "real_production_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n REAL PRODUCTION REPORT: {report_path}")
        print(f"\n SUMMARY:")
        print("="*50)
        print(f" Files processed: {len(self.results)}")
        print(f" Total size: {total_size:.1f} MB")
        print(f" Average time: {avg_time:.2f}s")
        print(f" AI success: {ai_count}/{len(self.results)}")
        
        print(f"\n CATEGORIES:")
        for cat, subcats in categories.items():
            print(f"   {cat}: {sum(subcats.values())} files")
            for sub, count in subcats.items():
                print(f"       {sub}: {count}")
        
        print(f"\n FILE TYPES:")
        for ftype, count in file_types.items():
            print(f"   {ftype.upper()}: {count}")
        
        print(f"\n REAL PRODUCTION COMPLETE!")
        print(" Dashboard ready to load fresh data!")

async def main():
    processor = RealProductionProcessor()
    await processor.process_real_files()

if __name__ == "__main__":
    asyncio.run(main())
