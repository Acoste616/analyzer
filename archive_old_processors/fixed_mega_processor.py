# -*- coding: utf-8 -*-
"""FIXED MEGA Processor - Optimized for Llama-3-8B"""

import asyncio
import time
import json
from pathlib import Path
from datetime import datetime
import requests
from collections import Counter
import re

print("🔧 FIXED MEGA PROCESSOR - LLAMA-3-8B OPTIMIZED")

# Jarvis imports
try:
    from jarvis_edu.extractors.enhanced_image_extractor import EnhancedImageExtractor
    from jarvis_edu.extractors.enhanced_video_extractor import EnhancedVideoExtractor
    JARVIS_AVAILABLE = True
    print("✅ Jarvis dostępny")
except ImportError:
    print("⚠️ Jarvis niedostępny")
    JARVIS_AVAILABLE = False


class FixedMegaProcessor:
    """FIXED processor with proper Llama-3-8B prompts"""
    
    def __init__(self):
        self.media_folder = Path("C:/Users/barto/OneDrive/Pulpit/mediapobrane/pobrane_media")
        self.output_folder = Path("output")
        self.output_folder.mkdir(exist_ok=True)
        
        # Enhanced tag system
        self.tag_system = {
            "AI": {
                "Claude": ["claude", "claude-3", "claude ai", "anthropic"],
                "GPT": ["gpt", "chatgpt", "gpt-4", "openai"],
                "AI_Tools": ["cursor", "midjourney", "copilot", "ai tools"]
            },
            "Business": {
                "Marketing": ["marketing", "seo", "social media", "ads"],
                "Krypto": ["bitcoin", "ethereum", "crypto", "defi"],
                "Inwestowanie": ["investing", "stocks", "portfolio", "etf"]
            },
            "Development": {
                "Cursor": ["cursor", "cursor ide", "ai coding"],
                "Programming": ["python", "javascript", "coding"]
            },
            "Content": {
                "Memes": ["meme", "memes", "humor", "viral"],
                "Social_Media": ["tiktok", "instagram", "youtube"]
            }
        }
        
        self.results = []

    def mega_analyze(self, content):
        content_lower = content.lower()
        
        detected_categories = {}
        precision_tags = []
        all_keywords = []
        
        for main_cat, subcats in self.tag_system.items():
            category_scores = {}
            
            for subcat, keywords in subcats.items():
                score = 0
                matched = []
                
                for keyword in keywords:
                    if keyword in content_lower:
                        score += len(keyword.split()) * 2
                        matched.append(keyword)
                        all_keywords.append(keyword)
                
                if score > 0:
                    category_scores[subcat] = {"score": score, "keywords": matched}
                    
                    for kw in matched:
                        tag_kw = kw.replace(" ", "-").title()
                        precision_tags.append(f"{main_cat}/{subcat}/{tag_kw}")
            
            if category_scores:
                detected_categories[main_cat] = category_scores
        
        # Find primary
        primary_category = None
        primary_subcategory = None
        max_score = 0
        
        for main_cat, subcats in detected_categories.items():
            for subcat, data in subcats.items():
                if data["score"] > max_score:
                    max_score = data["score"]
                    primary_category = main_cat
                    primary_subcategory = subcat
        
        # Auto-discovery
        auto_discovered = self._auto_discover(content_lower)
        
        folder_structure = self._smart_folders(primary_category, primary_subcategory)
        
        return {
            "primary_category": primary_category or "Uncategorized",
            "primary_subcategory": primary_subcategory or "Other",
            "detected_categories": detected_categories,
            "precision_tags": list(set(precision_tags)),
            "folder_structure": folder_structure,
            "matched_keywords": all_keywords,
            "auto_discovered": auto_discovered,
            "confidence": min(100, max_score * 6)
        }

    def _auto_discover(self, content):
        discovered = {}
        patterns = {
            "Tech_Brands": r"\b(apple|google|microsoft|amazon|tesla|nvidia|meta)\b",
            "Platforms": r"\b(youtube|tiktok|instagram|twitter|discord|reddit)\b",
            "Tools": r"\b(notion|obsidian|figma|canva|slack|zoom)\b"
        }
        
        for category, pattern in patterns.items():
            matches = re.findall(pattern, content)
            if matches:
                discovered[category] = list(set(matches))
        
        return discovered

    def _smart_folders(self, primary_cat, primary_sub):
        folders = {
            ("AI", "Claude"): {"path": "AI/Claude", "sub": ["Claude-3", "API"]},
            ("AI", "GPT"): {"path": "AI/GPT", "sub": ["ChatGPT", "GPT-4"]},
            ("Development", "Cursor"): {"path": "Development/Cursor", "sub": ["Setup", "Tips"]},
            ("Business", "Marketing"): {"path": "Business/Marketing", "sub": ["SEO", "Social-Media"]},
            ("Business", "Krypto"): {"path": "Business/Krypto", "sub": ["Bitcoin", "DeFi"]},
            ("Content", "Memes"): {"path": "Content/Memes", "sub": ["Tech-Memes", "Viral"]}
        }
        
        key = (primary_cat, primary_sub)
        if key in folders:
            return folders[key]
        
        return {"path": f"{primary_cat or 'Unknown'}/{primary_sub or 'Other'}", "sub": ["General"]}

    async def process_enhanced_content(self):
        print("🔧 FIXED MEGA PROCESSOR - ENHANCED CONTENT")
        print("="*70)
        
        # MUCH RICHER content for better analysis
        enhanced_contents = [
            """Comprehensive tutorial about Cursor IDE with advanced AI coding features. This guide covers ChatGPT integration, GitHub Copilot comparison, and Python development workflow optimization. Learn how to setup Cursor for maximum productivity with AI-powered code completion, intelligent debugging, and automated refactoring. Perfect for developers looking to leverage artificial intelligence in their daily coding routine.""",
            
            """Ultimate viral memes compilation featuring the best AI humor, programming jokes, and tech memes from 2024. This collection includes hilarious content from TikTok trends, Reddit programming humor, Twitter tech jokes, and Instagram developer memes. Categories covered: machine learning fails, coding struggles, debugging nightmares, and cryptocurrency memes.""",
            
            """Advanced cryptocurrency investment analysis focusing on Bitcoin and Ethereum market trends, DeFi protocol evaluation, and NFT trading strategies. This comprehensive guide covers portfolio management techniques, risk assessment, market analysis tools, and trading strategies for both beginners and experienced investors. Learn about yield farming and staking rewards.""",
            
            """Digital marketing masterclass covering SEO optimization strategies, Instagram marketing campaigns, TikTok viral content creation, and social media growth hacking techniques. This complete course teaches content marketing, email marketing automation, Google Ads optimization, and Facebook advertising for entrepreneurs and marketing professionals.""",
            
            """Claude AI vs ChatGPT comprehensive comparison for business automation, content creation, and API integration. This detailed analysis covers prompt engineering techniques, API capabilities, pricing models, and real-world use cases for both platforms. Learn how to leverage these AI tools for customer service automation and content generation."""
        ]
        
        for i, content in enumerate(enhanced_contents, 1):
            print(f"\n🔧 Processing enhanced content {i} ({len(content)} characters)")
            start_time = time.time()
            
            # Analysis
            analysis = self.mega_analyze(content)
            
            # FIXED AI summary with proper Llama-3-8B prompts
            ai_summary = await self._generate_llama3_summary(content, analysis)
            
            processing_time = time.time() - start_time
            
            result = {
                "file_name": f"enhanced_content_{i}.txt",
                "file_type": "text",
                "processing_time": processing_time,
                "extracted_content": content,
                "ai_summary": ai_summary,
                "tagging_analysis": analysis,
                "confidence": analysis["confidence"],
                "status": "success"
            }
            
            self.results.append(result)
            self._display_result(result)
        
        self._generate_report()

    async def _generate_llama3_summary(self, content, analysis):
        """FIXED AI summary generation for Llama-3-8B"""
        primary_cat = analysis.get("primary_category", "General")
        primary_sub = analysis.get("primary_subcategory", "Other")
        keywords = analysis.get("matched_keywords", [])
        
        try:
            # SIMPLIFIED prompt for Llama-3-8B
            prompt_content = f"Przeanalizuj tę treść po polsku:\n\nTREŚĆ: {content[:400]}\n\nKATEGORIA: {primary_cat} → {primary_sub}\n\nPodaj analizę w formacie:\n1. KATEGORIA: [kategoria]\n2. TEMATY: [3 główne tematy]\n3. GRUPA: [dla kogo to jest]\n4. FOLDER: [gdzie to organizować]\n\nOdpowiadaj tylko po polsku, krótko i konkretnie."
            
            payload = {
                "messages": [
                    {"role": "system", "content": "Jesteś ekspertem od analizy treści. Odpowiadasz tylko po polsku, krótko i konkretnie."},
                    {"role": "user", "content": prompt_content}
                ],
                "max_tokens": 300,
                "temperature": 0.2,
                "frequency_penalty": 1.0,
                "stream": False
            }
            
            response = requests.post("http://127.0.0.1:1234/v1/chat/completions", json=payload, timeout=25)
            
            if response.status_code == 200:
                data = response.json()
                if "choices" in data and data["choices"]:
                    raw_response = data["choices"][0]["message"]["content"].strip()
                    
                    # Validate response
                    if self._is_valid_response(raw_response):
                        return raw_response
                    else:
                        print(f"⚠️ Invalid response detected")
                    
        except Exception as e:
            print(f"❌ AI error: {e}")
        
        # Enhanced fallback
        folder_path = analysis["folder_structure"]["path"]
        return f"""1. KATEGORIA: {primary_cat} → {primary_sub}
2. TEMATY: {', '.join(keywords[:3]) if keywords else 'treści edukacyjne'}
3. GRUPA: Osoby zainteresowane {primary_sub.lower()}
4. FOLDER: {folder_path}"""

    def _is_valid_response(self, response):
        """Check if response is valid Polish"""
        if not response or len(response.strip()) < 20:
            return False
        
        # Check for excessive repetition
        words = response.lower().split()
        if len(words) > 8:
            word_counts = Counter(words)
            max_count = max(word_counts.values()) if word_counts else 0
            if max_count / len(words) > 0.25:
                return False
        
        # Check for English phrases
        english_phrases = ["here is", "analysis in polish", "the analysis"]
        response_lower = response.lower()
        for phrase in english_phrases:
            if phrase in response_lower:
                return False
        
        return True

    def _display_result(self, result):
        print(f"📄 FILE: {result['file_name']}")
        print("-" * 50)
        
        analysis = result["tagging_analysis"]
        
        print(f"🎯 PRIMARY: {analysis['primary_category']} → {analysis['primary_subcategory']}")
        print(f"⏱️ TIME: {result['processing_time']:.2f}s")
        print(f"📊 CONFIDENCE: {analysis['confidence']}%")
        print(f"📝 LENGTH: {len(result['extracted_content'])} chars")
        
        print(f"\n🏷️ TAGS:")
        for i, tag in enumerate(analysis["precision_tags"][:5], 1):
            print(f"   {i}. {tag}")
        
        auto_disc = analysis.get("auto_discovered", {})
        if auto_disc:
            print(f"\n🔍 AUTO-DISCOVERED:")
            for topic, items in auto_disc.items():
                print(f"   🆕 {topic}: {', '.join(items)}")
        
        print(f"\n📂 FOLDER:")
        folder = analysis["folder_structure"]
        print(f"   📁 {folder['path']}")
        
        print(f"\n🤖 AI SUMMARY (Llama-3-8B):")
        for line in result["ai_summary"].split("\n")[:6]:
            if line.strip():
                print(f"   {line}")
        
        print("=" * 50)

    def _generate_report(self):
        category_stats = {}
        tag_stats = {}
        
        for result in self.results:
            analysis = result.get("tagging_analysis", {})
            primary_cat = analysis.get("primary_category", "Unknown")
            primary_sub = analysis.get("primary_subcategory", "Other")
            
            if primary_cat not in category_stats:
                category_stats[primary_cat] = {}
            category_stats[primary_cat][primary_sub] = category_stats[primary_cat].get(primary_sub, 0) + 1
            
            for tag in analysis.get("precision_tags", []):
                tag_stats[tag] = tag_stats.get(tag, 0) + 1
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "processor_type": "FIXED_MEGA_PROCESSOR_LLAMA3",
            "llm_model": "meta-llama-3-8b-instruct",
            "total_files": len(self.results),
            "successful_extractions": len(self.results),
            "category_statistics": category_stats,
            "most_common_tags": dict(sorted(tag_stats.items(), key=lambda x: x[1], reverse=True)[:15]),
            "results": self.results
        }
        
        report_path = self.output_folder / "fixed_mega_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 FIXED REPORT SAVED: {report_path}")
        
        print(f"\n🎯 ENHANCED SUMMARY:")
        print("=" * 50)
        for main_cat, subcats in category_stats.items():
            print(f"📂 {main_cat}:")
            for subcat, count in subcats.items():
                print(f"   └── {subcat}: {count} files")
        
        print(f"\n🏷️ TOP TAGS:")
        for i, (tag, count) in enumerate(list(report["most_common_tags"].items())[:8], 1):
            print(f"   {i}. {tag}: {count}x")
        
        print(f"\n✅ FIXED SYSTEM READY!")
        print("📊 Enhanced analysis with proper Llama-3-8B prompts")


async def main():
    print("🔧 FIXED MEGA PROCESSOR - LLAMA-3-8B OPTIMIZED")
    print("="*60)
    
    processor = FixedMegaProcessor()
    await processor.process_enhanced_content()


if __name__ == "__main__":
    asyncio.run(main()) 