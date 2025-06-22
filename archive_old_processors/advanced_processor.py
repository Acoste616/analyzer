# -*- coding: utf-8 -*-
"""Advanced Content Processor - Precise Tagging System"""

import asyncio
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import requests
from collections import Counter

print("🚀 ADVANCED CONTENT PROCESSOR - LOADING")

# Try Jarvis imports
try:
    from jarvis_edu.extractors.enhanced_image_extractor import EnhancedImageExtractor
    from jarvis_edu.extractors.enhanced_video_extractor import EnhancedVideoExtractor
    JARVIS_AVAILABLE = True
    print("✅ Jarvis dostępny")
except ImportError:
    print("⚠️ Jarvis niedostępny - using mock data")
    JARVIS_AVAILABLE = False


class AdvancedContentProcessor:
    """Advanced processor with PRECISE tagging system per user requirements"""
    
    def __init__(self):
        self.media_folder = Path("C:/Users/barto/OneDrive/Pulpit/mediapobrane/pobrane_media")
        self.output_folder = Path("output")
        self.output_folder.mkdir(exist_ok=True)
        
        # PRECISE TAG SYSTEM - exactly as requested by user
        self.tag_system = {
            "AI": {
                "Claude": ["claude", "claude-3", "claude ai", "anthropic", "claude sonnet"],
                "GPT": ["gpt", "chatgpt", "gpt-4", "gpt-3", "openai", "chat gpt"],
                "AI_Tools": ["midjourney", "stable diffusion", "dall-e", "copilot", "cursor"],
                "AI_Concepts": ["machine learning", "deep learning", "prompt engineering"]
            },
            "Business": {
                "Marketing": ["marketing", "seo", "social media marketing", "content marketing", "email marketing"],
                "Inwestowanie": ["investing", "stocks", "portfolio", "dividends", "etf", "financial planning"],
                "Krypto": ["bitcoin", "ethereum", "crypto", "blockchain", "defi", "nft", "trading"]
            }
        }
        
        self.results = []

    def analyze_content_with_precision_tags(self, content: str) -> Dict[str, Any]:
        """PRECISE tagging analysis per user requirements: AI/Claude/GPT, Business/Marketing/Krypto"""
        content_lower = content.lower()
        
        detected_categories = {}
        precision_tags = []
        all_matched_keywords = []
        
        for main_category, subcategories in self.tag_system.items():
            category_scores = {}
            
            for subcategory, keywords in subcategories.items():
                score = 0
                matched_keywords = []
                
                for keyword in keywords:
                    if keyword in content_lower:
                        score += 1
                        matched_keywords.append(keyword)
                        all_matched_keywords.append(keyword)
                
                if score > 0:
                    category_scores[subcategory] = {
                        'score': score,
                        'keywords': matched_keywords
                    }
                    
                    # Generate PRECISE HIERARCHICAL TAGS as requested
                    for keyword in matched_keywords:
                        if subcategory == "Claude":
                            precision_tags.append(f"AI/Claude/{keyword.replace(' ', '-').title()}")
                        elif subcategory == "GPT":
                            precision_tags.append(f"AI/GPT/{keyword.replace(' ', '-').title()}")
                        elif subcategory == "Marketing":
                            precision_tags.append(f"Business/Marketing/{keyword.replace(' ', '-').title()}")
                        elif subcategory == "Krypto":
                            precision_tags.append(f"Business/Krypto/{keyword.replace(' ', '-').title()}")
                        elif subcategory == "Inwestowanie":
                            precision_tags.append(f"Business/Inwestowanie/{keyword.replace(' ', '-').title()}")
                        else:
                            precision_tags.append(f"{main_category}/{subcategory}/{keyword.replace(' ', '-').title()}")
            
            if category_scores:
                detected_categories[main_category] = category_scores
        
        # Find primary category/subcategory
        primary_category = None
        primary_subcategory = None
        max_score = 0
        
        for main_cat, sub_cats in detected_categories.items():
            for sub_cat, data in sub_cats.items():
                if data['score'] > max_score:
                    max_score = data['score']
                    primary_category = main_cat
                    primary_subcategory = sub_cat
        
        folder_structure = self._generate_folder_structure(primary_category, primary_subcategory)
        
        return {
            "primary_category": primary_category or "Uncategorized",
            "primary_subcategory": primary_subcategory or "Other",
            "detected_categories": detected_categories,
            "precision_tags": list(set(precision_tags)),
            "folder_structure": folder_structure,
            "matched_keywords": all_matched_keywords,
            "confidence": min(100, max_score * 15)
        }

    def _generate_folder_structure(self, primary_cat: str, primary_sub: str) -> Dict[str, Any]:
        """Generate folder structure exactly as user requested"""
        if not primary_cat or not primary_sub:
            return {"suggested_path": "Uncategorized", "sub_folders": ["Other"]}
        
        if primary_cat == "AI":
            if primary_sub == "Claude":
                return {"suggested_path": "AI/Claude", "sub_folders": ["Claude-3", "Anthropic-Tools"]}
            elif primary_sub == "GPT":
                return {"suggested_path": "AI/GPT", "sub_folders": ["ChatGPT", "GPT-4", "OpenAI-Tools"]}
        elif primary_cat == "Business":
            if primary_sub == "Marketing":
                return {"suggested_path": "Business/Marketing", "sub_folders": ["SEO", "Social-Media"]}
            elif primary_sub == "Krypto":
                return {"suggested_path": "Business/Krypto", "sub_folders": ["Bitcoin", "Ethereum", "DeFi"]}
            elif primary_sub == "Inwestowanie":
                return {"suggested_path": "Business/Inwestowanie", "sub_folders": ["Stocks", "ETF"]}
        
        return {"suggested_path": f"{primary_cat}/{primary_sub}", "sub_folders": ["General"]}

    async def process_with_advanced_tagging(self):
        """Main processing function with advanced tagging"""
        print("🚀 ADVANCED CONTENT PROCESSOR - PRECISE TAGGING SYSTEM")
        print("="*80)
        print("📂 SYSTEM TAGOWANIA:")
        print("   • AI/Claude/ - dla materiałów o Claude AI")
        print("   • AI/GPT/ - dla materiałów o ChatGPT, GPT-4")
        print("   • Business/Marketing/ - marketing, SEO, reklama")
        print("   • Business/Krypto/ - Bitcoin, Ethereum, DeFi")
        print("   • Business/Inwestowanie/ - akcje, ETF, portfolio")
        print("="*80)
        
        # Generate mock content for testing
        mock_contents = [
            "Tutorial o ChatGPT i Claude AI do automatyzacji biznesu. Porównanie modeli GPT-4 vs Claude-3.",
            "Strategie inwestycyjne w kryptowaluty: Bitcoin, Ethereum i protokoły DeFi. Zarządzanie portfolio.",
            "Masterclass digital marketing: SEO, social media advertising, content marketing. Optymalizacja konwersji.",
            "Programowanie AI: machine learning, deep learning. Wykorzystanie OpenAI API i Anthropic Claude."
        ]
        
        for i, content in enumerate(mock_contents, 1):
            print(f"\n📁 Processing mock content {i}")
            start_time = time.time()
            
            # ADVANCED TAGGING ANALYSIS
            tagging_analysis = self.analyze_content_with_precision_tags(content)
            
            # AI SUMMARY
            ai_summary = await self._generate_ai_summary(content, tagging_analysis)
            
            processing_time = time.time() - start_time
            
            result = {
                'file_name': f'mock_content_{i}.txt',
                'file_type': 'text',
                'processing_time': processing_time,
                'extracted_content': content,
                'ai_summary': ai_summary,
                'tagging_analysis': tagging_analysis,
                'confidence': 85
            }
            
            self.results.append(result)
            self._display_result(result)
        
        self._generate_comprehensive_report()

    async def _generate_ai_summary(self, content: str, tagging: Dict) -> str:
        """Generate AI summary with precise categorization"""
        primary_cat = tagging.get('primary_category', 'General')
        primary_sub = tagging.get('primary_subcategory', 'Other')
        precision_tags = tagging.get('precision_tags', [])
        
        try:
            prompt = f"""CATEGORIZATION TASK:

CATEGORY: {primary_cat} → {primary_sub}
TAGS: {', '.join(precision_tags[:6])}

CONTENT: {content}

Provide:
1. EXACT CATEGORY
2. MAIN TOPICS (3-4 key points)
3. TARGET AUDIENCE
4. FOLDER SUGGESTION

Respond in Polish, be specific. NO repetition."""
            
            payload = {
                "messages": [
                    {"role": "system", "content": "Content categorization expert. Structured analysis in Polish. No repetition."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 250,
                "temperature": 0.1,
                "frequency_penalty": 1.5,
                "stream": False
            }
            
            response = requests.post('http://127.0.0.1:1234/v1/chat/completions', json=payload, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and data['choices']:
                    return data['choices'][0]['message']['content'].strip()
                    
        except Exception as e:
            print(f"AI summary error: {e}")
        
        # Fallback response
        folder_path = tagging['folder_structure']['suggested_path']
        return f"""KATEGORIA: {primary_cat} → {primary_sub}

GŁÓWNE TEMATY: {', '.join(tagging.get('matched_keywords', [])[:4])}
GRUPA DOCELOWA: Osoby zainteresowane {primary_sub.lower()}
ORGANIZACJA: {folder_path}
PRECISION TAGS: {', '.join(precision_tags[:4])}"""

    def _display_result(self, result: Dict[str, Any]):
        """Display processing result with advanced tagging"""
        print(f"📄 FILE: {result['file_name']}")
        print("-" * 60)
        
        tagging = result['tagging_analysis']
        
        print(f"🎯 PRIMARY: {tagging['primary_category']} → {tagging['primary_subcategory']}")
        print(f"⏱️ TIME: {result['processing_time']:.2f}s")
        print(f"📊 CONFIDENCE: {tagging['confidence']}%")
        
        print(f"\n🏷️ PRECISION TAGS:")
        for tag in tagging['precision_tags'][:6]:
            print(f"   • {tag}")
        
        print(f"\n📂 FOLDER STRUCTURE:")
        folder_struct = tagging['folder_structure']
        print(f"   📁 {folder_struct['suggested_path']}")
        for subfolder in folder_struct.get('sub_folders', [])[:3]:
            print(f"      └── {subfolder}")
        
        print(f"\n🤖 AI SUMMARY:")
        print(result['ai_summary'][:200] + "..." if len(result['ai_summary']) > 200 else result['ai_summary'])
        
        print("=" * 60)

    def _generate_comprehensive_report(self):
        """Generate comprehensive report with statistics"""
        category_stats = {}
        tag_stats = {}
        
        for result in self.results:
            tagging = result.get('tagging_analysis', {})
            primary_cat = tagging.get('primary_category', 'Unknown')
            primary_sub = tagging.get('primary_subcategory', 'Other')
            
            if primary_cat not in category_stats:
                category_stats[primary_cat] = {}
            if primary_sub not in category_stats[primary_cat]:
                category_stats[primary_cat][primary_sub] = 0
            category_stats[primary_cat][primary_sub] += 1
            
            for tag in tagging.get('precision_tags', []):
                if tag not in tag_stats:
                    tag_stats[tag] = 0
                tag_stats[tag] += 1
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'processor_type': 'ADVANCED_CONTENT_PROCESSOR_PRECISION_TAGGING',
            'total_files': len(self.results),
            'successful_extractions': len(self.results),
            'category_statistics': category_stats,
            'most_common_tags': dict(sorted(tag_stats.items(), key=lambda x: x[1], reverse=True)[:15]),
            'avg_processing_time': sum(r.get('processing_time', 0) for r in self.results) / len(self.results) if self.results else 0,
            'results': self.results
        }
        
        report_path = self.output_folder / 'advanced_content_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 COMPREHENSIVE REPORT SAVED: {report_path}")
        
        print(f"\n🎯 CATEGORIZATION SUMMARY:")
        print("=" * 60)
        for main_cat, sub_cats in category_stats.items():
            print(f"📂 {main_cat}:")
            for sub_cat, count in sub_cats.items():
                print(f"   └── {sub_cat}: {count} files")
        
        print(f"\n🏷️ TOP PRECISION TAGS:")
        print("=" * 60)
        for tag, count in list(report['most_common_tags'].items())[:10]:
            print(f"   • {tag}: {count}x")
        
        print(f"\n✅ ADVANCED TAGGING SYSTEM READY!")
        print("📊 Dashboard will be updated with new data")
        print("🏷️ Structure for organizing materials:")
        print("   📁 AI/Claude/")
        print("   📁 AI/GPT/")
        print("   📁 Business/Marketing/")
        print("   📁 Business/Krypto/")
        print("   📁 Business/Inwestowanie/")


async def main():
    """Main function"""
    print("🚀 ADVANCED CONTENT PROCESSOR WITH PRECISION TAGGING")
    print("🎯 Designed for: AI/Claude/GPT, Business/Marketing/Krypto/Inwestowanie")
    print("=" * 80)
    
    processor = AdvancedContentProcessor()
    await processor.process_with_advanced_tagging()


if __name__ == '__main__':
    asyncio.run(main())
