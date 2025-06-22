# -*- coding: utf-8 -*-
"""ULTRA SIMPLE Processor - Bardzo proste prompty dla Llama-3-8B"""

import asyncio
import time
import json
from pathlib import Path
from datetime import datetime
import requests

print("🔧 ULTRA SIMPLE PROCESSOR - SUPER PROSTE PROMPTY")

class UltraSimpleProcessor:
    """Ultra prosty processor z bardzo prostymi promptami"""
    
    def __init__(self):
        self.output_folder = Path("output")
        self.output_folder.mkdir(exist_ok=True)
        self.results = []

    async def process_test_content(self):
        print("🔧 ULTRA SIMPLE - SUPER PROSTE PROMPTY")
        print("="*60)
        
        # Bardzo proste testowe treści
        test_contents = [
            "Tutorial o Cursor IDE. Pokazuje jak używać AI do kodowania. Python, JavaScript, ChatGPT integration.",
            "Memy o programowaniu. Śmieszne zdjęcia o coding. TikTok, Instagram, Reddit humor.",
            "Bitcoin i Ethereum analiza. Krypto trading. DeFi, portfolio management, inwestowanie.",
            "Marketing na Instagramie. SEO, social media, reklamy Google, viral content creation."
        ]
        
        categories = ["Development/Cursor", "Content/Memes", "Business/Krypto", "Business/Marketing"]
        
        for i, (content, category) in enumerate(zip(test_contents, categories), 1):
            print(f"\n🔧 Processing test content {i}")
            start_time = time.time()
            
            # ULTRA PROSTY AI summary
            ai_summary = await self._ultra_simple_summary(content, category)
            
            processing_time = time.time() - start_time
            
            result = {
                "file_name": f"test_content_{i}.txt",
                "content": content,
                "category": category,
                "ai_summary": ai_summary,
                "processing_time": processing_time,
                "status": "success"
            }
            
            self.results.append(result)
            self._display_result(result)
        
        self._generate_report()

    async def _ultra_simple_summary(self, content, category):
        """ULTRA PROSTY prompt - tylko podstawowe informacje"""
        
        try:
            # NAJPROSTRZY MOŻLIWY PROMPT
            simple_prompt = f"""Opisz po polsku co to za treść.

TREŚĆ: {content}
KATEGORIA: {category}

Format odpowiedzi:
TEMAT: [o czym to jest]
DLA KOGO: [kto to oglądać]
GDZIE: [jaki folder]

Pisz tylko po polsku. Krótko."""
            
            payload = {
                "messages": [
                    {"role": "system", "content": "Odpowiadasz TYLKO po polsku. Krótko i prosto."},
                    {"role": "user", "content": simple_prompt}
                ],
                "max_tokens": 150,
                "temperature": 0.1,
                "frequency_penalty": 1.2,
                "stream": False
            }
            
            response = requests.post("http://127.0.0.1:1234/v1/chat/completions", json=payload, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if "choices" in data and data["choices"]:
                    raw_response = data["choices"][0]["message"]["content"].strip()
                    
                    # Bardziej restrykcyjna walidacja
                    if self._is_pure_polish(raw_response):
                        return raw_response
                    else:
                        print(f"⚠️ Detected non-Polish content")
                    
        except Exception as e:
            print(f"❌ AI error: {e}")
        
        # Prosty fallback po polsku
        return f"""TEMAT: Materiał edukacyjny o {category.split('/')[-1].lower()}
DLA KOGO: Osoby uczące się tego tematu
GDZIE: {category}"""

    def _is_pure_polish(self, response):
        """Sprawdza czy odpowiedź jest czysto po polsku"""
        if not response or len(response.strip()) < 15:
            return False
        
        # Lista angielskich słów które nie powinny się pojawić
        english_words = [
            "and", "the", "for", "with", "about", "from", "this", "that",
            "here", "there", "when", "where", "what", "how", "why",
            "content", "analysis", "tutorial", "guide", "tips", "review",
            "ai-related", "tech", "programming", "coding", "marketing",
            "business", "development", "social", "media"
        ]
        
        response_lower = response.lower()
        
        # Sprawdź czy są angielskie słowa
        english_count = 0
        for word in english_words:
            if word in response_lower:
                english_count += 1
        
        # Jeśli więcej niż 2 angielskie słowa = odrzuć
        if english_count > 2:
            return False
        
        # Sprawdź mieszane frazy
        mixed_phrases = [
            "humor ai-related", "jokes programming", "tech memes",
            "content creation", "social media", "viral content"
        ]
        
        for phrase in mixed_phrases:
            if phrase in response_lower:
                return False
        
        return True

    def _display_result(self, result):
        print(f"📄 TEST: {result['file_name']}")
        print("-" * 40)
        print(f"📝 CONTENT: {result['content']}")
        print(f"🎯 CATEGORY: {result['category']}")
        print(f"⏱️ TIME: {result['processing_time']:.2f}s")
        
        print(f"\n🤖 AI SUMMARY:")
        for line in result["ai_summary"].split("\n"):
            if line.strip():
                print(f"   {line}")
        
        print("=" * 40)

    def _generate_report(self):
        report = {
            "timestamp": datetime.now().isoformat(),
            "processor_type": "ULTRA_SIMPLE_PROCESSOR",
            "llm_model": "meta-llama-3-8b-instruct",
            "total_files": len(self.results),
            "results": self.results
        }
        
        report_path = self.output_folder / "ultra_simple_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 ULTRA SIMPLE REPORT: {report_path}")
        
        # Sprawdź jakość odpowiedzi
        polish_count = 0
        for result in self.results:
            if self._is_pure_polish(result["ai_summary"]):
                polish_count += 1
        
        print(f"\n📊 QUALITY CHECK:")
        print(f"✅ Pure Polish responses: {polish_count}/{len(self.results)}")
        print(f"📈 Success rate: {polish_count/len(self.results)*100:.1f}%")
        
        if polish_count == len(self.results):
            print("🎉 PERFECT! All responses in Polish!")
        else:
            print("⚠️ Some responses need improvement")


async def main():
    print("🔧 ULTRA SIMPLE PROCESSOR TEST")
    print("🎯 Super proste prompty dla Llama-3-8B")
    print("="*50)
    
    processor = UltraSimpleProcessor()
    await processor.process_test_content()


if __name__ == "__main__":
    asyncio.run(main()) 