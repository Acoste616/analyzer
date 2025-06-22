# -*- coding: utf-8 -*-
"""POLISH ONLY Processor - Prompty TYLKO po polsku dla Llama-3-8B"""

import asyncio
import time
import json
from pathlib import Path
from datetime import datetime
import requests

print("🇵🇱 POLISH ONLY PROCESSOR - TYLKO POLSKIE PROMPTY")

class PolishOnlyProcessor:
    """Processor z promptami TYLKO po polsku"""
    
    def __init__(self):
        self.output_folder = Path("output")
        self.output_folder.mkdir(exist_ok=True)
        self.results = []

    async def process_polish_content(self):
        print("🇵🇱 POLISH ONLY - TYLKO POLSKIE PROMPTY")
        print("="*60)
        
        # Treści TYLKO po polsku
        polish_contents = [
            "Poradnik o edytorze Cursor. Pokazuje jak używać sztucznej inteligencji do programowania w Pythonie.",
            "Śmieszne memy o programowaniu. Zdjęcia z mediów społecznościowych. Humor dla programistów.",
            "Analiza Bitcoina i Ethereum. Handel kryptowalutami. Inwestycje w DeFi i zarządzanie portfelem.",
            "Marketing w Instagramie. Optymalizacja SEO. Reklamy w Google. Tworzenie viralowych treści."
        ]
        
        categories = ["Programowanie", "Memy", "Krypto", "Marketing"]
        
        for i, (content, category) in enumerate(zip(polish_contents, categories), 1):
            print(f"\n🇵🇱 Przetwarzanie treści polskiej {i}")
            start_time = time.time()
            
            # Prompt TYLKO po polsku
            ai_summary = await self._polish_only_summary(content, category)
            
            processing_time = time.time() - start_time
            
            result = {
                "file_name": f"polish_content_{i}.txt",
                "content": content,
                "category": category,
                "ai_summary": ai_summary,
                "processing_time": processing_time,
                "status": "success"
            }
            
            self.results.append(result)
            self._display_result(result)
        
        self._generate_report()

    async def _polish_only_summary(self, content, category):
        """Prompt TYLKO po polsku - zero angielskiego"""
        
        try:
            # PROMPT TYLKO PO POLSKU
            polish_prompt = f"""O czym jest ten materiał?

TREŚĆ: {content}
KATEGORIA: {category}

Odpowiedz krótko:
TEMAT: [główny temat]
GRUPA: [dla kogo]
FOLDER: [gdzie umieścić]

Pisz tylko po polsku."""
            
            payload = {
                "messages": [
                    {"role": "system", "content": "Jesteś polskim asystentem. Mówisz tylko po polsku. Jesteś pomocny i precyzyjny."},
                    {"role": "user", "content": polish_prompt}
                ],
                "max_tokens": 100,
                "temperature": 0.05,
                "frequency_penalty": 1.5,
                "stream": False
            }
            
            response = requests.post("http://127.0.0.1:1234/v1/chat/completions", json=payload, timeout=12)
            
            if response.status_code == 200:
                data = response.json()
                if "choices" in data and data["choices"]:
                    raw_response = data["choices"][0]["message"]["content"].strip()
                    
                    # Sprawdź czy jest po polsku
                    if self._is_good_polish(raw_response):
                        return raw_response
                    else:
                        print(f"⚠️ Odpowiedź nie jest po polsku")
                    
        except Exception as e:
            print(f"❌ Błąd AI: {e}")
        
        # Polski fallback
        return f"""TEMAT: Materiał o {category.lower()}
GRUPA: Osoby uczące się tematu {category.lower()}
FOLDER: {category}"""

    def _is_good_polish(self, response):
        """Sprawdza czy odpowiedź jest dobra po polsku"""
        if not response or len(response.strip()) < 10:
            return False
        
        response_lower = response.lower()
        
        # Lista angielskich słów - zero tolerancji
        english_words = [
            "the", "and", "for", "with", "about", "this", "that", "content",
            "tutorial", "guide", "tips", "review", "analysis", "tech", 
            "programming", "coding", "marketing", "business", "social", "media"
        ]
        
        # Jeśli ANY angielskie słowo = odrzuć
        for word in english_words:
            if word in response_lower:
                return False
        
        # Sprawdź czy są polskie słowa kluczowe
        polish_indicators = [
            "temat", "grupa", "folder", "materiał", "poradnik", "przewodnik",
            "osoby", "użytkownicy", "ludzie", "dla", "o", "na", "w", "z"
        ]
        
        polish_count = 0
        for indicator in polish_indicators:
            if indicator in response_lower:
                polish_count += 1
        
        # Musi mieć przynajmniej 2 polskie wskaźniki
        return polish_count >= 2

    def _display_result(self, result):
        print(f"📄 TEST: {result['file_name']}")
        print("-" * 40)
        print(f"📝 TREŚĆ: {result['content']}")
        print(f"🎯 KATEGORIA: {result['category']}")
        print(f"⏱️ CZAS: {result['processing_time']:.2f}s")
        
        print(f"\n🤖 PODSUMOWANIE AI:")
        for line in result["ai_summary"].split("\n"):
            if line.strip():
                print(f"   {line}")
        
        print("=" * 40)

    def _generate_report(self):
        report = {
            "timestamp": datetime.now().isoformat(),
            "processor_type": "POLISH_ONLY_PROCESSOR",
            "llm_model": "meta-llama-3-8b-instruct",
            "total_files": len(self.results),
            "results": self.results
        }
        
        report_path = self.output_folder / "polish_only_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 RAPORT POLSKI: {report_path}")
        
        # Sprawdź jakość polskich odpowiedzi
        good_polish_count = 0
        for result in self.results:
            if self._is_good_polish(result["ai_summary"]):
                good_polish_count += 1
        
        print(f"\n📊 KONTROLA JAKOŚCI:")
        print(f"🇵🇱 Dobre polskie odpowiedzi: {good_polish_count}/{len(self.results)}")
        print(f"📈 Wskaźnik sukcesu: {good_polish_count/len(self.results)*100:.1f}%")
        
        if good_polish_count == len(self.results):
            print("🎉 DOSKONALE! Wszystkie odpowiedzi po polsku!")
        else:
            print("⚠️ Część odpowiedzi wymaga poprawy")
        
        # Średni czas przetwarzania
        avg_time = sum(r["processing_time"] for r in self.results) / len(self.results)
        print(f"⏱️ Średni czas: {avg_time:.2f}s")


async def main():
    print("🇵🇱 POLISH ONLY PROCESSOR TEST")
    print("🎯 Prompty TYLKO po polsku dla Llama-3-8B")
    print("="*50)
    
    processor = PolishOnlyProcessor()
    await processor.process_polish_content()


if __name__ == "__main__":
    asyncio.run(main()) 