#!/usr/bin/env python3
"""
Setup i konfiguracja LM Studio dla Jarvis EDU
=============================================

Skrypt sprawdza, konfiguruje i pomaga w instalacji LM Studio
"""

import sys
import json
import asyncio
import requests
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.prompt import Prompt, Confirm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from jarvis_edu.processors.lm_studio import LMStudioProcessor
    from jarvis_edu.core.logger import get_logger
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're in the project directory")
    sys.exit(1)

console = Console()
logger = get_logger("lm_studio_setup")


class LMStudioConfigurator:
    """LM Studio configurator and helper."""
    
    def __init__(self):
        self.lm_studio = LMStudioProcessor()
        self.recommended_models = [
            {
                "name": "mistral-7b-instruct-v0.3",
                "size": "4.4 GB",
                "description": "Szybki i skuteczny model do zadań edukacyjnych",
                "good_for": ["Tworzenie lekcji", "Analiza treści", "Szybkie odpowiedzi"],
                "requirements": "8 GB RAM"
            },
            {
                "name": "llama-3.1-8b-instruct",
                "size": "4.7 GB", 
                "description": "Bardzo dobry model ogólnego użytku",
                "good_for": ["Szczegółowa analiza", "Złożone zadania", "Wysokiej jakości treść"],
                "requirements": "12 GB RAM"
            },
            {
                "name": "phi-3-mini-4k-instruct",
                "size": "2.3 GB",
                "description": "Lekki model dla słabszych komputerów",
                "good_for": ["Podstawowa analiza", "Szybkie przetwarzanie", "Mało zasobów"],
                "requirements": "4 GB RAM"
            }
        ]
    
    async def check_connection(self):
        """Sprawdz połączenie z LM Studio."""
        console.print("\n[bold]🔍 Sprawdzanie połączenia z LM Studio[/bold]")
        
        try:
            is_available = await self.lm_studio.is_available()
            
            if is_available:
                console.print("[green]✅ LM Studio działa[/green]")
                
                connection_info = await self.lm_studio.test_connection()
                console.print(f"[blue]📡 Endpoint: {connection_info['endpoint']}[/blue]")
                
                models = connection_info.get("models", [])
                if models:
                    console.print(f"[green]🧠 Załadowanych modeli: {len(models)}[/green]")
                    return True, models
                else:
                    console.print("[yellow]⚠️  Brak załadowanych modeli[/yellow]")
                    return True, []
            else:
                console.print("[red]❌ LM Studio nie odpowiada[/red]")
                return False, []
                
        except Exception as e:
            console.print(f"[red]❌ Błąd połączenia: {e}[/red]")
            return False, []
    
    def show_installation_guide(self):
        """Pokaż przewodnik instalacji LM Studio."""
        console.print("\n[bold blue]📥 Przewodnik instalacji LM Studio[/bold blue]")
        
        steps = [
            "1. 🌐 Przejdź na stronę: https://lmstudio.ai/",
            "2. 📥 Pobierz LM Studio dla Windows",
            "3. 🔧 Zainstaluj aplikację (wymaga ~500 MB)",
            "4. 🚀 Uruchom LM Studio",
            "5. 🧠 Pobierz i załaduj model (patrz rekomendacje poniżej)",
            "6. ▶️  Uruchom serwer (Local Server)",
            "7. ✅ Sprawdź połączenie tym skryptem ponownie"
        ]
        
        for step in steps:
            console.print(step)
        
        console.print(Panel(
            "[bold]⚠️  Wymagania systemowe:[/bold]\n"
            "• Windows 10/11 (64-bit)\n"
            "• 8+ GB RAM (dla większości modeli)\n"
            "• 5-10 GB wolnego miejsca na dysku\n"
            "• Zalecane: GPU z 4+ GB VRAM",
            title="Wymagania",
            border_style="yellow"
        ))
    
    def show_model_recommendations(self):
        """Pokaż rekomendowane modele."""
        console.print("\n[bold]🧠 Rekomendowane modele[/bold]")
        
        table = Table(title="Modele do pobrania w LM Studio")
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Rozmiar", style="yellow")
        table.add_column("Opis", style="white")
        table.add_column("Zastosowanie", style="green")
        table.add_column("Wymagania", style="red")
        
        for model in self.recommended_models:
            table.add_row(
                model["name"],
                model["size"],
                model["description"][:30] + "..." if len(model["description"]) > 30 else model["description"],
                ", ".join(model["good_for"][:2]),
                model["requirements"]
            )
        
        console.print(table)
        
        console.print("\n[bold yellow]💡 Jak pobrać model w LM Studio:[/bold yellow]")
        console.print("1. Otwórz zakładkę 'Search' w LM Studio")
        console.print("2. Wyszukaj nazwę modelu (np. 'mistral-7b-instruct')")
        console.print("3. Kliknij 'Download' przy wybranym modelu")
        console.print("4. Po pobraniu przejdź do 'My Models'")
        console.print("5. Kliknij 'Load' przy modelu")
        console.print("6. Przejdź do 'Local Server' i kliknij 'Start Server'")
    
    async def test_model_performance(self):
        """Testuj wydajność modelu."""
        console.print("\n[bold]🧪 Test wydajności modelu[/bold]")
        
        test_prompts = [
            {
                "prompt": "Napisz krótką lekcję o podstawach Pythona w formacie JSON z polami: title, summary, content",
                "description": "Test tworzenia lekcji"
            },
            {
                "prompt": "Przeanalizuj ten tekst i wyodrębnij kluczowe tematy: 'Python to język programowania'",
                "description": "Test analizy tekstu"
            }
        ]
        
        for i, test in enumerate(test_prompts, 1):
            console.print(f"\n[blue]Test {i}: {test['description']}[/blue]")
            
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Generowanie odpowiedzi...", total=1)
                    
                    import time
                    start_time = time.time()
                    
                    response = await self.lm_studio.generate_lesson_from_prompt(test["prompt"])
                    
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    progress.update(task, completed=1)
                
                if response:
                    console.print(f"[green]✅ Sukces ({duration:.2f}s)[/green]")
                    console.print(f"[dim]Odpowiedź: {str(response)[:100]}...[/dim]")
                else:
                    console.print("[yellow]⚠️  Pusta odpowiedź[/yellow]")
                    
            except Exception as e:
                console.print(f"[red]❌ Błąd: {e}[/red]")
    
    def show_server_configuration(self):
        """Pokaż konfigurację serwera."""
        console.print("\n[bold]⚙️  Konfiguracja serwera LM Studio[/bold]")
        
        config_tips = [
            "🌐 **Port:** 1234 (domyślny, nie zmieniaj bez potrzeby)",
            "🔧 **Max Tokens:** 4000-8000 (dla dłuższych tekstów)",
            "🎯 **Temperature:** 0.3-0.7 (0.3 = bardziej deterministyczne)",
            "⚡ **GPU Layers:** Maksymalna liczba dla GPU (jeśli masz)",
            "🧠 **Context Length:** 4096+ (dla dłuższych kontekstów)",
            "💾 **Thread Count:** Liczba rdzeni CPU - 1"
        ]
        
        for tip in config_tips:
            console.print(tip)
        
        console.print(Panel(
            "[bold]🚀 Optymalne ustawienia dla Jarvis EDU:[/bold]\n"
            "• Temperature: 0.3 (stabilne wyniki)\n"
            "• Max Tokens: 4000 (długie lekcje)\n"
            "• Top P: 0.9\n"
            "• Frequency Penalty: 0.1\n"
            "• Presence Penalty: 0.1",
            title="Zalecane ustawienia",
            border_style="green"
        ))
    
    def create_config_files(self):
        """Stwórz pliki konfiguracyjne."""
        console.print("\n[bold]📄 Tworzenie plików konfiguracyjnych[/bold]")
        
        # Aktualizuj config/auto-mode.yaml
        config_path = Path("config/auto-mode.yaml")
        if config_path.exists():
            console.print(f"[blue]✅ Plik konfiguracyjny istnieje: {config_path}[/blue]")
        else:
            console.print("[yellow]⚠️  Plik konfiguracyjny nie istnieje[/yellow]")
            
        # Stwórz przykładowy plik .env jeśli nie istnieje
        env_path = Path(".env")
        if not env_path.exists():
            env_content = """# Jarvis EDU Environment Configuration
# LM Studio Configuration
LLM__PROVIDER=lmstudio
LLM__ENDPOINT=http://localhost:1234/v1
LLM__MODEL=mistral-7b-instruct
LLM__TEMPERATURE=0.3
LLM__MAX_TOKENS=4000

# Other settings
DEBUG=false
LOG_LEVEL=INFO
"""
            env_path.write_text(env_content, encoding='utf-8')
            console.print(f"[green]✅ Utworzono plik .env: {env_path}[/green]")
        else:
            console.print(f"[blue]✅ Plik .env istnieje: {env_path}[/blue]")
    
    async def run_complete_setup(self):
        """Uruchom kompletny setup."""
        console.print(Panel.fit(
            "[bold blue]🤖 LM Studio Setup dla Jarvis EDU[/bold blue]\n"
            "Automatyczna konfiguracja i optymalizacja",
            border_style="blue"
        ))
        
        # 1. Sprawdź połączenie
        is_connected, models = await self.check_connection()
        
        if not is_connected:
            # 2. Pokaż przewodnik instalacji
            self.show_installation_guide()
            
            if Confirm.ask("\n❓ Czy chcesz zobaczyć rekomendowane modele?"):
                self.show_model_recommendations()
            
            console.print("\n[yellow]⏸️  Uruchom LM Studio i spróbuj ponownie[/yellow]")
            return
        
        # 3. Sprawdź modele
        if not models:
            console.print("\n[yellow]📥 Brak załadowanych modeli[/yellow]")
            self.show_model_recommendations()
            
            if Confirm.ask("\n❓ Czy chcesz zobaczyć konfigurację serwera?"):
                self.show_server_configuration()
            
            console.print("\n[yellow]⏸️  Załaduj model w LM Studio i spróbuj ponownie[/yellow]")
            return
        
        # 4. Pokaż załadowane modele
        console.print(f"\n[green]✅ Znaleziono {len(models)} załadowanych modeli:[/green]")
        for model in models[:3]:  # Pokaż pierwsze 3
            console.print(f"  • {model.get('id', 'Unknown')}")
        
        # 5. Test wydajności
        if Confirm.ask("\n❓ Czy chcesz przetestować wydajność modelu?"):
            await self.test_model_performance()
        
        # 6. Konfiguracja
        if Confirm.ask("\n❓ Czy chcesz zobaczyć optymalne ustawienia serwera?"):
            self.show_server_configuration()
        
        # 7. Pliki konfiguracyjne
        self.create_config_files()
        
        # 8. Podsumowanie
        console.print("\n" + "="*60)
        console.print("[bold green]🎉 LM Studio jest gotowy do pracy![/bold green]")
        console.print("="*60)
        console.print("[green]✅ Połączenie: OK[/green]")
        console.print(f"[green]✅ Modele: {len(models)} załadowanych[/green]")
        console.print("[green]✅ Konfiguracja: Gotowa[/green]")
        
        console.print("\n[bold]🚀 Następne kroki:[/bold]")
        console.print("1. 🔧 Uruchom: python test_connections.py")
        console.print("2. 🎯 Uruchom Jarvis: python start_jarvis.py")
        console.print("3. 📁 Wrzuć pliki do folderu obserwowanego")
        console.print("4. 🎉 Ciesz się automatycznymi lekcjami!")


async def main():
    """Main function."""
    configurator = LMStudioConfigurator()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "check":
            await configurator.check_connection()
        elif command == "install":
            configurator.show_installation_guide()
        elif command == "models":
            configurator.show_model_recommendations()
        elif command == "config":
            configurator.show_server_configuration()
        elif command == "test":
            await configurator.test_model_performance()
        else:
            console.print(f"[red]❌ Nieznana komenda: {command}[/red]")
            console.print("\n[bold]Dostępne komendy:[/bold]")
            console.print("• check - sprawdź połączenie")
            console.print("• install - przewodnik instalacji") 
            console.print("• models - rekomendowane modele")
            console.print("• config - konfiguracja serwera")
            console.print("• test - test wydajności")
    else:
        await configurator.run_complete_setup()


if __name__ == "__main__":
    asyncio.run(main()) 