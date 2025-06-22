#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test weryfikacji systemu Jarvis EDU Extractor
Sprawdza wszystkie komponenty i zależności
"""

import os
import sys
import subprocess
import platform
import json
import requests
from pathlib import Path
from datetime import datetime
import importlib
import torch

# Kolory dla terminala
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header():
    """Wyświetl nagłówek"""
    print("\n" + "="*60)
    print(f"{Colors.BOLD}🧠 JARVIS EDU EXTRACTOR - TEST SYSTEMU{Colors.RESET}")
    print("="*60)
    print(f"📅 Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💻 System: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print("="*60 + "\n")

def test_python_version():
    """Test wersji Pythona"""
    print(f"{Colors.BOLD}1. Weryfikacja wersji Pythona{Colors.RESET}")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"{Colors.GREEN}✅ Python {version.major}.{version.minor}.{version.micro} - OK{Colors.RESET}")
        if version.minor == 13:
            print(f"{Colors.YELLOW}⚠️  Python 3.13 może mieć problemy z niektórymi bibliotekami AI{Colors.RESET}")
        return True
    else:
        print(f"{Colors.RED}❌ Python {version.major}.{version.minor} - Za stara wersja (wymagane 3.10+){Colors.RESET}")
        return False

def test_gpu_cuda():
    """Test GPU i CUDA"""
    print(f"\n{Colors.BOLD}2. Weryfikacja GPU/CUDA{Colors.RESET}")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"{Colors.GREEN}✅ GPU: {gpu_name}{Colors.RESET}")
            print(f"{Colors.GREEN}✅ CUDA: {cuda_version}{Colors.RESET}")
            print(f"{Colors.GREEN}✅ VRAM: {memory:.1f} GB{Colors.RESET}")
            
            # Test alokacji pamięci
            try:
                test_tensor = torch.randn(1000, 1000).cuda()
                print(f"{Colors.GREEN}✅ Test alokacji GPU: OK{Colors.RESET}")
                del test_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"{Colors.RED}❌ Test alokacji GPU: {e}{Colors.RESET}")
            
            return True
        else:
            print(f"{Colors.RED}❌ GPU niedostępne - system będzie używał CPU{Colors.RESET}")
            return False
            
    except ImportError:
        print(f"{Colors.RED}❌ PyTorch nie jest zainstalowany{Colors.RESET}")
        return False

def test_lm_studio():
    """Test połączenia z LM Studio"""
    print(f"\n{Colors.BOLD}3. Weryfikacja LM Studio{Colors.RESET}")
    
    lm_studio_url = "http://127.0.0.1:1234"
    
    try:
        # Test podstawowego połączenia
        response = requests.get(f"{lm_studio_url}/v1/models", timeout=5)
        
        if response.status_code == 200:
            print(f"{Colors.GREEN}✅ LM Studio działa na {lm_studio_url}{Colors.RESET}")
            
            models = response.json().get("data", [])
            if models:
                print(f"{Colors.GREEN}✅ Załadowany model: {models[0]['id']}{Colors.RESET}")
            else:
                print(f"{Colors.YELLOW}⚠️  Brak załadowanych modeli w LM Studio{Colors.RESET}")
            
            # Test API completions
            test_prompt = {
                "model": models[0]['id'] if models else "test",
                "prompt": "Test",
                "max_tokens": 10,
                "temperature": 0.1
            }
            
            comp_response = requests.post(
                f"{lm_studio_url}/v1/completions",
                json=test_prompt,
                timeout=10
            )
            
            if comp_response.status_code == 200:
                print(f"{Colors.GREEN}✅ API completions działa poprawnie{Colors.RESET}")
                return True
            else:
                print(f"{Colors.YELLOW}⚠️  API completions zwróciło status: {comp_response.status_code}{Colors.RESET}")
                return True  # LM Studio działa, ale może być problem z modelem
                
        else:
            print(f"{Colors.RED}❌ LM Studio zwróciło status: {response.status_code}{Colors.RESET}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"{Colors.RED}❌ Nie można połączyć z LM Studio na {lm_studio_url}{Colors.RESET}")
        print(f"{Colors.YELLOW}   Upewnij się, że LM Studio jest uruchomione{Colors.RESET}")
        return False
    except Exception as e:
        print(f"{Colors.RED}❌ Błąd podczas testu LM Studio: {e}{Colors.RESET}")
        return False

def test_tesseract():
    """Test instalacji Tesseract OCR"""
    print(f"\n{Colors.BOLD}4. Weryfikacja Tesseract OCR{Colors.RESET}")
    
    try:
        # Test przez subprocess
        result = subprocess.run(
            ['tesseract', '--version'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"{Colors.GREEN}✅ {version_line}{Colors.RESET}")
            return True
        else:
            print(f"{Colors.RED}❌ Tesseract nie działa poprawnie{Colors.RESET}")
            return False
            
    except FileNotFoundError:
        print(f"{Colors.RED}❌ Tesseract nie jest zainstalowany{Colors.RESET}")
        print(f"{Colors.YELLOW}   Zainstaluj: https://github.com/UB-Mannheim/tesseract/wiki{Colors.RESET}")
        return False
    except Exception as e:
        print(f"{Colors.RED}❌ Błąd podczas testu Tesseract: {e}{Colors.RESET}")
        return False

def test_ffmpeg():
    """Test instalacji FFmpeg"""
    print(f"\n{Colors.BOLD}5. Weryfikacja FFmpeg{Colors.RESET}")
    
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"{Colors.GREEN}✅ {version_line}{Colors.RESET}")
            return True
        else:
            print(f"{Colors.RED}❌ FFmpeg nie działa poprawnie{Colors.RESET}")
            return False
            
    except FileNotFoundError:
        print(f"{Colors.RED}❌ FFmpeg nie jest zainstalowany{Colors.RESET}")
        print(f"{Colors.YELLOW}   Windows: choco install ffmpeg{Colors.RESET}")
        print(f"{Colors.YELLOW}   Lub pobierz z: https://ffmpeg.org/download.html{Colors.RESET}")
        return False
    except Exception as e:
        print(f"{Colors.RED}❌ Błąd podczas testu FFmpeg: {e}{Colors.RESET}")
        return False

def test_required_folders():
    """Test wymaganych folderów"""
    print(f"\n{Colors.BOLD}6. Weryfikacja struktury folderów{Colors.RESET}")
    
    required_folders = [
        Path("./data"),
        Path("./data/uploads"),
        Path("./data/outputs"),
        Path("./data/vector_db"),
        Path("./logs"),
        Path("./models"),
        Path(os.path.expanduser("~/.jarvis")),
        Path(os.path.expanduser("~/.jarvis/knowledge_base"))
    ]
    
    all_ok = True
    for folder in required_folders:
        if folder.exists():
            print(f"{Colors.GREEN}✅ {folder}{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}⚠️  {folder} - Tworzę...{Colors.RESET}")
            try:
                folder.mkdir(parents=True, exist_ok=True)
                print(f"{Colors.GREEN}   ✓ Utworzono{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.RED}   ✗ Błąd: {e}{Colors.RESET}")
                all_ok = False
    
    return all_ok

def test_python_packages():
    """Test kluczowych pakietów Python"""
    print(f"\n{Colors.BOLD}7. Weryfikacja pakietów Python{Colors.RESET}")
    
    packages = {
        "torch": "PyTorch",
        "transformers": "HuggingFace Transformers",
        "whisper": "OpenAI Whisper",
        "cv2": "OpenCV",
        "PIL": "Pillow",
        "chromadb": "ChromaDB",
        "streamlit": "Streamlit",
        "fastapi": "FastAPI",
        "easyocr": "EasyOCR",
        "moviepy": "MoviePy",
        "sentence_transformers": "Sentence Transformers"
    }
    
    results = {}
    for package, name in packages.items():
        try:
            if package == "cv2":
                import cv2
                version = cv2.__version__
            elif package == "PIL":
                from PIL import Image
                import PIL
                version = PIL.__version__
            else:
                module = importlib.import_module(package)
                version = getattr(module, "__version__", "unknown")
            
            print(f"{Colors.GREEN}✅ {name} ({version}){Colors.RESET}")
            results[package] = True
        except ImportError:
            print(f"{Colors.RED}❌ {name} - nie zainstalowany{Colors.RESET}")
            results[package] = False
        except Exception as e:
            print(f"{Colors.YELLOW}⚠️  {name} - błąd: {e}{Colors.RESET}")
            results[package] = False
    
    return all(results.values())

def test_jarvis_modules():
    """Test modułów Jarvis"""
    print(f"\n{Colors.BOLD}8. Weryfikacja modułów Jarvis{Colors.RESET}")
    
    # Dodaj ścieżkę do src
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    modules = [
        "jarvis_edu.core.config",
        "jarvis_edu.processors.lm_studio",
        "jarvis_edu.pipeline.auto_processor",
        "jarvis_edu.extractors.enhanced_image_extractor",
        "jarvis_edu.extractors.enhanced_video_extractor",
        "jarvis_edu.extractors.enhanced_audio_extractor",
        "jarvis_edu.storage.knowledge_base"
    ]
    
    all_ok = True
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"{Colors.GREEN}✅ {module}{Colors.RESET}")
        except ImportError as e:
            print(f"{Colors.RED}❌ {module} - {e}{Colors.RESET}")
            all_ok = False
        except Exception as e:
            print(f"{Colors.YELLOW}⚠️  {module} - błąd: {e}{Colors.RESET}")
            all_ok = False
    
    return all_ok

def test_env_file():
    """Test pliku .env"""
    print(f"\n{Colors.BOLD}9. Weryfikacja konfiguracji .env{Colors.RESET}")
    
    env_path = Path(".env")
    
    if env_path.exists():
        print(f"{Colors.GREEN}✅ Plik .env istnieje{Colors.RESET}")
        
        # Sprawdź kluczowe zmienne
        required_vars = [
            "DEVICE",
            "WHISPER_MODEL",
            "EMBEDDING_MODEL",
            "VECTOR_DB_PATH"
        ]
        
        with open(env_path, 'r') as f:
            content = f.read()
            
        for var in required_vars:
            if var in content:
                print(f"{Colors.GREEN}   ✓ {var}{Colors.RESET}")
            else:
                print(f"{Colors.YELLOW}   ⚠️ Brak {var}{Colors.RESET}")
        
        return True
    else:
        print(f"{Colors.YELLOW}⚠️  Brak pliku .env - tworzę domyślny...{Colors.RESET}")
        
        default_env = """# Jarvis EDU Extractor Configuration
DEVICE=cuda
WHISPER_MODEL=medium
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_DB_PATH=./data/vector_db
LM_STUDIO_URL=http://127.0.0.1:1234
BATCH_SIZE=3
MAX_FRAMES=10
LOG_LEVEL=INFO
"""
        
        try:
            with open(env_path, 'w') as f:
                f.write(default_env)
            print(f"{Colors.GREEN}   ✓ Utworzono domyślny .env{Colors.RESET}")
            return True
        except Exception as e:
            print(f"{Colors.RED}   ✗ Błąd tworzenia .env: {e}{Colors.RESET}")
            return False

def generate_summary(results):
    """Generuj podsumowanie testów"""
    print("\n" + "="*60)
    print(f"{Colors.BOLD}📊 PODSUMOWANIE TESTÓW{Colors.RESET}")
    print("="*60)
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    failed = total - passed
    
    print(f"\n{Colors.BOLD}Wyniki:{Colors.RESET}")
    print(f"  {Colors.GREEN}✅ Zaliczone: {passed}/{total}{Colors.RESET}")
    print(f"  {Colors.RED}❌ Nieudane: {failed}/{total}{Colors.RESET}")
    
    if failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 SYSTEM GOTOWY DO PRACY!{Colors.RESET}")
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  SYSTEM WYMAGA POPRAWEK{Colors.RESET}")
        print(f"\n{Colors.BOLD}Zalecane działania:{Colors.RESET}")
        
        if not results.get("lm_studio", True):
            print(f"  • Uruchom LM Studio i załaduj model")
        if not results.get("tesseract", True):
            print(f"  • Zainstaluj Tesseract OCR")
        if not results.get("ffmpeg", True):
            print(f"  • Zainstaluj FFmpeg")
        if not results.get("packages", True):
            print(f"  • Zainstaluj brakujące pakiety Python: pip install -r requirements.txt")
    
    # Zapisz raport
    report = {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "platform": platform.system(),
            "python": sys.version,
            "cuda_available": torch.cuda.is_available() if 'torch' in sys.modules else False
        },
        "tests": results,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "status": "ready" if failed == 0 else "needs_fixes"
        }
    }
    
    report_path = Path("./test_results/system_test_report.json")
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Raport zapisany: {report_path}")

def main():
    """Główna funkcja testowa"""
    print_header()
    
    results = {}
    
    # Uruchom testy
    results["python"] = test_python_version()
    results["gpu"] = test_gpu_cuda()
    results["lm_studio"] = test_lm_studio()
    results["tesseract"] = test_tesseract()
    results["ffmpeg"] = test_ffmpeg()
    results["folders"] = test_required_folders()
    results["packages"] = test_python_packages()
    results["jarvis"] = test_jarvis_modules()
    results["env"] = test_env_file()
    
    # Podsumowanie
    generate_summary(results)

if __name__ == "__main__":
    main() 