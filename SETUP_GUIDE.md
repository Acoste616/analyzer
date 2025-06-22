# 🧠 Jarvis EDU - Przewodnik Instalacji i Konfiguracji

Kompletny przewodnik instalacji wszystkich komponentów systemu Jarvis EDU na Windows.

## 📋 Spis treści

1. [Wymagania systemowe](#wymagania-systemowe)
2. [Instalacja Python](#instalacja-python)
3. [Instalacja FFmpeg](#instalacja-ffmpeg)
4. [Instalacja LM Studio](#instalacja-lm-studio)
5. [Konfiguracja projektu](#konfiguracja-projektu)
6. [Test systemu](#test-systemu)
7. [Rozwiązywanie problemów](#rozwiązywanie-problemów)

## 🖥️ Wymagania systemowe

- **System operacyjny:** Windows 10/11 (64-bit)
- **RAM:** 8 GB (minimum), 16 GB (zalecane)
- **Dysk:** 10 GB wolnego miejsca
- **GPU:** Opcjonalne, ale przyspiesza przetwarzanie AI

## 🐍 Instalacja Python

### Metoda 1: Microsoft Store (Zalecana)
1. Otwórz Microsoft Store
2. Wyszukaj "Python 3.11" lub "Python 3.12"
3. Kliknij "Pobierz" i zainstaluj
4. Python automatycznie zostanie dodany do PATH

### Metoda 2: Python.org
1. Przejdź na https://www.python.org/downloads/
2. Pobierz najnowszą wersję Python 3.11 lub 3.12
3. **WAŻNE:** Podczas instalacji zaznacz "Add Python to PATH"
4. Kliknij "Customize installation"
5. Zaznacz wszystkie opcje
6. Kliknij "Install"

### Weryfikacja instalacji Python
```powershell
python --version
# lub
py --version
```

Oczekiwany output: `Python 3.11.x` lub `Python 3.12.x`

## 🎬 Instalacja FFmpeg

FFmpeg jest wymagany do przetwarzania plików wideo i audio.

### Metoda 1: Winget (Windows 10/11)
```powershell
winget install ffmpeg
```

### Metoda 2: Chocolatey
1. Zainstaluj Chocolatey: https://chocolatey.org/install
2. Uruchom jako Administrator:
```powershell
choco install ffmpeg
```

### Metoda 3: Ręczna instalacja
1. Przejdź na https://ffmpeg.org/download.html#build-windows
2. Pobierz "Windows builds by BtbN"
3. Pobierz wersję "ffmpeg-master-latest-win64-gpl.zip"
4. Wypakuj do `C:\ffmpeg\`
5. Struktura powinna wyglądać: `C:\ffmpeg\bin\ffmpeg.exe`

### Dodanie FFmpeg do PATH
1. Otwórz "Ustawienia systemu" -> "Zaawansowane ustawienia systemu"
2. Kliknij "Zmienne środowiskowe"
3. W sekcji "Zmienne systemowe" znajdź "Path"
4. Kliknij "Edytuj" -> "Nowy"
5. Dodaj ścieżkę: `C:\ffmpeg\bin`
6. Kliknij "OK" i zrestartuj terminal

### Weryfikacja instalacji FFmpeg
```powershell
ffmpeg -version
```

## 🤖 Instalacja LM Studio

LM Studio to aplikacja do uruchamiania lokalnych modeli AI.

### Instalacja
1. Przejdź na https://lmstudio.ai/
2. Kliknij "Download for Windows"
3. Uruchom pobrany plik instalacyjny
4. Postępuj zgodnie z instrukcjami instalatora

### Konfiguracja LM Studio
1. **Uruchom LM Studio**
2. **Pobierz model:**
   - Przejdź do zakładki "Search" 
   - Wyszukaj jeden z rekomendowanych modeli:
     - `mistral-7b-instruct` (4.4 GB) - szybki i skuteczny
     - `llama-3.1-8b-instruct` (4.7 GB) - bardzo dobra jakość
     - `phi-3-mini-4k-instruct` (2.3 GB) - dla słabszych komputerów
   - Kliknij "Download" przy wybranym modelu

3. **Załaduj model:**
   - Przejdź do zakładki "My Models"
   - Znajdź pobrany model
   - Kliknij "Load"
   - Poczekaj na załadowanie

4. **Uruchom serwer:**
   - Przejdź do zakładki "Local Server"
   - Kliknij "Start Server"
   - Serwer powinien uruchomić się na `localhost:1234`

### Optymalne ustawienia LM Studio
- **Port:** 1234 (domyślny)
- **Max Tokens:** 4000
- **Temperature:** 0.3
- **Context Length:** 4096
- **GPU Layers:** Maksymalna liczba (jeśli masz GPU)

## 🔧 Konfiguracja projektu

### Instalacja zależności projektu
```powershell
# Przejdź do katalogu projektu
cd "C:\Users\barto\OneDrive\Pulpit\jarvis analyzer"

# Zainstaluj zależności
pip install -e .

# Lub ręcznie:
pip install openai-whisper pillow opencv-python numpy pandas requests aiohttp rich pydantic loguru pyyaml
```

### Tworzenie pliku .env
Skopiuj plik `env.example` na `.env` i dostosuj ustawienia:

```env
# LM Studio Configuration
LLM__PROVIDER=lmstudio
LLM__ENDPOINT=http://localhost:1234/v1
LLL__MODEL=mistral-7b-instruct
LLM__TEMPERATURE=0.3
LLM__MAX_TOKENS=4000

# Paths
INPUT_DIR=./input
OUTPUT_DIR=./output
```

### Konfiguracja folderów
```powershell
# Utwórz wymagane katalogi
mkdir -p input, output, logs, backup, cache
```

## 🧪 Test systemu

### Test 1: Sprawdzenie wszystkich komponentów
```powershell
python test_connections.py
```

Ten skrypt sprawdzi:
- ✅ Połączenie z LM Studio
- ✅ Dostępność FFmpeg
- ✅ Instalację Whisper
- ✅ Zależności Python
- ✅ Pliki konfiguracyjne

### Test 2: Konfiguracja LM Studio
```powershell
python setup_lm_studio.py
```

### Test 3: Uruchomienie systemu
```powershell
python start_jarvis.py
```

## 🔧 Permanent Paths

System automatycznie szuka narzędzi w następujących lokalizacjach:

### FFmpeg
- `C:\ffmpeg\bin\ffmpeg.exe`
- `C:\Program Files\ffmpeg\bin\ffmpeg.exe`
- `C:\tools\ffmpeg\bin\ffmpeg.exe`
- `%USERPROFILE%\ffmpeg\bin\ffmpeg.exe`
- PATH system

### Python
- Automatycznie wykrywany przez `sys.executable`
- Pip w `Scripts\pip.exe` względem Python

### Whisper Cache
- `%USERPROFILE%\.cache\whisper`

## 🚨 Rozwiązywanie problemów

### Problem: "Python nie jest rozpoznany"
**Rozwiązanie:**
1. Reinstaluj Python z opcją "Add to PATH"
2. Lub dodaj ręcznie do PATH: `C:\Users\[username]\AppData\Local\Programs\Python\Python311\`
3. Zrestartuj terminal

### Problem: "FFmpeg nie jest rozpoznany"
**Rozwiązanie:**
1. Sprawdź czy FFmpeg jest w `C:\ffmpeg\bin\`
2. Dodaj `C:\ffmpeg\bin\` do PATH
3. Zrestartuj terminal
4. Sprawdź: `ffmpeg -version`

### Problem: "LM Studio nie odpowiada"
**Rozwiązanie:**
1. Uruchom LM Studio
2. Załaduj model w zakładce "My Models"
3. Uruchom serwer w "Local Server"
4. Sprawdź czy port 1234 nie jest zajęty

### Problem: "Brak modelu AI"
**Rozwiązanie:**
1. W LM Studio przejdź do "Search"
2. Pobierz `mistral-7b-instruct`
3. Załaduj model w "My Models"
4. Upewnij się że serwer jest uruchomiony

### Problem: "Whisper nie działa"
**Rozwiązanie:**
```powershell
pip install --upgrade openai-whisper
# lub
pip install --force-reinstall openai-whisper
```

### Problem: "Import Error"
**Rozwiązanie:**
```powershell
# Zainstaluj wszystkie zależności
pip install -e .

# Lub ręcznie pojedynczo:
pip install openai-whisper
pip install pillow
pip install opencv-python
pip install rich
pip install pydantic
pip install loguru
```

## 📱 Kontakt i pomoc

W przypadku problemów:
1. Uruchom `python test_connections.py` i sprawdź wyniki
2. Sprawdź logi w folderze `logs/`
3. Sprawdź plik README.md dla dodatkowych informacji

## 🚀 Szybki start

Po zakończeniu instalacji:

1. **Sprawdź system:**
   ```powershell
   python test_connections.py
   ```

2. **Uruchom Jarvis:**
   ```powershell
   python start_jarvis.py
   ```

3. **Wrzuć pliki do folderu:**
   ```
   C:\Users\barto\OneDrive\Pulpit\mediapobrane\pobrane_media
   ```

4. **Sprawdź wyniki w:**
   ```
   C:\Users\barto\JarvisEDU\output
   ```

🎉 **Gotowe! System Jarvis EDU jest skonfigurowany i gotowy do pracy!** 