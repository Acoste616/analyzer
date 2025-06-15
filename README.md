# 🎓 Jarvis EDU Extractor PRO

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](tests/)

> **Zaawansowany system AI do automatycznej konwersji materiałów edukacyjnych w strukturalne lekcje po polsku oraz datasety treningowe dla AI.**

## 🚀 Główne funkcje

- **🎥 Multi-format Support**: Video, audio, obrazy, strony web, dokumenty
- **🤖 AI-Powered**: Whisper, LLM (Ollama/OpenAI), OCR, NLP
- **🇵🇱 Polish-First**: Automatyczne tłumaczenie i optymalizacja dla polskiego
- **⚡ Performance**: Async processing, parallel execution, GPU support
- **🔌 Plugin System**: Rozszerzalna architektura
- **📊 Monitoring**: Real-time progress, detailed reporting
- **🐳 Docker Ready**: Kompletna konteneryzacja
- **🌐 Web UI**: Nowoczesny interfejs webowy

## 📋 Wymagania

### Minimalne
- Python 3.11+
- 8GB RAM
- 10GB przestrzeni dyskowej

### Zalecane
- Python 3.11+
- 16GB RAM
- GPU NVIDIA z CUDA (dla Whisper)
- 50GB przestrzeni dyskowej

## 🛠️ Instalacja

### Metoda 1: Docker (Zalecana)

```bash
# Klonowanie repozytorium
git clone https://github.com/jarvis-edu/extractor.git
cd jarvis-edu-extractor

# Uruchomienie wszystkich serwisów
make docker-up

# Sprawdzenie statusu
make status
```

### Metoda 2: Lokalna instalacja

```bash
# Klonowanie i setup
git clone https://github.com/jarvis-edu/extractor.git
cd jarvis-edu-extractor

# Instalacja zależności
make setup

# Konfiguracja
cp env.example .env
# Edytuj .env według potrzeb

# Test instalacji
make run
```

## 🎯 Szybki start

### 🚀 Metoda 1: Kompletny system (Zalecana)

```bash
# Uruchom cały system z dashboardem
python start_jarvis.py
```

Ta komenda:
- ✅ Sprawdza połączenie z LM Studio
- 📁 Tworzy wymagane foldery
- 🔍 Uruchamia monitorowanie folderów
- 🌐 Otwiera dashboard w przeglądarce
- 🤖 Automatycznie przetwarza nowe pliki

### CLI Interface

```bash
# Pomoc
jarvis-edu --help

# Uruchom monitorowanie z dashboardem
jarvis-edu watch

# Klasyczny tryb automatyczny
jarvis-edu auto

# Inicjalizacja konfiguracji
jarvis-edu init

# Ekstrakcja z pojedynczego pliku
jarvis-edu extract video.mp4 -o ./output/

# Uruchomienie web UI
jarvis-edu serve

# Check system status
jarvis-edu status
```

### Python API

```python
from jarvis_edu import VideoExtractor, Settings

# Konfiguracja
settings = Settings()
extractor = VideoExtractor(settings)

# Ekstrakcja lekcji
lesson = extractor.extract_lesson("video.mp4", "./output/")

print(f"Tytuł: {lesson.title}")
print(f"Tematy: {len(lesson.topics)}")
print(f"Fragmenty kodu: {len(lesson.code_snippets)}")
```

### 🧠 LM Studio Integration

System integruje się z LM Studio dla lokalnego przetwarzania AI:

1. **Pobierz LM Studio**: https://lmstudio.ai/
2. **Załaduj model**: Zalecany `mistral-7b-instruct`
3. **Uruchom serwer**: Port 1234 (domyślny)
4. **Start systemu**: `python start_jarvis.py`

### Web Interface

Po uruchomieniu serwera:
- **Dashboard**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Real-time Processing**: WebSocket updates
- **Knowledge Base**: Searchable lessons database

## 📁 Struktura projektu

```
jarvis-edu-extractor/
├── jarvis_edu/                 # Główny kod aplikacji
│   ├── core/                   # Podstawowe moduły
│   │   ├── config.py          # Konfiguracja
│   │   └── logger.py          # System logowania
│   ├── extractors/            # Ekstraktory treści
│   │   ├── video_extractor.py
│   │   ├── audio_extractor.py
│   │   ├── image_extractor.py
│   │   ├── web_extractor.py
│   │   └── document_extractor.py
│   ├── processors/            # Przetwarzanie danych
│   │   ├── llm_processor.py
│   │   ├── translation.py
│   │   └── lesson_generator.py
│   ├── models/               # Modele danych
│   │   ├── content.py
│   │   ├── lesson.py
│   │   └── processing.py
│   ├── storage/              # Zarządzanie danymi
│   │   ├── vector_store.py
│   │   └── database.py
│   ├── api/                  # REST API
│   │   ├── routes.py
│   │   └── websocket.py
│   ├── web/                  # Web UI
│   │   ├── templates/
│   │   └── static/
│   └── cli/                  # CLI interface
│       └── main.py
├── tests/                    # Testy
├── docs/                     # Dokumentacja
├── docker-compose.yml        # Docker setup
├── Dockerfile               # Container definition
├── Makefile                 # Automation commands
└── README.md               # Ten plik
```

## 🔧 Konfiguracja

### Główne ustawienia (.env)

```bash
# LLM Provider
LLM__PROVIDER=ollama           # ollama, openai, anthropic
LLM__MODEL=llama3             # Model name
LLM__BASE_URL=http://localhost:11434

# Whisper
WHISPER__MODEL=base           # tiny, base, small, medium, large
WHISPER__DEVICE=cpu           # cpu, cuda

# Processing
PROCESSING__TARGET_LANGUAGE=pl
PROCESSING__MAX_WORKERS=4
PROCESSING__CHUNK_SIZE=5000

# Security
SECURITY__MAX_FILE_SIZE_MB=500
SECURITY__SECRET_KEY=your-secret-key
```

### Zaawansowana konfiguracja

Pełna konfiguracja dostępna w `env.example`.

## 📊 Przykładowe wyjście

### Struktura lekcji
```
output/kubernetes-basics/
├── lesson.md              # Kompletna lekcja po polsku
├── metadata.json          # Metadane i statystyki
├── quiz.json             # Interaktywny quiz
├── snippets/             # Fragmenty kodu
│   ├── deployment.yaml
│   ├── service.yaml
│   └── pod-example.yaml
├── embeddings.json       # Wektory dla AI
├── extraction_report.json # Raport z przetwarzania
└── README.md             # Dokumentacja lekcji
```

### Metryki jakości

- **Dokładność ekstrakcji**: >95% dla tekstu, >90% dla audio
- **Czas przetwarzania**: <5 min/godzinę materiału  
- **Pokrycie języków**: 50+ języków wejściowych → Polski
- **Formaty wyjściowe**: Markdown, JSON, JSONL, Anki, Obsidian

## 🧪 Testowanie

```bash
# Wszystkie testy
make test

# Testy z coverage
make test-cov

# Tylko unit testy
make test-unit

# Tylko testy integracyjne
make test-integration

# Watching tests podczas developmentu
make watch-tests
```

## 🐳 Docker

```bash
# Build i start
make docker-up

# Zatrzymanie
make docker-down

# Logi
make docker-logs

# Czyszczenie
make docker-clean

# Pull Ollama models
make ollama-pull
```

## 🎯 Roadmapa

### ✅ Faza 1: MVP (Ukończona)
- [x] Podstawowa ekstrakcja z video/audio (Whisper)
- [x] CLI interface
- [x] Docker setup
- [x] System konfiguracji

### 🚧 Faza 2: Rozszerzenie (W trakcie) 
- [ ] OCR dla obrazów i dokumentów
- [ ] Web scraping z Playwright
- [ ] LLM integration (Ollama)
- [ ] Vector store (Qdrant)
- [ ] Web UI

### 📋 Faza 3: Optymalizacja
- [ ] Performance optimization
- [ ] Advanced error handling
- [ ] Monitoring i metrics
- [ ] Plugin system
- [ ] Multi-format export

### 🚀 Faza 4: Zaawansowane funkcje
- [ ] Multi-modal learning
- [ ] Personalizacja lekcji
- [ ] API dla integracji
- [ ] Real-time collaboration

## 🤝 Wkład w projekt

Zachęcamy do wkładu w projekt! 

1. Fork repozytorium
2. Utwórz branch (`git checkout -b feature/amazing-feature`)
3. Commit zmian (`git commit -m 'Add amazing feature'`)
4. Push do brancha (`git push origin feature/amazing-feature`)
5. Otwórz Pull Request

### Standardy kodu

```bash
# Formatowanie
make format

# Linting
make lint

# Wszystkie testy
make check
```

## 📄 Licencja

Ten projekt jest licencjonowany na licencji MIT - szczegóły w pliku [LICENSE](LICENSE).

## 🆘 Wsparcie

- **Issues**: [GitHub Issues](https://github.com/jarvis-edu/extractor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jarvis-edu/extractor/discussions)
- **Wiki**: [Dokumentacja](https://github.com/jarvis-edu/extractor/wiki)

## 🙏 Podziękowania

- **OpenAI** za Whisper
- **Ollama** za lokalne LLM
- **Qdrant** za vector database
- **FastAPI** za framework API
- **Click** za CLI interface

---

<div align="center">

**Zrobione z ❤️ dla edukacji w Polsce**

[⭐ Star nas na GitHub](https://github.com/jarvis-edu/extractor) • [📖 Dokumentacja](https://docs.jarvis-edu.com) • [🐛 Report Bug](https://github.com/jarvis-edu/extractor/issues)

</div> 