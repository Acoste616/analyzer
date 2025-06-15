# 🔄 Jarvis EDU Migration Guide

## Przegląd zmian

### ✅ **Główne ulepszenia w nowej wersji:**

1. **🧠 Inteligentna analiza treści** - System teraz rzeczywiście analizuje pliki zamiast tworzyć dummy lekcje
2. **🔧 Poprawki bazodanowe** - Rozwiązano problemy z NULL JSON columns
3. **📝 Lepszy logging** - Poprawiono format JSON i obsługę błędów
4. **🎯 Precyzyjne ekstraktory** - Dedykowane analizatory dla obrazów, wideo i audio

### 🆕 **Nowe funkcje:**

- **OCR dla obrazów** - Ekstrakcja tekstu z materiałów wizualnych
- **Inteligentne nazewnictwo** - Kończy się era `undefined_1923973051680772191`
- **Analiza typu treści** - Automatyczne rozpoznawanie materiałów edukacyjnych
- **Szacowanie czasu** - Realistyczne przewidywania długości lekcji
- **Kategoryzacja** - Automatyczne przypisywanie kategorii i tagów

## 🚀 Instrukcje migracji

### 1. **Backup istniejących danych**
```bash
# Zatrzymaj system
Ctrl+C w terminalu z Jarvis

# Skopiuj bazę danych (jeśli istnieje)
copy jarvis_edu.db jarvis_edu.db.backup
```

### 2. **Zainstaluj nowe zależności**
```bash
# Aktualizuj Python dependencies
pip install -e .

# Opcjonalnie: Zainstaluj OCR libraries
pip install easyocr pytesseract
```

### 3. **Uruchom migrację bazy**
```bash
# Automatyczna migracja NULL columns
python migrate_database.py
```

### 4. **Uruchom nowy system**
```bash
# Użyj Python 3.10 (nie 3.13!)
python start_jarvis.py
```

## 📊 **Porównanie: Przed vs Po**

### ❌ **PRZED (dummy system):**
```
Lekcja: undefined_1923973051680772191_photo_1_20250518
Automatycznie wygenerowana lekcja z pliku undefined_1923973051680772191_photo_1_20250518
Tags: #auto-generated #test
```

### ✅ **PO (inteligentna analiza):**
```
Podstawy programowania w Python
Materiał wideo (25 min): Wprowadzenie do zmiennych i funkcji w języku Python.
Zawiera praktyczne przykłady i ćwiczenia. (Transkrypcja: ~2150 słów)
Tags: #auto-generated #video #high-confidence #programowanie
Kategoria: Programowanie
Trudność: beginner
```

## 🎯 **Nowe możliwości systemu**

### **Inteligentne przetwarzanie plików:**

| Typ pliku | Metoda analizy | Przykład wyniku |
|-----------|----------------|-----------------|
| **📸 Obrazy** | OCR (EasyOCR/Tesseract) | Ekstrakcja tekstu z slajdów, notatek |
| **🎥 Wideo** | Smart preview + metadata | Analiza rozmiaru, szacowanie czasu |
| **🎵 Audio** | Analiza + typ zawartości | Rozpoznawanie podcastów, wykładów |

### **Automatyczna kategoryzacja:**
- 🔢 **Matematyka** - rozpoznaje słowa kluczowe
- 💻 **Programowanie** - wykrywa kod, języki programowania  
- 📚 **Języki** - identyfikuje materiały językowe
- 🔬 **Nauki ścisłe** - fizyka, chemia, biologia
- 🎨 **Ogólne** - gdy nie ma jasnej kategorii

### **Zarządzanie dużymi plikami:**
```
[125MB] Wykład o algorytmach sortowania
📊 ŚREDNI PLIK: Zalecane przetwarzanie partiami
⏱️ Szacowany czas przetwarzania: 5-20 minut
🔄 Opcje: Fragment/Partie/Pełna analiza
```

## ⚠️ **Ważne uwagi**

### **Kompatybilność Python:**
- ✅ **Python 3.10, 3.11, 3.12** 
- ❌ **Python 3.13** (jeszcze nieobsługiwane)

### **Performance dla dużych plików:**
- **< 100MB** - Pełna analiza automatyczna
- **100MB - 500MB** - Smart preview z opcjami
- **> 500MB** - Preview + przetwarzanie partiami

### **OCR Requirements (opcjonalne):**
```bash
# Dla lepszej analizy obrazów:
pip install easyocr          # Preferowane
pip install pytesseract      # Alternatywa
```

## 🔧 **Troubleshooting**

### **Problem:** `undefined_` nadal w nazwach
**Rozwiązanie:** Usuń stare pliki z folderu watch, system przetworzy je ponownie

### **Problem:** Import errors po aktualizacji  
**Rozwiązanie:** `pip install -e .` ponownie

### **Problem:** NULL metadata errors
**Rozwiązanie:** Uruchom `python migrate_database.py`

### **Problem:** System wolno przetwarza
**Rozwiązanie:** Sprawdź rozmiary plików, używaj podglądów dla dużych

## 🎉 **Co dalej?**

1. **Testuj nowy system** - Prześlij przykładowe pliki
2. **Sprawdź dashboard** - Zobacz inteligentne nazwy lekcji  
3. **Konfiguruj OCR** - Dla lepszej analizy obrazów
4. **Optymalizuj** - Dostosuj threshold dla dużych plików

## 📞 **Wsparcie**

W razie problemów:
1. Sprawdź logi w konsoli
2. Uruchom `python migrate_database.py`
3. Zrestartuj system
4. Zgłoś issue z szczegółami błędu

---

**🚀 Ciesz się nowym, inteligentnym systemem analizy treści!** 