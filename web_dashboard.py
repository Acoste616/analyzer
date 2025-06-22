#!/usr/bin/env python3
"""Web Dashboard dla Jarvis EDU Extractor"""

import streamlit as st
import pandas as pd
import json
import torch
import requests
from pathlib import Path
from datetime import datetime
import os
import time

# Page config
st.set_page_config(
    page_title="Jarvis EDU Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_jarvis_availability():
    """Sprawdzenie dostępności Jarvis ekstraktów"""
    try:
        # Test importów Jarvis
        from jarvis_edu.core.config import get_settings
        from jarvis_edu.extractors.enhanced_image_extractor import EnhancedImageExtractor
        from jarvis_edu.extractors.enhanced_video_extractor import EnhancedVideoExtractor
        from jarvis_edu.extractors.enhanced_audio_extractor import EnhancedAudioExtractor
        from jarvis_edu.processors.lm_studio import LMStudioProcessor
        
        # Test inicjalizacji
        settings = get_settings()
        image_extractor = EnhancedImageExtractor()
        video_extractor = EnhancedVideoExtractor()
        audio_extractor = EnhancedAudioExtractor()
        llm_processor = LMStudioProcessor()
        
        return True
    except Exception as e:
        st.error(f"Błąd Jarvis: {e}")
        return False

def check_system_status():
    """Sprawdzanie statusu systemu"""
    status = {
        'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        'gpu_available': False,
        'gpu_name': 'N/A',
        'gpu_memory': 'N/A',
        'gpu_memory_allocated': 0,
        'gpu_memory_reserved': 0,
        'gpu_memory_free': 0,
        'gpu_memory_total': 0,
        'gpu_memory_usage_percent': 0,
        'lm_studio_connected': False,
        'lm_studio_model': 'N/A',
        'media_folder_exists': False,
        'media_files_count': 0,
        'jarvis_available': False,
        'tesseract_available': False
    }
    
    # GPU check z detalami
    try:
        if torch.cuda.is_available():
            status['gpu_available'] = True
            status['gpu_name'] = torch.cuda.get_device_name(0)
            device_props = torch.cuda.get_device_properties(0)
            status['gpu_memory_total'] = device_props.total_memory / 1024**3
            status['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3
            status['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3
            status['gpu_memory_free'] = status['gpu_memory_total'] - status['gpu_memory_reserved']
            status['gpu_memory_usage_percent'] = (status['gpu_memory_reserved'] / status['gpu_memory_total']) * 100
            status['gpu_memory'] = f"{status['gpu_memory_total']:.1f} GB"
    except:
        pass
    
    # LM Studio check
    try:
        response = requests.get('http://127.0.0.1:1234/v1/models', timeout=3)
        if response.status_code == 200:
            models_data = response.json()
            status['lm_studio_connected'] = True
            if 'data' in models_data and models_data['data']:
                status['lm_studio_model'] = models_data['data'][0].get('id', 'Unknown')
    except:
        pass
    
    # Media folder check
    media_folder = Path("C:/Users/barto/OneDrive/Pulpit/mediapobrane/pobrane_media")
    if media_folder.exists():
        status['media_folder_exists'] = True
        files = list(media_folder.rglob("*"))
        status['media_files_count'] = len([f for f in files if f.is_file()])
    
    # Jarvis check
    status['jarvis_available'] = check_jarvis_availability()
    
    # Tesseract check
    try:
        import pytesseract
        from PIL import Image
        # Test małego obrazu
        test_img = Image.new('RGB', (50, 20), color='white')
        pytesseract.image_to_string(test_img)
        status['tesseract_available'] = True
    except:
        status['tesseract_available'] = False
    
    return status

def load_processing_results():
    """Ładowanie wyników przetwarzania z wszystkich raportów - MEGA ADVANCED SUPPORT"""
    results = []
    processing_type = "MOCK"
    jarvis_available = False
    latest_report_time = None
    report_source = "none"
    category_stats = {}
    auto_discovery_stats = {}
    
    # Lista wszystkich możliwych raportów
    report_files = [
        ("output/mega_advanced_report.json", "MEGA_ADVANCED"),
        ("output/advanced_content_report.json", "ADVANCED"),
        ("output/batch_processing_report.json", "BATCH"),
        ("output/quick_test_report.json", "QUICK")
    ]
    
    # Znajdź najnowszy raport
    for report_path_str, report_type in report_files:
        report_path = Path(report_path_str)
        if report_path.exists():
            try:
                report_time = report_path.stat().st_mtime
                
                # Użyj najnowszego raportu
                if latest_report_time is None or report_time > latest_report_time:
                    latest_report_time = report_time
                    
                    with open(report_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Clear previous results
                    results.clear()
                    category_stats.clear()
                    auto_discovery_stats.clear()
                    
                    # Load results
                    results.extend(data.get('results', []))
                    
                    # Determine processing type
                    processor_type = data.get('processor_type', 'UNKNOWN')
                    if "MEGA" in processor_type:
                        processing_type = "MEGA_ADVANCED"
                    elif "ADVANCED" in processor_type:
                        processing_type = "ADVANCED"
                    elif data.get('processing_type'):
                        processing_type = data.get('processing_type', 'MOCK')
                    else:
                        processing_type = "REAL"
                    
                    jarvis_available = data.get('jarvis_available', True)  # Assume true for newer reports
                    report_source = report_type
                    
                    # Load additional stats from mega/advanced reports
                    category_stats = data.get('category_statistics', {})
                    auto_discovery_stats = data.get('auto_discovery_stats', {})
                    
            except Exception as e:
                print(f"Error loading {report_path}: {e}")
                continue
    
    return results, processing_type, jarvis_available, report_source, category_stats, auto_discovery_stats

def gpu_memory_monitor():
    """Monitor pamięci GPU z real-time danymi"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'free': total - reserved,
            'total': total,
            'usage_percent': (reserved / total) * 100
        }
    return None

def main():
    """Główna funkcja dashboard"""
    
    # Header
    st.title("🧠 Jarvis EDU Extractor Dashboard")
    st.markdown("---")
    
    # Auto-refresh controls
    col1, col2 = st.columns([1, 4])
    with col1:
        auto_refresh = st.checkbox("🔄 Auto-refresh (30s)", value=False)
    with col2:
        if st.button("🔄 Odśwież teraz"):
            st.rerun()
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(1)  # Prevent too fast refresh
        st.rerun()
    
    # Sidebar
    st.sidebar.title("⚙️ Kontrola Systemu")
    
    # System status
    status = check_system_status()
    
    st.sidebar.markdown("### 📊 Status Systemu")
    
    # Python
    st.sidebar.metric("Python", status['python_version'])
    
    # GPU
    gpu_color = "🟢" if status['gpu_available'] else "🔴"
    st.sidebar.markdown(f"{gpu_color} **GPU:** {status['gpu_name']}")
    if status['gpu_available']:
        st.sidebar.text(f"VRAM: {status['gpu_memory']}")
        st.sidebar.text(f"Używane: {status['gpu_memory_usage_percent']:.1f}%")
    
    # LM Studio
    lm_color = "🟢" if status['lm_studio_connected'] else "🔴"
    st.sidebar.markdown(f"{lm_color} **LM Studio:** {'Połączony' if status['lm_studio_connected'] else 'Brak połączenia'}")
    if status['lm_studio_connected']:
        st.sidebar.text(f"Model: {status['lm_studio_model']}")
    
    # Jarvis
    jarvis_color = "🟢" if status['jarvis_available'] else "🔴"
    st.sidebar.markdown(f"{jarvis_color} **Jarvis:** {'Dostępny' if status['jarvis_available'] else 'Niedostępny'}")
    
    # Tesseract
    tesseract_color = "🟢" if status['tesseract_available'] else "🔴"
    st.sidebar.markdown(f"{tesseract_color} **Tesseract:** {'Dostępny' if status['tesseract_available'] else 'Niedostępny'}")
    
    # Media folder
    media_color = "🟢" if status['media_folder_exists'] else "🔴"
    st.sidebar.markdown(f"{media_color} **Folder mediów:** {'Dostępny' if status['media_folder_exists'] else 'Brak'}")
    if status['media_folder_exists']:
        st.sidebar.text(f"Pliki: {status['media_files_count']:,}")
    
    # GPU Memory monitoring widget
    if st.sidebar.button("🧹 Clear GPU Cache"):
        torch.cuda.empty_cache()
        st.sidebar.success("GPU cache cleared!")
        
    gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
    st.sidebar.progress(gpu_memory / gpu_total if gpu_total > 0 else 0)
    st.sidebar.caption(f"GPU: {gpu_memory:.1f}/{gpu_total:.1f} GB")
    
    # Main content
    col1, col2, col3 = st.columns(3)
    
    # System health calculation
    health_factors = [
        status['gpu_available'],
        status['lm_studio_connected'], 
        status['media_folder_exists'],
        status['jarvis_available'],
        status['tesseract_available']
    ]
    health_score = sum(health_factors) / len(health_factors) * 100
    
    with col1:
        health_status = "OPTIMAL" if health_score == 100 else "PARTIAL" if health_score >= 80 else "DEGRADED"
        st.metric(
            "🎯 System Status",
            health_status,
            delta=f"{health_score:.0f}% sprawny"
        )
    
    with col2:
        st.metric(
            "🖥️ GPU",
            status['gpu_name'] if status['gpu_available'] else "Niedostępny",
            delta=f"{status['gpu_memory_usage_percent']:.1f}% użycia" if status['gpu_available'] else "Brak CUDA"
        )
    
    with col3:
        st.metric(
            "📁 Media Files",
            f"{status['media_files_count']:,}" if status['media_folder_exists'] else "0",
            delta="Gotowe do przetwarzania" if status['media_files_count'] > 0 else "Brak plików"
        )
    
    # REAL-TIME GPU Memory Monitor
    if status['gpu_available']:
        st.markdown("### 🚀 Real-Time GPU Memory Monitor")
        gpu_mem = gpu_memory_monitor()
        
        if gpu_mem:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Allocated", f"{gpu_mem['allocated']:.2f} GB")
            with col2:
                st.metric("Reserved", f"{gpu_mem['reserved']:.2f} GB") 
            with col3:
                st.metric("Free", f"{gpu_mem['free']:.2f} GB")
            with col4:
                st.metric("Usage", f"{gpu_mem['usage_percent']:.1f}%")
            
            # Real-time progress bar
            progress_value = gpu_mem['reserved'] / gpu_mem['total']
            st.progress(progress_value)
            
            # Color-coded usage indicator
            if gpu_mem['usage_percent'] > 80:
                st.error(f"⚠️ Wysokie zużycie GPU Memory: {gpu_mem['usage_percent']:.1f}%")
            elif gpu_mem['usage_percent'] > 50:
                st.warning(f"🔶 Średnie zużycie GPU Memory: {gpu_mem['usage_percent']:.1f}%")
            else:
                st.success(f"✅ Normalne zużycie GPU Memory: {gpu_mem['usage_percent']:.1f}%")
    
    # Processing Results with MEGA ADVANCED detection
    st.markdown("### 📋 Wyniki Przetwarzania (MEGA Advanced System)")
    
    results, processing_type, jarvis_available, report_source, category_stats, auto_discovery_stats = load_processing_results()
    
    if results:
        # Advanced processing status indicators
        if processing_type == "MEGA_ADVANCED":
            st.success("🚀 Dane z MEGA ADVANCED PROCESSOR!")
            st.info(f"🔍 Source: {report_source} | Comprehensive Tagging + Auto-Discovery")
        elif processing_type == "ADVANCED":
            st.success("✨ Dane z ADVANCED PROCESSOR!")
            st.info(f"🔍 Source: {report_source} | Enhanced Tagging System")
        elif processing_type == "REAL":
            st.success("🎉 Dane z PRAWDZIWEGO przetwarzania AI!")
        else:
            st.warning("⚠️ Dane z MOCK/FALLBACK przetwarzania")
        
        if jarvis_available:
            st.success("✅ Jarvis ekstraktory: Aktywne")
        else:
            st.error("❌ Jarvis ekstraktory: Niedostępne")
        
        # Real-time file modification check
        mega_report_path = Path("output/mega_advanced_report.json")
        if mega_report_path.exists():
            mod_time = datetime.fromtimestamp(mega_report_path.stat().st_mtime)
            time_diff = datetime.now() - mod_time
            if time_diff.total_seconds() < 300:  # 5 minutes
                st.info(f"🕐 Ostatnia aktualizacja MEGA: {time_diff.total_seconds():.0f} sekund temu")
            else:
                st.info(f"🕐 Ostatnia aktualizacja MEGA: {mod_time.strftime('%H:%M:%S')}")
        elif Path("output/batch_processing_report.json").exists():
            batch_report_path = Path("output/batch_processing_report.json")
            mod_time = datetime.fromtimestamp(batch_report_path.stat().st_mtime)
            time_diff = datetime.now() - mod_time
            st.info(f"🕐 Ostatnia aktualizacja: {mod_time.strftime('%H:%M:%S')}")
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Przetworzone pliki", len(results))
        
        with col2:
            success_count = len([r for r in results if r.get('status') == 'success'])
            st.metric("Sukces", success_count)
        
        with col3:
            avg_time = sum(r.get('processing_time', 0) for r in results) / len(results) if results else 0
            st.metric("Średni czas", f"{avg_time:.2f}s")
        
        with col4:
            total_size = sum(r.get('file_size_mb', 0) for r in results) if results else 0
            st.metric("Łączny rozmiar", f"{total_size:.1f} MB")
        
        # MEGA ADVANCED FEATURES - Category Statistics & Auto-Discovery
        if category_stats or auto_discovery_stats:
            st.markdown("### 🎯 MEGA Advanced Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if category_stats:
                    st.markdown("#### 📂 Comprehensive Categories")
                    
                    # Create expandable category view
                    for main_category, subcategories in category_stats.items():
                        with st.expander(f"📂 {main_category} ({sum(subcategories.values())} plików)"):
                            for subcat, count in sorted(subcategories.items(), key=lambda x: x[1], reverse=True):
                                st.write(f"└── **{subcat}**: {count} plików")
            
            with col2:
                if auto_discovery_stats:
                    st.markdown("#### 🔍 Auto-Discovery Results")
                    
                    for discovery_type, items in auto_discovery_stats.items():
                        with st.expander(f"🆕 {discovery_type} ({len(items)} wykrytych)"):
                            for item in items[:10]:  # Show first 10 items
                                st.write(f"• {item}")
                            if len(items) > 10:
                                st.write(f"... i {len(items) - 10} więcej")
        
        # PRECISION TAGS from MEGA processor
        if processing_type in ["MEGA_ADVANCED", "ADVANCED"] and results:
            st.markdown("### 🏷️ Precision Tags Analysis")
            
            # Collect all precision tags
            all_precision_tags = []
            for result in results:
                tagging_analysis = result.get('tagging_analysis', {})
                precision_tags = tagging_analysis.get('precision_tags', [])
                all_precision_tags.extend(precision_tags)
            
            if all_precision_tags:
                # Count tag frequency
                tag_counts = {}
                for tag in all_precision_tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
                
                # Display top tags
                top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:15]
                
                col1, col2, col3 = st.columns(3)
                
                for i, (tag, count) in enumerate(top_tags):
                    col_index = i % 3
                    if col_index == 0:
                        col1.write(f"**{tag}**: {count}x")
                    elif col_index == 1:
                        col2.write(f"**{tag}**: {count}x")
                    else:
                        col3.write(f"**{tag}**: {count}x")
        
        # Results table - adapted for MEGA processor format
        if results:
            df = pd.DataFrame(results)
            
            # Handle different data formats
            if 'file_path' in df.columns:
                df['file_name'] = df['file_path'].apply(lambda x: Path(x).name if x else 'N/A')
            else:
                df['file_name'] = df.get('file_name', 'Unknown')
            
            df['status_icon'] = df.get('status', 'success').apply(lambda x: "✅" if x == 'success' else "❌")
            
            # Select columns that exist
            available_columns = ['status_icon', 'file_name', 'file_type']
            column_renames = {
                'status_icon': 'Status',
                'file_name': 'Nazwa pliku',
                'file_type': 'Typ'
            }
            
            if 'file_size_mb' in df.columns:
                available_columns.append('file_size_mb')
                column_renames['file_size_mb'] = 'Rozmiar (MB)'
            
            if 'processing_time' in df.columns:
                available_columns.append('processing_time')
                column_renames['processing_time'] = 'Czas (s)'
            
            if 'confidence' in df.columns:
                available_columns.append('confidence')
                column_renames['confidence'] = 'Confidence (%)'
            
            # Display table
            display_df = df[available_columns].rename(columns=column_renames)
            st.dataframe(display_df, use_container_width=True)
        
        # Show sample extracted content - MEGA processor compatible
        if st.button("📖 Pokaż próbki treści MEGA"):
            for i, result in enumerate(results[:3], 1):
                file_name = result.get('file_name', 'Unknown')
                if 'file_path' in result:
                    file_name = Path(result['file_path']).name
                
                with st.expander(f"Plik {i}: {file_name}"):
                    st.write(f"**Typ:** {result.get('file_type', 'N/A')}")
                    st.write(f"**Czas przetwarzania:** {result.get('processing_time', 0):.2f}s")
                    st.write(f"**Status:** {result.get('status', 'success')}")
                    st.write(f"**Confidence:** {result.get('confidence', 'N/A')}%")
                    
                    if result.get('error'):
                        st.error(f"**Błąd:** {result['error']}")
                    
                    # Handle different content field names
                    content = result.get('extracted_content') or result.get('extracted_text', '')
                    if content:
                        st.write("**Wyodrębniona treść:**")
                        st.text_area("Treść", value=content, height=100, 
                                   key=f"content_{i}", label_visibility="hidden")
                    
                    # AI Summary
                    ai_summary = result.get('ai_summary') or result.get('llm_summary', '')
                    if ai_summary:
                        st.write("**Podsumowanie AI:**")
                        st.text_area("Podsumowanie", value=ai_summary, height=100, 
                                   key=f"summary_{i}", label_visibility="hidden")
                    
                    # MEGA Advanced Features - show tagging analysis
                    if result.get('tagging_analysis') and processing_type in ["MEGA_ADVANCED", "ADVANCED"]:
                        st.write("**MEGA Tagging Analysis:**")
                        tagging = result['tagging_analysis']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Primary Category:** {tagging.get('primary_category', 'N/A')}")
                            st.write(f"**Primary Subcategory:** {tagging.get('primary_subcategory', 'N/A')}")
                            
                            # Show folder structure
                            folder_struct = tagging.get('folder_structure', {})
                            if folder_struct:
                                st.write(f"**Suggested Folder:** {folder_struct.get('path', 'N/A')}")
                        
                        with col2:
                            # Show precision tags
                            precision_tags = tagging.get('precision_tags', [])
                            if precision_tags:
                                st.write("**Precision Tags:**")
                                for tag in precision_tags[:5]:
                                    st.write(f"• {tag}")
                            
                            # Show auto-discovered topics
                            auto_discovered = tagging.get('auto_discovered', {})
                            if auto_discovered:
                                st.write("**Auto-Discovered:**")
                                for topic, items in auto_discovered.items():
                                    st.write(f"• {topic}: {', '.join(items[:3])}")
        
    else:
        st.info("Brak wyników przetwarzania. Uruchom test_batch_processing.py aby wygenerować dane.")
        
        if st.button("🚀 Uruchom prawdziwe przetwarzanie"):
            st.info("Uruchamianie test_batch_processing.py...")
            st.code("python test_batch_processing.py")
    
    # File Upload Section
    st.markdown("### 📤 Upload Plików")
    
    uploaded_files = st.file_uploader(
        "Wybierz pliki do przetwarzania",
        accept_multiple_files=True,
        type=['mp4', 'avi', 'mkv', 'mov', 'jpg', 'jpeg', 'png', 'bmp']
    )
    
    if uploaded_files:
        st.success(f"Wybrano {len(uploaded_files)} plików")
        
        for file in uploaded_files:
            st.write(f"📁 {file.name} ({file.size / 1024 / 1024:.1f} MB)")
        
        if st.button("🚀 Rozpocznij przetwarzanie"):
            st.info("Funkcja przetwarzania będzie dostępna w pełnej wersji")
            st.balloons()
    
    # Quick Actions
    st.markdown("### ⚡ Szybkie Akcje")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🧪 Test systemu"):
            st.info("Test systemu uruchomiony - sprawdź terminal")
            st.code("python test_system.py")
    
    with col2:
        if st.button("🚀 MEGA Processor"):
            st.info("MEGA ADVANCED Processor uruchomiony - sprawdź terminal")
            st.code("python mega_processor.py")
    
    with col3:
        if st.button("📊 Batch Processing"):
            st.info("Batch przetwarzanie uruchomione - sprawdź terminal")
            st.code("python test_batch_processing.py")
    
    with col4:
        if st.button("🧹 Wyczyść GPU"):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                st.success("Cache GPU wyczyszczony!")
                st.rerun()  # Refresh to show updated GPU memory
            else:
                st.warning("GPU niedostępny")
    
    # System Information
    with st.expander("🔧 Informacje systemowe"):
        st.write(f"**Python:** {status['python_version']}")
        st.write(f"**GPU:** {status['gpu_name']}")
        st.write(f"**VRAM:** {status['gpu_memory']}")
        st.write(f"**LM Studio:** {'✅ Połączony' if status['lm_studio_connected'] else '❌ Brak połączenia'}")
        st.write(f"**Model:** {status['lm_studio_model']}")
        st.write(f"**Jarvis:** {'✅ Dostępny' if status['jarvis_available'] else '❌ Niedostępny'}")
        st.write(f"**Tesseract:** {'✅ Dostępny' if status['tesseract_available'] else '❌ Niedostępny'}")
        st.write(f"**Pliki mediów:** {status['media_files_count']:,}")
        
        st.metric("System Health Score", f"{health_score:.0f}%")
        
        if health_score == 100:
            st.success("System w pełnej gotowości! 🎉")
        elif health_score >= 80:
            st.warning("System częściowo gotowy ⚠️")
        else:
            st.error("System wymaga napraw ❌")
    
    # Footer with current time
    st.markdown("---")
    st.markdown("🧠 **Jarvis EDU Extractor** - AI-powered educational content processing system")
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    st.markdown(f"*Aktualizacja: {current_time}* | *Auto-refresh: {'🟢 ON' if auto_refresh else '🔴 OFF'}*")

if __name__ == "__main__":
    main()
