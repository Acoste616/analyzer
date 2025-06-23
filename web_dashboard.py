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
import sqlite3
import subprocess
import sys

# Page config
st.set_page_config(
    page_title="Jarvis EDU Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

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
        'python_version': sys.version.split()[0],
        'gpu_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Niedostępny",
        'gpu_memory': f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.is_available() else "N/A",
        'gpu_memory_usage_percent': (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100) if torch.cuda.is_available() else 0,
        'lm_studio_connected': False,
        'lm_studio_model': 'Niedostępny',
        'media_folder_exists': False,
        'media_files_count': 0,
        'jarvis_available': check_jarvis_availability(),
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
    media_folder = Path("data/uploads")
    if media_folder.exists():
        status['media_folder_exists'] = True
        files = list(media_folder.rglob("*"))
        status['media_files_count'] = len([f for f in files if f.is_file()])
    
    # Tesseract check
    try:
        result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            status['tesseract_available'] = True
    except:
        pass
    
    return status

def get_database_connection():
    """Get SQLite database connection."""
    db_path = Path("data/jarvis_edu.db")
    if not db_path.exists():
        return None
    return sqlite3.connect(str(db_path))

def load_processed_files_from_db():
    """Load processed files from database."""
    conn = get_database_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, filepath, status, file_metadata, created_at, updated_at, error_message
            FROM file_entries 
            WHERE status = 'COMPLETED'
            ORDER BY updated_at DESC
            LIMIT 100
        """)
        
        results = []
        for row in cursor.fetchall():
            file_id, filepath, status, metadata, created_at, updated_at, error_msg = row
            
            # Parse metadata
            try:
                file_metadata = json.loads(metadata) if metadata else {}
            except:
                file_metadata = {}
            
            # Extract content if available
            extracted_content = file_metadata.get('extracted_content', '')
            ai_summary = file_metadata.get('ai_summary', '')
            processing_results = file_metadata.get('processing_results', {})
            
            result = {
                'id': file_id,
                'file_name': Path(filepath).name,
                'file_path': filepath,
                'status': status,
                'file_type': file_metadata.get('file_type', 'unknown'),
                'file_size_mb': file_metadata.get('file_size_mb', 0),
                'processing_time': file_metadata.get('processing_time', 0),
                'confidence': file_metadata.get('confidence', 0),
                'extracted_content': extracted_content,
                'ai_summary': ai_summary,
                'processing_results': processing_results,
                'created_at': created_at,
                'updated_at': updated_at,
                'error_message': error_msg,
                'metadata': file_metadata
            }
            results.append(result)
        
        conn.close()
        return results
        
    except Exception as e:
        st.error(f"Error loading processed files: {e}")
        conn.close()
        return []

def get_database_stats():
    """Get database statistics."""
    conn = get_database_connection()
    if not conn:
        return None
    
    try:
        cursor = conn.cursor()
        
        # Total files
        cursor.execute("SELECT COUNT(*) FROM file_entries")
        total = cursor.fetchone()[0]
        
        # Status breakdown
        cursor.execute("SELECT status, COUNT(*) FROM file_entries GROUP BY status")
        status_counts = dict(cursor.fetchall())
        
        # Recent activity (last 24 hours)
        cursor.execute("""
            SELECT COUNT(*) FROM file_entries 
            WHERE updated_at > datetime('now', '-1 day')
        """)
        recent_activity = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total': total,
            'completed': status_counts.get('COMPLETED', 0),
            'processing': status_counts.get('PROCESSING', 0), 
            'pending': status_counts.get('PENDING', 0),
            'failed': status_counts.get('FAILED', 0),
            'recent_activity': recent_activity,
            'success_rate': (status_counts.get('COMPLETED', 0) / total * 100) if total > 0 else 0
        }
        
    except Exception as e:
        st.error(f"Error getting database stats: {e}")
        conn.close()
        return None

def get_file_details(file_id):
    """Get detailed information about a specific file."""
    conn = get_database_connection()
    if not conn:
        return None
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, filepath, status, file_metadata, created_at, updated_at, error_message, content_type
            FROM file_entries 
            WHERE id = ?
        """, (file_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        file_id, filepath, status, metadata, created_at, updated_at, error_msg, content_type = row
        
        # Parse metadata
        try:
            file_metadata = json.loads(metadata) if metadata else {}
        except:
            file_metadata = {}
        
        conn.close()
        
        return {
            'id': file_id,
            'filepath': filepath,
            'file_name': Path(filepath).name,
            'status': status,
            'content_type': content_type,
            'created_at': created_at,
            'updated_at': updated_at,
            'error_message': error_msg,
            'metadata': file_metadata,
            'extracted_content': file_metadata.get('extracted_content', ''),
            'ai_summary': file_metadata.get('ai_summary', ''),
            'processing_results': file_metadata.get('processing_results', {}),
            'file_size_mb': file_metadata.get('file_size_mb', 0),
            'processing_time': file_metadata.get('processing_time', 0),
            'confidence': file_metadata.get('confidence', 0)
        }
        
    except Exception as e:
        st.error(f"Error getting file details: {e}")
        conn.close()
        return None

def load_processing_results():
    """Ładowanie wyników przetwarzania z bazy danych i JSON raportów"""
    # First try to load from database
    db_results = load_processed_files_from_db()
    
    if db_results:
        return db_results, "DATABASE", True, "SQLite Database", {}, {}
    
    # Fallback to JSON reports (existing code)
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

def load_processor_state():
    """Load real-time processor state from shared files"""
    processor_state = {
        'active_tasks': 0,
        'queued_tasks': 0,
        'processed_today': 0,
        'uptime_minutes': 0,
        'active_files': [],
        'running': False
    }
    
    processor_state_file = Path("output/processor_state.json")
    
    if processor_state_file.exists():
        try:
            with open(processor_state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
                processor_state.update(state_data)
                processor_state['running'] = True
        except Exception as e:
            st.sidebar.error(f"Error loading processor state: {e}")
    
    return processor_state

def load_processing_history():
    """Load processing history for charts"""
    history_file = Path("output/processing_history.json")
    
    if history_file.exists():
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
                return history
        except Exception as e:
            st.sidebar.error(f"Error loading processing history: {e}")
    
    return None

def main():
    """Główna funkcja dashboard"""
    
    # Auto-refresh session state initialization
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    # Header
    st.title("🧠 Jarvis EDU Extractor Dashboard")
    st.markdown("---")
    
    # Auto-refresh controls
    col1, col2 = st.columns([1, 4])
    with col1:
        auto_refresh = st.checkbox("🔄 Auto-refresh (5s)", value=True)
    with col2:
        if st.button("🔄 Odśwież teraz"):
            st.session_state.last_refresh = datetime.now()
            st.rerun()
    
    # Auto-refresh logic - every 5 seconds
    if auto_refresh:
        time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
        if time_since_refresh > 5:
            st.session_state.last_refresh = datetime.now()
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
    
    # Real-time processing status
    st.markdown("### 📊 Real-Time Processing Status")
    
    # Load processor state from continuous processor
    processor_state = load_processor_state()
    
    if processor_state['running']:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🔄 Active Tasks", processor_state.get('active_tasks', 0))
        
        with col2:
            st.metric("📋 Queued", processor_state.get('queued_tasks', 0))
        
        with col3:
            st.metric("✅ Processed Today", processor_state.get('processed_today', 0))
        
        with col4:
            uptime = processor_state.get('uptime_minutes', 0)
            st.metric("⏱️ Uptime", f"{uptime//60}h {uptime%60}m")
        
        # Current processing files
        active_files = processor_state.get('active_files', [])
        if active_files:
            st.markdown("#### 🎬 Currently Processing:")
            for file in active_files[:5]:
                progress = file.get('progress', 0)
                st.progress(progress / 100.0 if progress <= 100 else 1.0)
                elapsed_time = file.get('elapsed_time', 0)
                st.caption(f"{file['name']} - {file['type']} - {elapsed_time}s")
        else:
            st.info("💤 No files currently being processed")
        
        # Processing speed chart
        if st.checkbox("📈 Show processing speed chart"):
            history = load_processing_history()
            if history and 'entries' in history:
                try:
                    # Create DataFrame
                    df = pd.DataFrame(history['entries'])
                    if not df.empty and 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.set_index('timestamp')
                        
                        # Files per hour chart
                        hourly = df.resample('1H').size()
                        if not hourly.empty:
                            st.line_chart(hourly, use_container_width=True)
                            st.caption("Files processed per hour")
                        else:
                            st.info("No hourly data available yet")
                    else:
                        st.info("No processing history data available")
                except Exception as e:
                    st.error(f"Error creating chart: {e}")
            else:
                st.info("No processing history available")
        
        # Live processor logs (if available)
        if st.checkbox("📄 Show live processor logs"):
            log_file = Path("logs/jarvis_continuous.log")
            if log_file.exists():
                try:
                    # Read last 10 lines
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        last_lines = lines[-10:] if len(lines) > 10 else lines
                    
                    st.text_area("Recent Processor Logs", 
                               value=''.join(last_lines), 
                               height=200, 
                               key="processor_logs")
                except Exception as e:
                    st.error(f"Error reading logs: {e}")
            else:
                st.info("Log file not found")
    else:
        st.info("🔄 Processor not running or state not available")
        
        if st.button("🚀 Start Continuous Processor"):
            st.info("To start the processor, run: python start_jarvis.py")
            st.code("python start_jarvis.py")
    
    # Database Statistics
    st.markdown("### 🗄️ Database Statistics")
    
    db_stats = get_database_stats()
    if db_stats:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📁 Total Files", f"{db_stats['total']:,}")
        
        with col2:
            st.metric("✅ Completed", f"{db_stats['completed']:,}")
        
        with col3:
            st.metric("🔄 Processing", f"{db_stats['processing']:,}")
        
        with col4:
            st.metric("📋 Pending", f"{db_stats['pending']:,}")
        
        with col5:
            st.metric("🎯 Success Rate", f"{db_stats['success_rate']:.1f}%")
        
        # Progress bar for completion
        if db_stats['total'] > 0:
            progress = db_stats['completed'] / db_stats['total']
            st.progress(progress)
            st.caption(f"Progress: {db_stats['completed']}/{db_stats['total']} files completed")
    else:
        st.info("Database not available or empty")
    
    # Processed Files Browser
    st.markdown("### 📂 Przeglądarka Przetworzonych Plików")
    
    # Load processed files from database
    processed_files = load_processed_files_from_db()
    
    if processed_files:
        st.success(f"✅ Znaleziono {len(processed_files)} przetworzonych plików w bazie danych")
        
        # Search and filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_term = st.text_input("🔍 Szukaj pliku", placeholder="Wprowadź nazwę pliku...")
        
        with col2:
            file_types = list(set([f.get('file_type', 'unknown') for f in processed_files]))
            selected_type = st.selectbox("📄 Typ pliku", ["Wszystkie"] + file_types)
        
        with col3:
            sort_by = st.selectbox("📊 Sortuj według", ["Najnowsze", "Nazwa", "Rozmiar", "Czas przetwarzania"])
        
        # Filter files
        filtered_files = processed_files
        
        if search_term:
            filtered_files = [f for f in filtered_files if search_term.lower() in f['file_name'].lower()]
        
        if selected_type != "Wszystkie":
            filtered_files = [f for f in filtered_files if f.get('file_type') == selected_type]
        
        # Sort files
        if sort_by == "Nazwa":
            filtered_files = sorted(filtered_files, key=lambda x: x['file_name'])
        elif sort_by == "Rozmiar":
            filtered_files = sorted(filtered_files, key=lambda x: x.get('file_size_mb', 0), reverse=True)
        elif sort_by == "Czas przetwarzania":
            filtered_files = sorted(filtered_files, key=lambda x: x.get('processing_time', 0), reverse=True)
        # Default: Najnowsze (already sorted by updated_at DESC)
        
        st.info(f"Pokazuję {len(filtered_files)} z {len(processed_files)} plików")
        
        # Files display with expandable details
        if filtered_files:
            # Show files in batches for better performance
            files_per_page = 10
            page = st.selectbox("📄 Strona", range(1, (len(filtered_files) // files_per_page) + 2))
            start_idx = (page - 1) * files_per_page
            end_idx = start_idx + files_per_page
            
            current_files = filtered_files[start_idx:end_idx]
            
            for i, file_info in enumerate(current_files):
                with st.expander(f"📄 {file_info['file_name']} ({file_info.get('file_type', 'unknown')})"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**📁 Ścieżka:** {file_info['file_path']}")
                        st.write(f"**📊 Rozmiar:** {file_info.get('file_size_mb', 0):.2f} MB")
                        st.write(f"**⏱️ Czas przetwarzania:** {file_info.get('processing_time', 0):.2f}s")
                        st.write(f"**🎯 Confidence:** {file_info.get('confidence', 0)}%")
                        st.write(f"**📅 Zaktualizowano:** {file_info.get('updated_at', 'N/A')}")
                    
                    with col2:
                        st.write(f"**🆔 ID:** {file_info['id']}")
                        st.write(f"**✅ Status:** {file_info['status']}")
                        
                        # Quick actions
                        if st.button(f"📖 Pokaż szczegóły", key=f"details_{file_info['id']}"):
                            st.session_state[f"show_details_{file_info['id']}"] = True
                    
                    # Show detailed content if requested
                    if st.session_state.get(f"show_details_{file_info['id']}", False):
                        st.markdown("---")
                        
                        # Extracted content
                        if file_info.get('extracted_content'):
                            st.write("**📄 Wyodrębniona treść:**")
                            content = file_info['extracted_content']
                            if len(content) > 500:
                                st.text_area(
                                    "Treść",
                                    value=content[:500] + "...",
                                    height=150,
                                    key=f"content_short_{file_info['id']}",
                                    label_visibility="hidden"
                                )
                                if st.button(f"Pokaż pełną treść", key=f"full_content_{file_info['id']}"):
                                    st.text_area(
                                        "Pełna treść",
                                        value=content,
                                        height=300,
                                        key=f"content_full_{file_info['id']}",
                                        label_visibility="hidden"
                                    )
                            else:
                                st.text_area(
                                    "Treść",
                                    value=content,
                                    height=150,
                                    key=f"content_{file_info['id']}",
                                    label_visibility="hidden"
                                )
                        
                        # AI Summary
                        if file_info.get('ai_summary'):
                            st.write("**🤖 Podsumowanie AI:**")
                            st.text_area(
                                "Podsumowanie",
                                value=file_info['ai_summary'],
                                height=100,
                                key=f"summary_{file_info['id']}",
                                label_visibility="hidden"
                            )
                        
                        # Processing results
                        if file_info.get('processing_results'):
                            st.write("**⚙️ Wyniki przetwarzania:**")
                            st.json(file_info['processing_results'])
                        
                        # Hide details button
                        if st.button(f"🔼 Ukryj szczegóły", key=f"hide_{file_info['id']}"):
                            st.session_state[f"show_details_{file_info['id']}"] = False
                            st.rerun()
            
            # Navigation info
            total_pages = (len(filtered_files) // files_per_page) + 1
            st.info(f"Strona {page} z {total_pages} | Pliki {start_idx + 1}-{min(end_idx, len(filtered_files))} z {len(filtered_files)}")
            
        else:
            st.info("Brak plików spełniających kryteria wyszukiwania")
    
    else:
        st.info("Brak przetworzonych plików w bazie danych")
        st.write("💡 **Wskazówka:** Uruchom ciągły processor aby przetworzyć pliki:")
        st.code("python start_jarvis.py")
    
    # Processing Results with MEGA ADVANCED detection (fallback)
    st.markdown("### 📋 Wyniki Przetwarzania (Legacy/Fallback)")
    
    results, processing_type, jarvis_available, report_source, category_stats, auto_discovery_stats = load_processing_results()
    
    if results:
        # Advanced processing status indicators
        if processing_type == "DATABASE":
            st.success("🗄️ Dane z bazy danych SQLite!")
            st.info(f"🔍 Source: {report_source} | Real-time Database")
        elif processing_type == "MEGA_ADVANCED":
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
