#!/usr/bin/env python3
"""
Jarvis EDU Extractor - Startup Script
=====================================

This script starts the complete Jarvis EDU system:
- Checks LM Studio connection
- Starts folder watching
- Launches web dashboard
- Opens browser automatically

Usage: python start_jarvis.py
"""

import sys
import asyncio
import subprocess
import webbrowser
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import requests
    import uvicorn
    from rich.console import Console
    from rich.panel import Panel
    from jarvis_edu.pipeline.auto_processor import AutoProcessor
    from jarvis_edu.api.dashboard import create_dashboard_app
    from jarvis_edu.processors.lm_studio import LMStudioProcessor
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("💡 Install with: pip install -e .")
    sys.exit(1)

console = Console()


def check_lm_studio():
    """Check if LM Studio is running."""
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json().get("data", [])
            if models:
                model_name = models[0].get("id", "Unknown")
                console.print(f"[green]✅ LM Studio connected - Model: {model_name}[/green]")
                return True
            else:
                console.print("[yellow]⚠️  LM Studio connected but no models loaded[/yellow]")
                return False
    except:
        pass
    
    console.print("[red]❌ LM Studio not available[/red]")
    console.print("\n[bold]LM Studio Setup Instructions:[/bold]")
    console.print("1. Download LM Studio from: https://lmstudio.ai/")
    console.print("2. Install and launch LM Studio")
    console.print("3. Browse and download a model (recommended: mistral-7b-instruct)")
    console.print("4. Load the model in LM Studio")
    console.print("5. Make sure the server is running on localhost:1234")
    return False


def check_folders():
    """Check and create required folders."""
    folders = [
        "C:\\Users\\barto\\OneDrive\\Pulpit\\mediapobrane\\pobrane_media",
        "output",
        "logs",
        "data"
    ]
    
    for folder in folders:
        path = Path(folder)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            console.print(f"[blue]📁 Created folder: {folder}[/blue]")
        else:
            console.print(f"[green]✅ Folder exists: {folder}[/green]")


async def start_system():
    """Start the complete Jarvis EDU system."""
    
    # Display banner
    console.print(Panel.fit(
        "[bold blue]🧠 Jarvis EDU Extractor PRO[/bold blue]\n"
        "Advanced AI-powered educational content processor\n"
        "[dim]Starting system...[/dim]",
        border_style="blue"
    ))
    
    # Check prerequisites
    console.print("\n[bold]🔍 System Check[/bold]")
    lm_studio_ok = check_lm_studio()
    
    console.print("\n[bold]📁 Folder Setup[/bold]")
    check_folders()
    
    # Initialize components
    console.print("\n[bold]⚙️  Initializing Components[/bold]")
    try:
        processor = AutoProcessor()
        dashboard_app = create_dashboard_app()
        console.print("[green]✅ Components initialized[/green]")
    except Exception as e:
        console.print(f"[red]❌ Initialization failed: {e}[/red]")
        return
    
    # Start processor
    console.print("\n[bold]🚀 Starting Services[/bold]")
    try:
        await processor.start()
        console.print("[green]✅ File watcher started[/green]")
        console.print(f"[blue]📁 Watching: {processor.watch_folder}[/blue]")
        console.print(f"[blue]📤 Output: {processor.output_folder}[/blue]")
    except Exception as e:
        console.print(f"[red]❌ Failed to start processor: {e}[/red]")
        return
    
    # Start dashboard
    try:
        config = uvicorn.Config(
            dashboard_app,
            host="localhost",
            port=8000,
            log_level="warning",  # Reduce uvicorn noise
            access_log=False
        )
        server = uvicorn.Server(config)
        
        console.print("[green]✅ Dashboard server starting...[/green]")
        console.print("[blue]🌐 Dashboard: http://localhost:8000[/blue]")
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(3)
            try:
                webbrowser.open("http://localhost:8000")
                console.print("[green]🌐 Browser opened[/green]")
            except:
                console.print("[yellow]⚠️  Could not open browser automatically[/yellow]")
        
        import threading
        threading.Thread(target=open_browser, daemon=True).start()
        
        # Display status
        console.print("\n" + "="*60)
        console.print("[bold green]🎉 Jarvis EDU System Started![/bold green]")
        console.print("="*60)
        console.print("📊 Dashboard: http://localhost:8000")
        console.print("📁 Drop files into: C:\\Users\\barto\\OneDrive\\Pulpit\\mediapobrane\\pobrane_media")
        console.print("🤖 LM Studio: " + ("Connected" if lm_studio_ok else "Not available (fallback mode)"))
        console.print("\n[yellow]Press Ctrl+C to stop[/yellow]")
        console.print("="*60 + "\n")
        
        # Run server
        await server.serve()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]🛑 Shutdown requested...[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Dashboard failed: {e}[/red]")
    finally:
        processor.stop()
        console.print("[green]✅ System stopped cleanly[/green]")


def main():
    """Main entry point."""
    try:
        asyncio.run(start_system())
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Goodbye![/yellow]")
    except Exception as e:
        console.print(f"[red]💥 Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main() 