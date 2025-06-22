#!/usr/bin/env python3
"""
Test wszystkich połączeń systemowych Jarvis EDU
===============================================

Sprawdza:
- Połączenie z LM Studio
- Dostępność FFmpeg
- Whisper
- Wszystkie niezbędne komponenty
"""

import sys
import asyncio
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from jarvis_edu.processors.lm_studio import LMStudioProcessor
    from jarvis_edu.core.system_paths import get_system_path_manager, get_system_info
    from jarvis_edu.core.logger import get_logger
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're in the project directory and dependencies are installed")
    sys.exit(1)

console = Console()
logger = get_logger("test_connections")


async def test_lm_studio():
    """Test LM Studio connection."""
    console.print("\n[bold]🤖 Testing LM Studio Connection[/bold]")
    
    lm_studio = LMStudioProcessor()
    
    try:
        # Test basic connection
        is_available = await lm_studio.is_available()
        
        if is_available:
            console.print("[green]✅ LM Studio is running[/green]")
            
            # Get detailed connection info
            connection_info = await lm_studio.test_connection()
            
            if connection_info["status"] == "connected":
                console.print(f"[blue]📡 Endpoint: {connection_info['endpoint']}[/blue]")
                
                models = connection_info.get("models", [])
                if models:
                    console.print(f"[green]🧠 Available models: {len(models)}[/green]")
                    
                    # Display models table
                    table = Table(title="Available Models")
                    table.add_column("Model ID", style="cyan")
                    table.add_column("Object", style="magenta")
                    
                    for model in models[:5]:  # Show first 5 models
                        table.add_row(
                            model.get("id", "Unknown"),
                            model.get("object", "model")
                        )
                    
                    console.print(table)
                    
                    if len(models) > 5:
                        console.print(f"[dim]... and {len(models) - 5} more models[/dim]")
                else:
                    console.print("[yellow]⚠️  No models loaded in LM Studio[/yellow]")
                    console.print("[blue]💡 Load a model in LM Studio to enable AI processing[/blue]")
                
                # Test generation
                console.print("\n[bold]🧪 Testing AI Generation[/bold]")
                try:
                    test_response = await lm_studio.generate_lesson_from_prompt(
                        "Create a simple test response in JSON format with fields: title, summary, test_status"
                    )
                    
                    if test_response:
                        console.print("[green]✅ AI generation working[/green]")
                        console.print(f"[dim]Test response: {json.dumps(test_response, indent=2)[:200]}...[/dim]")
                    else:
                        console.print("[yellow]⚠️  AI generation returned empty response[/yellow]")
                        
                except Exception as e:
                    console.print(f"[red]❌ AI generation test failed: {e}[/red]")
            
            else:
                console.print(f"[red]❌ Connection error: {connection_info.get('message', 'Unknown error')}[/red]")
        
        else:
            console.print("[red]❌ LM Studio not available[/red]")
            console.print("\n[bold yellow]LM Studio Setup Guide:[/bold yellow]")
            console.print("1. 📥 Download from: https://lmstudio.ai/")
            console.print("2. 🚀 Launch LM Studio")
            console.print("3. 🧠 Download a model (recommended: mistral-7b-instruct or llama-3.1-8b)")
            console.print("4. ▶️  Load the model")
            console.print("5. 🌐 Start the server (default: localhost:1234)")
            console.print("6. 🔄 Re-run this test")
            
            return False
    
    except Exception as e:
        console.print(f"[red]❌ LM Studio test failed: {e}[/red]")
        return False
    
    return is_available


def test_system_tools():
    """Test system tools availability."""
    console.print("\n[bold]🔧 Testing System Tools[/bold]")
    
    path_manager = get_system_path_manager()
    system_info = get_system_info()
    
    # Create tools status table
    table = Table(title="System Tools Status")
    table.add_column("Tool", style="cyan", no_wrap=True)
    table.add_column("Status", style="bold")
    table.add_column("Path", style="dim")
    table.add_column("Working", style="bold")
    
    tools_to_check = ["ffmpeg", "ffprobe", "python", "pip"]
    
    for tool in tools_to_check:
        is_available = path_manager.is_tool_available(tool)
        is_working = system_info.get(f"{tool}_working", False)
        tool_path = path_manager.get_tool_path(tool)
        
        status = "✅ Available" if is_available else "❌ Missing"
        working_status = "✅ Yes" if is_working else "❌ No"
        path_display = tool_path[:50] + "..." if tool_path and len(tool_path) > 50 else (tool_path or "Not found")
        
        table.add_row(tool, status, path_display, working_status)
    
    console.print(table)
    
    # Show installation instructions for missing tools
    missing_instructions = path_manager.install_missing_tools()
    if missing_instructions:
        console.print("\n[bold red]📋 Installation Instructions for Missing Tools:[/bold red]")
        for tool, instructions in missing_instructions.items():
            console.print(Panel(instructions.strip(), title=f"Install {tool.upper()}", border_style="red"))
    
    return all(path_manager.is_tool_available(tool) for tool in ["ffmpeg", "ffprobe"])


def test_whisper_setup():
    """Test Whisper setup."""
    console.print("\n[bold]🎤 Testing Whisper Setup[/bold]")
    
    try:
        import whisper
        console.print("[green]✅ Whisper module imported[/green]")
        
        # Check available models
        available_models = whisper.available_models()
        console.print(f"[blue]🧠 Available Whisper models: {', '.join(available_models)}[/blue]")
        
        # Try to load base model
        console.print("[yellow]⏳ Testing base model loading...[/yellow]")
        try:
            model = whisper.load_model("base")
            console.print("[green]✅ Base model loaded successfully[/green]")
            
            # Test model info
            console.print(f"[blue]📊 Model dimensions: {model.dims}[/blue]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to load base model: {e}[/red]")
            console.print("[yellow]💡 This might be first-time setup - model will download automatically when needed[/yellow]")
            return False
            
    except ImportError:
        console.print("[red]❌ Whisper not installed[/red]")
        console.print("[blue]💡 Install with: pip install openai-whisper[/blue]")
        return False


def test_python_dependencies():
    """Test Python dependencies."""
    console.print("\n[bold]🐍 Testing Python Dependencies[/bold]")
    
    required_packages = [
        ("whisper", "openai-whisper"),
        ("PIL", "Pillow"),
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("requests", "requests"),
        ("aiohttp", "aiohttp"),
        ("rich", "rich"),
        ("pydantic", "pydantic"),
        ("loguru", "loguru"),
        ("yaml", "PyYAML"),
        ("pathlib", "pathlib"),
    ]
    
    table = Table(title="Python Dependencies")
    table.add_column("Package", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Install Command", style="dim")
    
    missing_packages = []
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            table.add_row(package_name, "[green]✅ Installed[/green]", "")
        except ImportError:
            table.add_row(package_name, "[red]❌ Missing[/red]", f"pip install {package_name}")
            missing_packages.append(package_name)
    
    console.print(table)
    
    if missing_packages:
        console.print(f"\n[red]📦 Missing packages: {', '.join(missing_packages)}[/red]")
        console.print(f"[blue]💡 Install all: pip install {' '.join(missing_packages)}[/blue]")
        return False
    
    return True


def test_configuration():
    """Test configuration files."""
    console.print("\n[bold]⚙️  Testing Configuration[/bold]")
    
    config_files = [
        "config/auto-mode.yaml",
        ".env"
    ]
    
    for config_file in config_files:
        config_path = Path(config_file)
        if config_path.exists():
            console.print(f"[green]✅ {config_file} found[/green]")
            
            if config_file.endswith('.yaml'):
                try:
                    import yaml
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                    console.print(f"[blue]📋 Config sections: {list(config_data.keys())}[/blue]")
                except Exception as e:
                    console.print(f"[yellow]⚠️  Config file has issues: {e}[/yellow]")
        else:
            console.print(f"[yellow]⚠️  {config_file} not found[/yellow]")
            if config_file == ".env":
                console.print(f"[blue]💡 Copy from env.example if needed[/blue]")


async def main():
    """Main test function."""
    console.print(Panel.fit(
        "[bold blue]🧠 Jarvis EDU System Test[/bold blue]\n"
        "Testing all system components and connections",
        border_style="blue"
    ))
    
    # System info
    system_info = get_system_info()
    console.print(f"[blue]🖥️  Platform: {system_info['platform']}[/blue]")
    console.print(f"[blue]🐍 Python: {system_info['python_version']}[/blue]")
    
    results = {}
    
    # Test all components
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # LM Studio test
        task1 = progress.add_task("Testing LM Studio...", total=1)
        results["lm_studio"] = await test_lm_studio()
        progress.update(task1, completed=1)
        
        # System tools test
        task2 = progress.add_task("Testing system tools...", total=1)
        results["system_tools"] = test_system_tools()
        progress.update(task2, completed=1)
        
        # Whisper test
        task3 = progress.add_task("Testing Whisper...", total=1)
        results["whisper"] = test_whisper_setup()
        progress.update(task3, completed=1)
        
        # Dependencies test
        task4 = progress.add_task("Testing dependencies...", total=1)
        results["dependencies"] = test_python_dependencies()
        progress.update(task4, completed=1)
        
        # Configuration test
        task5 = progress.add_task("Testing configuration...", total=1)
        test_configuration()
        progress.update(task5, completed=1)
    
    # Summary
    console.print("\n" + "="*60)
    console.print("[bold]📊 TEST SUMMARY[/bold]")
    console.print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    for test_name, result in results.items():
        status = "[green]✅ PASS[/green]" if result else "[red]❌ FAIL[/red]"
        console.print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    console.print(f"\n[bold]Result: {passed_tests}/{total_tests} tests passed[/bold]")
    
    if passed_tests == total_tests:
        console.print("[bold green]🎉 All systems ready! You can start using Jarvis EDU.[/bold green]")
        console.print("[blue]🚀 Run: python start_jarvis.py[/blue]")
    else:
        console.print("[bold yellow]⚠️  Some components need attention before full functionality.[/bold yellow]")
        
        if not results["lm_studio"]:
            console.print("[yellow]🤖 Without LM Studio: Basic extraction will work, but no AI-enhanced lessons[/yellow]")
        if not results["system_tools"]:
            console.print("[yellow]🔧 Without FFmpeg: Video/audio processing will be limited[/yellow]")
        if not results["whisper"]:
            console.print("[yellow]🎤 Without Whisper: No audio transcription[/yellow]")
    
    console.print("\n[dim]💡 For help, check README.md or run individual component setup[/dim]")


if __name__ == "__main__":
    asyncio.run(main()) 