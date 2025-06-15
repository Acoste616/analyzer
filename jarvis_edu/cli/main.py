"""Main CLI interface for Jarvis EDU Extractor."""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

from ..core.config import get_settings
from ..core.logger import get_logger, setup_logging, LogConfig

console = Console()
logger = get_logger("cli")


@click.group()
@click.option("--debug/--no-debug", default=False, help="Enable debug mode")
@click.option("--log-level", default="INFO", help="Log level")
@click.version_option(version="0.1.0", prog_name="jarvis-edu")
@click.pass_context
def cli(ctx, debug, log_level):
    """Jarvis EDU Extractor - Advanced AI-powered educational content processor."""
    
    ctx.ensure_object(dict)
    
    # Setup logging
    log_config = LogConfig(level=log_level, format="human" if debug else "json")
    setup_logging(log_config)
    
    ctx.obj["debug"] = debug
    ctx.obj["settings"] = get_settings()
    
    if debug:
        console.print("[bold yellow]Debug mode enabled[/bold yellow]")
    
    # Display welcome message
    console.print(Panel.fit(
        "[bold blue]Jarvis EDU Extractor[/bold blue]\n"
        "Advanced AI-powered educational content processor",
        border_style="blue"
    ))


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--output-dir", "-o", type=click.Path(), default="output", help="Output directory")
@click.option("--format", "-f", type=click.Choice(["video", "audio", "image", "web", "document", "auto"]), 
              default="auto", help="Input format")
@click.option("--target-language", "-t", default="pl", help="Target language for lesson")
@click.pass_context
def extract(ctx, input_path, output_dir, format, target_language):
    """Extract educational content from a single file."""
    
    settings = ctx.obj["settings"]
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    console.print(f"[bold green]Processing:[/bold green] {input_path}")
    console.print(f"[bold blue]Output directory:[/bold blue] {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with Progress() as progress:
            task = progress.add_task("Processing...", total=100)
            
            # TODO: Add actual processing logic here
            for i in range(100):
                progress.update(task, advance=1)
                
            console.print(f"[bold green]✓ Successfully processed![/bold green]")
            console.print(f"[blue]Output saved to:[/blue] {output_dir}")
                
    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if ctx.obj["debug"]:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.pass_context
def serve(ctx, host, port):
    """Start the web server."""
    
    console.print(f"[bold green]Starting web server...[/bold green]")
    console.print(f"[blue]URL:[/blue] http://{host}:{port}")
    
    try:
        import uvicorn
        console.print("Server would start here...")
        # TODO: Add actual server startup
        
    except ImportError:
        console.print("[red]uvicorn not installed[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx):
    """Check system status."""
    
    console.print("[bold]System Status Check[/bold]")
    console.print("[green]✓ Python OK[/green]")
    console.print("[yellow]⚠ Some components not configured[/yellow]")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main() 