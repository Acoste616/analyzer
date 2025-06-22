"""System paths configuration for Windows."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass

from ..core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SystemPaths:
    """System paths configuration."""
    
    # FFmpeg paths (common Windows locations)
    ffmpeg_paths: List[str] = None
    ffprobe_paths: List[str] = None
    
    # Whisper model cache
    whisper_cache_dir: str = None
    
    # Python executable paths
    python_executable: str = None
    pip_executable: str = None
    
    # System tools
    system_tools: Dict[str, str] = None
    
    def __post_init__(self):
        if self.ffmpeg_paths is None:
            self.ffmpeg_paths = [
                "C:\\ffmpeg\\bin\\ffmpeg.exe",
                "C:\\ffmpeg\\ffmpeg.exe", 
                "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
                "C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe",
                "C:\\tools\\ffmpeg\\bin\\ffmpeg.exe",
                "C:\\Users\\Public\\ffmpeg\\bin\\ffmpeg.exe",
                os.path.expanduser("~\\ffmpeg\\bin\\ffmpeg.exe"),
                os.path.expanduser("~\\Desktop\\ffmpeg\\bin\\ffmpeg.exe"),
                os.path.expanduser("~\\Downloads\\ffmpeg\\bin\\ffmpeg.exe"),
                # Check PATH
                "ffmpeg.exe",
                "ffmpeg"
            ]
        
        if self.ffprobe_paths is None:
            self.ffprobe_paths = [
                "C:\\ffmpeg\\bin\\ffprobe.exe",
                "C:\\ffmpeg\\ffprobe.exe",
                "C:\\Program Files\\ffmpeg\\bin\\ffprobe.exe", 
                "C:\\Program Files (x86)\\ffmpeg\\bin\\ffprobe.exe",
                "C:\\tools\\ffmpeg\\bin\\ffprobe.exe",
                "C:\\Users\\Public\\ffmpeg\\bin\\ffprobe.exe",
                os.path.expanduser("~\\ffmpeg\\bin\\ffprobe.exe"),
                os.path.expanduser("~\\Desktop\\ffmpeg\\bin\\ffprobe.exe"),
                os.path.expanduser("~\\Downloads\\ffmpeg\\bin\\ffprobe.exe"),
                "ffprobe.exe",
                "ffprobe"
            ]
        
        if self.whisper_cache_dir is None:
            self.whisper_cache_dir = os.path.expanduser("~\\.cache\\whisper")
        
        if self.python_executable is None:
            self.python_executable = os.sys.executable
        
        if self.pip_executable is None:
            self.pip_executable = os.path.join(os.path.dirname(os.sys.executable), "Scripts", "pip.exe")
        
        if self.system_tools is None:
            self.system_tools = {}


class SystemPathManager:
    """Manages system paths and tool availability."""
    
    def __init__(self):
        self.paths = SystemPaths()
        self._tool_paths = {}
        self._discover_tools()
    
    def _discover_tools(self):
        """Discover available system tools."""
        logger.info("🔍 Discovering system tools...")
        
        # Discover FFmpeg
        ffmpeg_path = self._find_executable("ffmpeg", self.paths.ffmpeg_paths)
        if ffmpeg_path:
            self._tool_paths["ffmpeg"] = ffmpeg_path
            logger.info(f"✅ FFmpeg found: {ffmpeg_path}")
        else:
            logger.warning("❌ FFmpeg not found")
        
        # Discover FFprobe
        ffprobe_path = self._find_executable("ffprobe", self.paths.ffprobe_paths)
        if ffprobe_path:
            self._tool_paths["ffprobe"] = ffprobe_path
            logger.info(f"✅ FFprobe found: {ffprobe_path}")
        else:
            logger.warning("❌ FFprobe not found")
        
        # Check Python tools
        if os.path.exists(self.paths.python_executable):
            self._tool_paths["python"] = self.paths.python_executable
            logger.info(f"✅ Python found: {self.paths.python_executable}")
        
        if os.path.exists(self.paths.pip_executable):
            self._tool_paths["pip"] = self.paths.pip_executable
            logger.info(f"✅ Pip found: {self.paths.pip_executable}")
    
    def _find_executable(self, name: str, paths: List[str]) -> Optional[str]:
        """Find executable in given paths."""
        # First try shutil.which (checks PATH)
        which_result = shutil.which(name)
        if which_result:
            return which_result
        
        # Then try specific paths
        for path in paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        
        return None
    
    def get_tool_path(self, tool_name: str) -> Optional[str]:
        """Get path to a system tool."""
        return self._tool_paths.get(tool_name)
    
    def is_tool_available(self, tool_name: str) -> bool:
        """Check if tool is available."""
        return tool_name in self._tool_paths
    
    def test_tool(self, tool_name: str) -> bool:
        """Test if tool is working."""
        tool_path = self.get_tool_path(tool_name)
        if not tool_path:
            return False
        
        try:
            if tool_name in ["ffmpeg", "ffprobe"]:
                result = subprocess.run(
                    [tool_path, "-version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                return result.returncode == 0
            elif tool_name == "python":
                result = subprocess.run(
                    [tool_path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                return result.returncode == 0
            else:
                return os.path.exists(tool_path)
        
        except Exception as e:
            logger.error(f"Error testing {tool_name}: {e}")
            return False
    
    def get_system_info(self) -> Dict[str, any]:
        """Get system information."""
        info = {
            "available_tools": list(self._tool_paths.keys()),
            "tool_paths": self._tool_paths.copy(),
            "python_version": os.sys.version.split()[0],
            "platform": os.name,
            "whisper_cache": self.paths.whisper_cache_dir
        }
        
        # Test tools
        for tool in self._tool_paths:
            info[f"{tool}_working"] = self.test_tool(tool)
        
        return info
    
    def install_missing_tools(self) -> Dict[str, str]:
        """Provide installation instructions for missing tools."""
        instructions = {}
        
        if not self.is_tool_available("ffmpeg"):
            instructions["ffmpeg"] = """
🔧 FFmpeg Installation:
1. Download FFmpeg from: https://ffmpeg.org/download.html#build-windows
2. Extract to C:\\ffmpeg\\
3. Add C:\\ffmpeg\\bin to PATH environment variable
4. Restart terminal/IDE
            
Alternative: Use winget install ffmpeg or choco install ffmpeg
"""
        
        return instructions
    
    def create_whisper_cache_dir(self):
        """Create Whisper cache directory."""
        cache_dir = Path(self.paths.whisper_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Whisper cache directory: {cache_dir}")
        return str(cache_dir)


# Global instance
_system_path_manager = None


def get_system_path_manager() -> SystemPathManager:
    """Get global system path manager instance."""
    global _system_path_manager
    if _system_path_manager is None:
        _system_path_manager = SystemPathManager()
    return _system_path_manager


def get_tool_path(tool_name: str) -> Optional[str]:
    """Get path to a system tool."""
    return get_system_path_manager().get_tool_path(tool_name)


def is_tool_available(tool_name: str) -> bool:
    """Check if tool is available."""
    return get_system_path_manager().is_tool_available(tool_name)


def test_tool(tool_name: str) -> bool:
    """Test if tool is working."""
    return get_system_path_manager().test_tool(tool_name)


def get_system_info() -> Dict[str, any]:
    """Get system information."""
    return get_system_path_manager().get_system_info() 