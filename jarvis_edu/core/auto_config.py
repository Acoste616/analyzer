"""
Auto-mode configuration loader for Jarvis EDU Extractor.
Handles loading and validation of auto-mode.yaml configuration.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from loguru import logger


class SystemConfig(BaseModel):
    """System configuration settings."""
    auto_scan: bool = True
    watch_folder: str
    scan_interval: int = 60
    auto_process: bool = True
    delete_after_process: bool = False
    backup_original: bool = True
    backup_folder: str
    supported_formats: Dict[str, List[str]]
    max_file_size_mb: int = 2048
    max_concurrent_jobs: int = 3
    retry_attempts: int = 2
    timeout_minutes: int = 30

    @validator('watch_folder', 'backup_folder')
    def validate_paths(cls, v):
        """Validate that paths exist or can be created."""
        path = Path(v)
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {v}")
            except Exception as e:
                logger.warning(f"Cannot create directory {v}: {e}")
        return str(path.absolute())


class LLMConfig(BaseModel):
    """LLM configuration settings."""
    provider: str = "lmstudio"
    endpoint: str = "http://localhost:1234/v1"
    model: str = "mistral-7b-instruct"
    api_key: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 4096
    top_p: float = 0.9
    frequency_penalty: float = 0.1
    presence_penalty: float = 0.1
    fallback_providers: List[Dict[str, Any]] = []


class ProcessingConfig(BaseModel):
    """Content processing configuration."""
    audio_extraction: Dict[str, Any]
    whisper: Dict[str, Any]
    ocr: Dict[str, Any]
    text: Dict[str, Any]
    image: Dict[str, Any]


class OutputConfig(BaseModel):
    """Output configuration settings."""
    base_dir: str
    organize_by_date: bool = True
    organize_by_type: bool = True
    structure: Dict[str, str]
    formats: Dict[str, bool]
    naming: Dict[str, Any]

    @validator('base_dir')
    def validate_base_dir(cls, v):
        """Ensure output directory exists."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())


class LessonGenerationConfig(BaseModel):
    """Lesson generation settings."""
    enabled: bool = True
    structure: Dict[str, bool]
    difficulty_levels: List[str]
    default_difficulty: str = "intermediate"
    output_language: str = "pl"
    translate_content: bool = False
    quiz: Dict[str, Any]


class MonitoringConfig(BaseModel):
    """Monitoring and logging configuration."""
    log_level: str = "INFO"
    log_file: str
    max_log_size_mb: int = 100
    backup_count: int = 5
    collect_metrics: bool = True
    metrics_file: str
    notifications: Dict[str, Any]

    @validator('log_file', 'metrics_file')
    def validate_log_paths(cls, v):
        """Ensure log directories exist."""
        path = Path(v)
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())


class DatabaseConfig(BaseModel):
    """Database configuration."""
    type: str = "sqlite"
    path: str
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False
    auto_backup: bool = True
    backup_interval_hours: int = 24
    backup_retention_days: int = 30

    @validator('path')
    def validate_db_path(cls, v):
        """Ensure database directory exists."""
        path = Path(v)
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())


class SecurityConfig(BaseModel):
    """Security settings."""
    validate_files: bool = True
    scan_for_malware: bool = False
    max_file_size_mb: int = 2048
    content_filter: Dict[str, Any]
    api_rate_limiting: bool = True
    max_requests_per_minute: int = 60


class PerformanceConfig(BaseModel):
    """Performance settings."""
    max_cpu_percent: int = 80
    max_memory_percent: int = 70
    cache: Dict[str, Any]
    multiprocessing: bool = True
    max_workers: int = 4
    chunk_size: int = 1000


class IntegrationsConfig(BaseModel):
    """Integration settings."""
    vector_db: Dict[str, Any]
    apis: Dict[str, bool]
    webhooks: Dict[str, Any]


class DevelopmentConfig(BaseModel):
    """Development settings."""
    debug_mode: bool = False
    save_intermediate_files: bool = False
    verbose_logging: bool = False
    profiling: bool = False
    test_mode: bool = False
    mock_llm_responses: bool = False
    skip_heavy_processing: bool = False


class AutoModeConfig(BaseModel):
    """Complete auto-mode configuration."""
    system: SystemConfig
    llm: LLMConfig
    processing: ProcessingConfig
    output: OutputConfig
    lesson_generation: LessonGenerationConfig
    monitoring: MonitoringConfig
    database: DatabaseConfig
    security: SecurityConfig
    performance: PerformanceConfig
    integrations: IntegrationsConfig
    development: DevelopmentConfig

    class Config:
        extra = "forbid"  # Don't allow extra fields


class AutoConfigLoader:
    """Auto-mode configuration loader and manager."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration loader."""
        self.config_path = config_path or "config/auto-mode.yaml"
        self._config: Optional[AutoModeConfig] = None
    
    def load_config(self, reload: bool = False) -> AutoModeConfig:
        """Load configuration from YAML file."""
        if self._config is not None and not reload:
            return self._config
            
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(config_file, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            
            # Expand environment variables
            yaml_data = self._expand_env_vars(yaml_data)
            
            # Validate and create configuration
            self._config = AutoModeConfig(**yaml_data)
            
            logger.info(f"Auto-mode configuration loaded from: {self.config_path}")
            return self._config
            
        except Exception as e:
            logger.error(f"Failed to load auto-mode configuration: {e}")
            raise
    
    def _expand_env_vars(self, data: Any) -> Any:
        """Recursively expand environment variables in configuration."""
        if isinstance(data, dict):
            return {k: self._expand_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._expand_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            env_var = data[2:-1]
            return os.getenv(env_var, data)
        else:
            return data
    
    def get_config(self) -> AutoModeConfig:
        """Get loaded configuration."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def validate_config(self) -> bool:
        """Validate current configuration."""
        try:
            config = self.get_config()
            
            # Additional validation logic
            self._validate_paths(config)
            self._validate_llm_config(config.llm)
            self._validate_processing_config(config.processing)
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def _validate_paths(self, config: AutoModeConfig) -> None:
        """Validate all file paths in configuration."""
        paths_to_check = [
            config.system.watch_folder,
            config.system.backup_folder,
            config.output.base_dir,
        ]
        
        for path_str in paths_to_check:
            path = Path(path_str)
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise ValueError(f"Cannot create required directory {path}: {e}")
    
    def _validate_llm_config(self, llm_config: LLMConfig) -> None:
        """Validate LLM configuration."""
        # Check if endpoint is reachable (basic validation)
        if llm_config.provider == "lmstudio":
            if not llm_config.endpoint.startswith("http"):
                raise ValueError("LM Studio endpoint must be a valid HTTP URL")
    
    def _validate_processing_config(self, processing_config: ProcessingConfig) -> None:
        """Validate processing configuration."""
        # Validate Whisper model
        valid_whisper_models = ["tiny", "base", "small", "medium", "large"]
        whisper_model = processing_config.whisper.get("model", "base")
        if whisper_model not in valid_whisper_models:
            raise ValueError(f"Invalid Whisper model: {whisper_model}")
    
    def save_config(self, config: AutoModeConfig, path: Optional[str] = None) -> None:
        """Save configuration to YAML file."""
        save_path = path or self.config_path
        
        try:
            # Convert to dict and save
            config_dict = config.dict()
            
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def get_watch_folders(self) -> List[str]:
        """Get list of folders to watch."""
        config = self.get_config()
        return [config.system.watch_folder]
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of all supported file extensions."""
        config = self.get_config()
        extensions = []
        for format_list in config.system.supported_formats.values():
            extensions.extend(format_list)
        return extensions
    
    def is_file_supported(self, filepath: str) -> bool:
        """Check if file is supported based on extension."""
        ext = Path(filepath).suffix.lower()
        supported_extensions = self.get_supported_extensions()
        return ext in supported_extensions


# Global config loader instance
auto_config_loader = AutoConfigLoader()


def get_auto_config() -> AutoModeConfig:
    """Get the global auto-mode configuration."""
    return auto_config_loader.get_config()


def load_auto_config(config_path: Optional[str] = None) -> AutoModeConfig:
    """Load auto-mode configuration from file."""
    if config_path:
        loader = AutoConfigLoader(config_path)
        return loader.load_config()
    return auto_config_loader.load_config() 