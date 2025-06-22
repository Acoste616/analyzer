"""Configuration management using Pydantic Settings."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from functools import lru_cache

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class WhisperConfig(BaseModel):
    """Whisper configuration."""
    model: str = Field(default="base", description="Whisper model size")
    device: str = Field(default="cpu", description="Device to run on")
    language: Optional[str] = Field(default=None, description="Force language")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    
    @validator("model")
    def validate_model(cls, v):
        valid_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        if v not in valid_models:
            raise ValueError(f"Model must be one of {valid_models}")
        return v


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: str = Field(default="ollama", description="LLM provider")
    model: str = Field(default="llama3", description="Model name")
    base_url: str = Field(default="http://localhost:11434", description="API base URL")
    api_key: Optional[str] = Field(default=None, description="API key")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_tokens: int = Field(default=4000, description="Maximum tokens")
    
    @validator("provider")
    def validate_provider(cls, v):
        valid_providers = ["ollama", "openai", "anthropic", "local", "lmstudio"]
        if v not in valid_providers:
            raise ValueError(f"Provider must be one of {valid_providers}")
        return v


class DatabaseConfig(BaseModel):
    """Database configuration."""
    # Vector Database
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant port")
    qdrant_timeout: int = Field(default=60, description="Qdrant timeout")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379", description="Redis URL")
    redis_db: int = Field(default=0, description="Redis database number")
    
    # SQLite for metadata
    sqlite_path: str = Field(default="data/jarvis_edu.db", description="SQLite database path")
    
    # MinIO (optional object storage)
    minio_endpoint: Optional[str] = Field(default=None, description="MinIO endpoint")
    minio_access_key: Optional[str] = Field(default=None, description="MinIO access key")
    minio_secret_key: Optional[str] = Field(default=None, description="MinIO secret key")
    minio_bucket: str = Field(default="jarvis-edu", description="MinIO bucket name")


class ProcessingConfig(BaseModel):
    """Processing configuration."""
    # General
    max_workers: int = Field(default=4, description="Maximum worker processes")
    chunk_size: int = Field(default=5000, description="Text chunk size for processing")
    batch_size: int = Field(default=10, description="Batch size for processing")
    
    # Translation
    translation_provider: str = Field(default="google", description="Translation provider")
    target_language: str = Field(default="pl", description="Target language code")
    deepl_api_key: Optional[str] = Field(default=None, description="DeepL API key")
    
    # OCR
    ocr_confidence_threshold: float = Field(default=0.8, description="OCR confidence threshold")
    ocr_languages: List[str] = Field(default=["en", "pl"], description="OCR languages")
    
    # Video processing
    keyframe_interval: int = Field(default=30, description="Keyframe extraction interval (seconds)")
    max_video_size_mb: int = Field(default=1000, description="Maximum video size in MB")
    
    # Web scraping
    scraping_timeout: int = Field(default=30, description="Web scraping timeout")
    max_page_size_mb: int = Field(default=10, description="Maximum web page size")
    
    @validator("translation_provider")
    def validate_translation_provider(cls, v):
        valid_providers = ["google", "deepl", "azure", "aws"]
        if v not in valid_providers:
            raise ValueError(f"Translation provider must be one of {valid_providers}")
        return v


class SecurityConfig(BaseModel):
    """Security configuration."""
    # File validation
    max_file_size_mb: int = Field(default=500, description="Maximum file size in MB")
    allowed_extensions: List[str] = Field(
        default=[".mp4", ".avi", ".mov", ".mkv", ".webm", ".mp3", ".wav", ".pdf", ".docx", ".pptx"],
        description="Allowed file extensions"
    )
    scan_files: bool = Field(default=True, description="Enable virus scanning")
    
    # API security
    api_rate_limit: int = Field(default=100, description="API rate limit per minute")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    
    # Encryption
    secret_key: str = Field(default="your-secret-key-change-in-production", description="Secret key")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Token expiration time")


class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration."""
    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(default="json", description="Log format")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    # Sentry
    sentry_dsn: Optional[str] = Field(default=None, description="Sentry DSN")
    sentry_environment: str = Field(default="development", description="Sentry environment")
    
    # Prometheus
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=9090, description="Metrics server port")
    
    # Health checks
    health_check_interval: int = Field(default=30, description="Health check interval (seconds)")


class PathConfig(BaseModel):
    """Path configuration."""
    # Base directories
    base_dir: Path = Field(default=Path.cwd(), description="Base directory")
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    input_dir: Path = Field(default=Path("input"), description="Input directory")
    output_dir: Path = Field(default=Path("output"), description="Output directory")
    models_dir: Path = Field(default=Path("models"), description="Models directory")
    cache_dir: Path = Field(default=Path("cache"), description="Cache directory")
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")
    temp_dir: Path = Field(default=Path("temp"), description="Temporary directory")
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for path in [
            self.data_dir, self.input_dir, self.output_dir,
            self.models_dir, self.cache_dir, self.logs_dir, self.temp_dir
        ]:
            path.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Main application settings."""
    
    # App info
    app_name: str = Field(default="Jarvis EDU Extractor", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment")
    
    # API settings
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_prefix: str = Field(default="/api/v1", description="API prefix")
    
    # WebSocket settings
    websocket_port: int = Field(default=8001, description="WebSocket port")
    
    # Component configurations
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    
    # Plugin system
    plugins_enabled: bool = Field(default=True, description="Enable plugin system")
    plugins_dir: Path = Field(default=Path("plugins"), description="Plugins directory")
    
    # Feature flags
    features: Dict[str, bool] = Field(
        default={
            "web_ui": True,
            "websocket_progress": True,
            "batch_processing": True,
            "auto_translation": True,
            "code_validation": True,
            "vector_search": True,
            "export_formats": True,
            "plugin_system": True,
            "health_monitoring": True,
        },
        description="Feature flags"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = False
        
    @validator("environment")
    def validate_environment(cls, v):
        valid_envs = ["development", "testing", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v
    
    @validator("features")
    def validate_features(cls, v):
        """Validate feature dependencies."""
        if v.get("websocket_progress") and not v.get("web_ui"):
            raise ValueError("websocket_progress requires web_ui to be enabled")
        return v
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Create plugin directory
        if self.plugins_enabled:
            self.plugins_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize paths
        self.paths.__post_init__()
    
    def get_feature_flag(self, feature: str) -> bool:
        """Get feature flag value."""
        return self.features.get(feature, False)
    
    def enable_feature(self, feature: str) -> None:
        """Enable a feature."""
        self.features[feature] = True
    
    def disable_feature(self, feature: str) -> None:
        """Disable a feature."""
        self.features[feature] = False
    
    def get_database_url(self) -> str:
        """Get database URL."""
        return f"sqlite:///{self.database.sqlite_path}"
    
    def get_qdrant_url(self) -> str:
        """Get Qdrant URL."""
        return f"http://{self.database.qdrant_host}:{self.database.qdrant_port}"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def create_config_file(path: str = ".env") -> None:
    """Create example configuration file."""
    config_template = """# Jarvis EDU Extractor Configuration

# Application
APP_NAME=Jarvis EDU Extractor
APP_VERSION=0.1.0
DEBUG=false
ENVIRONMENT=development

# API
API_HOST=0.0.0.0
API_PORT=8000

# Whisper
WHISPER__MODEL=base
WHISPER__DEVICE=cpu

# LLM
LLM__PROVIDER=ollama
LLM__MODEL=llama3
LLM__BASE_URL=http://localhost:11434

# Database
DATABASE__QDRANT_HOST=localhost
DATABASE__QDRANT_PORT=6333
DATABASE__REDIS_URL=redis://localhost:6379

# Processing
PROCESSING__MAX_WORKERS=4
PROCESSING__TARGET_LANGUAGE=pl

# Security
SECURITY__MAX_FILE_SIZE_MB=500
SECURITY__SECRET_KEY=your-secret-key-change-in-production

# Monitoring
MONITORING__LOG_LEVEL=INFO
MONITORING__ENABLE_METRICS=true

# API Keys (optional)
LLM__API_KEY=
PROCESSING__DEEPL_API_KEY=
MONITORING__SENTRY_DSN=
"""
    
    with open(path, "w") as f:
        f.write(config_template)
    
    print(f"Configuration template created at {path}")


if __name__ == "__main__":
    # Create example config file
    create_config_file() 