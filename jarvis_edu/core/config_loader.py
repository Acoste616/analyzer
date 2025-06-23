"""
Configuration Loader for Jarvis EDU Continuous Processor
Handles loading and validation of YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProcessorConfig:
    """Configuration class for continuous processor"""
    
    # System settings
    watch_folder: str
    output_folder: str
    scan_interval: int
    auto_restart: bool
    max_restart_attempts: int
    restart_delay_multiplier: int
    
    # Processing settings
    max_concurrent_images: int
    max_concurrent_videos: int
    max_concurrent_audio: int
    gpu_memory_threshold: float
    gpu_cleanup_threshold: float
    max_retries: int
    retry_delay: int
    
    # Timeouts
    timeouts: Dict[str, Any]
    priorities: Dict[str, int]
    
    # Extraction settings
    ocr_engines: list
    ocr_languages: list
    ocr_confidence_threshold: float
    ocr_cache_results: bool
    max_image_dimension: int
    
    # Video settings
    keyframe_extraction: bool
    max_keyframes: int
    sample_long_videos: bool
    sample_interval: int
    sample_duration: int
    
    # Whisper settings
    whisper_model: str
    whisper_language: str
    whisper_device: str
    whisper_batch_size: int
    
    # AI settings
    ai_enable: bool
    ai_provider: str
    ai_endpoint: str
    ai_model: str
    ai_max_tokens: int
    ai_temperature: float
    ai_timeout: int
    
    # Storage settings
    knowledge_base_enable: bool
    categorize: bool
    generate_quizzes: bool
    organize_by_date: bool
    organize_by_type: bool
    keep_originals: bool
    
    # Cleanup settings
    cleanup_enable: bool
    cleanup_interval: int
    temp_file_age: int
    cache_size_limit: int
    
    # Monitoring settings
    system_check_interval: int
    log_level: str
    export_metrics: bool
    metrics_port: int
    dashboard_enable: bool
    dashboard_port: int
    auto_open_browser: bool
    
    # Alert thresholds
    alert_high_gpu_usage: float
    alert_high_cpu_usage: float
    alert_queue_backlog: int
    
    # Debug settings
    save_intermediate_files: bool
    verbose_logging: bool
    mock_ai_responses: bool
    process_limit: Optional[int]

class ConfigLoader:
    """Loads and validates configuration from YAML files"""
    
    def __init__(self, config_path: str = "config/continuous_processor.yaml"):
        self.config_path = Path(config_path)
        self._config_data: Optional[Dict[str, Any]] = None
    
    def load_config(self) -> ProcessorConfig:
        """Load and parse configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f)
            
            logger.info(f"Configuration loaded from: {self.config_path}")
            return self._parse_config()
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")
    
    def _parse_config(self) -> ProcessorConfig:
        """Parse loaded YAML data into ProcessorConfig object"""
        if not self._config_data:
            raise ValueError("No configuration data loaded")
        
        config = self._config_data
        
        return ProcessorConfig(
            # System settings
            watch_folder=config['system']['watch_folder'],
            output_folder=config['system']['output_folder'],
            scan_interval=config['system']['scan_interval'],
            auto_restart=config['system']['auto_restart'],
            max_restart_attempts=config['system']['max_restart_attempts'],
            restart_delay_multiplier=config['system']['restart_delay_multiplier'],
            
            # Processing settings
            max_concurrent_images=config['processing']['max_concurrent_images'],
            max_concurrent_videos=config['processing']['max_concurrent_videos'],
            max_concurrent_audio=config['processing']['max_concurrent_audio'],
            gpu_memory_threshold=config['processing']['gpu_memory_threshold'],
            gpu_cleanup_threshold=config['processing']['gpu_cleanup_threshold'],
            max_retries=config['processing']['max_retries'],
            retry_delay=config['processing']['retry_delay'],
            
            # Timeouts and priorities
            timeouts=config['processing']['timeouts'],
            priorities=config['processing']['priorities'],
            
            # OCR settings
            ocr_engines=config['extraction']['ocr']['engines'],
            ocr_languages=config['extraction']['ocr']['languages'],
            ocr_confidence_threshold=config['extraction']['ocr']['confidence_threshold'],
            ocr_cache_results=config['extraction']['ocr']['cache_results'],
            max_image_dimension=config['extraction']['ocr']['max_image_dimension'],
            
            # Video settings
            keyframe_extraction=config['extraction']['video']['keyframe_extraction'],
            max_keyframes=config['extraction']['video']['max_keyframes'],
            sample_long_videos=config['extraction']['video']['sample_long_videos'],
            sample_interval=config['extraction']['video']['sample_interval'],
            sample_duration=config['extraction']['video']['sample_duration'],
            
            # Whisper settings
            whisper_model=config['extraction']['whisper']['model'],
            whisper_language=config['extraction']['whisper']['language'],
            whisper_device=config['extraction']['whisper']['device'],
            whisper_batch_size=config['extraction']['whisper']['batch_size'],
            
            # AI settings
            ai_enable=config['extraction']['ai']['enable'],
            ai_provider=config['extraction']['ai']['provider'],
            ai_endpoint=config['extraction']['ai']['endpoint'],
            ai_model=config['extraction']['ai']['model'],
            ai_max_tokens=config['extraction']['ai']['max_tokens'],
            ai_temperature=config['extraction']['ai']['temperature'],
            ai_timeout=config['extraction']['ai']['timeout'],
            
            # Storage settings
            knowledge_base_enable=config['storage']['knowledge_base']['enable'],
            categorize=config['storage']['knowledge_base']['categorize'],
            generate_quizzes=config['storage']['knowledge_base']['generate_quizzes'],
            organize_by_date=config['storage']['organize_by_date'],
            organize_by_type=config['storage']['organize_by_type'],
            keep_originals=config['storage']['keep_originals'],
            
            # Cleanup settings
            cleanup_enable=config['storage']['cleanup']['enable'],
            cleanup_interval=config['storage']['cleanup']['interval'],
            temp_file_age=config['storage']['cleanup']['temp_file_age'],
            cache_size_limit=config['storage']['cleanup']['cache_size_limit'],
            
            # Monitoring settings
            system_check_interval=config['monitoring']['system_check_interval'],
            log_level=config['monitoring']['log_level'],
            export_metrics=config['monitoring']['export_metrics'],
            metrics_port=config['monitoring']['metrics_port'],
            dashboard_enable=config['monitoring']['dashboard']['enable'],
            dashboard_port=config['monitoring']['dashboard']['port'],
            auto_open_browser=config['monitoring']['dashboard']['auto_open_browser'],
            
            # Alert settings
            alert_high_gpu_usage=config['monitoring']['alerts']['high_gpu_usage'],
            alert_high_cpu_usage=config['monitoring']['alerts']['high_cpu_usage'],
            alert_queue_backlog=config['monitoring']['alerts']['queue_backlog'],
            
            # Debug settings
            save_intermediate_files=config['debug']['save_intermediate_files'],
            verbose_logging=config['debug']['verbose_logging'],
            mock_ai_responses=config['debug']['mock_ai_responses'],
            process_limit=config['debug']['process_limit'],
        )
    
    def get_timeout_for_file(self, file_type: str, file_size_mb: float) -> int:
        """Get appropriate timeout for file based on type and size"""
        if not self._config_data:
            raise ValueError("Configuration not loaded")
        
        timeouts = self._config_data['processing']['timeouts']
        
        if file_type == 'image':
            if file_size_mb < 10:
                return timeouts['image']['small']
            elif file_size_mb < 50:
                return timeouts['image']['medium']
            else:
                return timeouts['image']['large']
        
        elif file_type == 'video':
            if file_size_mb < 100:
                return timeouts['video']['small']
            elif file_size_mb < 500:
                return timeouts['video']['medium']
            elif file_size_mb < 1000:
                return timeouts['video']['large']
            else:
                return timeouts['video']['xlarge']
        
        elif file_type == 'audio':
            return timeouts['audio']['default']
        
        else:
            return 1800  # 30 minutes default
    
    def get_priority_for_file(self, file_type: str, file_size_mb: float) -> int:
        """Get processing priority for file"""
        if not self._config_data:
            raise ValueError("Configuration not loaded")
        
        priorities = self._config_data['processing']['priorities']
        
        if file_type == 'image':
            return priorities['image']
        elif file_type == 'video':
            if file_size_mb < 500:
                return priorities['video_small']
            else:
                return priorities['video_large']
        elif file_type == 'audio':
            return priorities['audio']
        else:
            return 3  # Medium priority default

# Global configuration instance
_config_loader: Optional[ConfigLoader] = None
_processor_config: Optional[ProcessorConfig] = None

def get_config() -> ProcessorConfig:
    """Get global processor configuration instance"""
    global _config_loader, _processor_config
    
    if _processor_config is None:
        if _config_loader is None:
            _config_loader = ConfigLoader()
        _processor_config = _config_loader.load_config()
    
    return _processor_config

def reload_config() -> ProcessorConfig:
    """Reload configuration from file"""
    global _config_loader, _processor_config
    
    _config_loader = ConfigLoader()
    _processor_config = _config_loader.load_config()
    
    return _processor_config 