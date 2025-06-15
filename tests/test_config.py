"""Tests for configuration system."""

import pytest
from jarvis_edu.core.config import Settings, WhisperConfig, LLMConfig


class TestSettings:
    """Test settings configuration."""
    
    def test_default_settings(self):
        """Test default settings creation."""
        settings = Settings()
        
        assert settings.app_name == "Jarvis EDU Extractor"
        assert settings.app_version == "0.1.0"
        assert settings.debug is False
        assert settings.environment == "development"
        
    def test_whisper_config(self):
        """Test Whisper configuration."""
        whisper_config = WhisperConfig()
        
        assert whisper_config.model == "base"
        assert whisper_config.device == "cpu"
        assert whisper_config.language is None
        assert whisper_config.temperature == 0.0
        
    def test_llm_config(self):
        """Test LLM configuration."""
        llm_config = LLMConfig()
        
        assert llm_config.provider == "ollama"
        assert llm_config.model == "llama3"
        assert llm_config.base_url == "http://localhost:11434"
        assert llm_config.temperature == 0.7
        
    def test_feature_flags(self):
        """Test feature flags."""
        settings = Settings()
        
        # Test default feature flags
        assert settings.get_feature_flag("web_ui") is True
        assert settings.get_feature_flag("websocket_progress") is True
        assert settings.get_feature_flag("batch_processing") is True
        assert settings.get_feature_flag("nonexistent_feature") is False
        
        # Test enabling/disabling features
        settings.enable_feature("test_feature")
        assert settings.get_feature_flag("test_feature") is True
        
        settings.disable_feature("test_feature")
        assert settings.get_feature_flag("test_feature") is False 