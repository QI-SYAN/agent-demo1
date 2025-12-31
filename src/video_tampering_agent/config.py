"""Application configuration management."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, HttpUrl, PositiveFloat
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseSettings):
    """LLM provider configuration."""

    model_config = SettingsConfigDict(
        env_prefix="MODEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: Literal["openai", "azure_openai", "ollama", "custom", "deepseek"] = "deepseek"
    model_name: str = Field(default="deepseek-chat", description="Default model identifier")
    api_key: Optional[str] = Field(default=None, description="API key for the configured provider")
    base_url: Optional[HttpUrl] = Field(
        default="https://api.deepseek.com/v1",
        description="Custom inference endpoint for self-hosted models or DeepSeek-compatible APIs",
    )
    request_timeout: PositiveFloat = Field(default=60.0, description="Timeout (seconds) for model calls")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class AgentSettings(BaseSettings):
    """Top-level configuration for the video tampering detection agent."""

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Literal["dev", "prod", "test"] = "dev"
    knowledge_base_path: Path = Field(default=Path("knowledge_base"))
    cache_directory: Path = Field(default=Path(".cache"))
    enable_telemetry: bool = False
    detection_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    max_video_duration_seconds: int = Field(default=900, ge=1)
    max_replan_loops: int = Field(default=1, ge=0, le=5, description="Maximum times the agent may replan/self-correct")
    model: ModelSettings = Field(default_factory=ModelSettings)


settings = AgentSettings()
