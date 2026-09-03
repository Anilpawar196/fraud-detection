"""API configuration from environment variables.

Deliberately small. The service has one external dependency — a model artifact on
disk — so there is nothing else to configure. Earlier versions carried Postgres
and Redis settings for prediction logging and an online feature store; both were
removed because neither was needed to demonstrate the model, and unexercised
infrastructure is worse than none.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the prediction service."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    log_level: str = "INFO"

    #: Path to the model bundle, relative to the repository root.
    model_artifact_path: str = "models/model_artifact.pkl"
    #: Reported by /health and in every prediction response.
    model_version: str = "1"

    # Risk banding is business configuration, not a model constant: the operating
    # point changes without retraining.
    risk_threshold_medium: float = Field(default=0.30, ge=0.0, le=1.0)
    risk_threshold_high: float = Field(default=0.70, ge=0.0, le=1.0)


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance — parsed once per process."""
    return Settings()
