"""YAML configuration loading with environment-variable overlay.

Config is data, not code: nothing in ``src/`` hardcodes a threshold, ratio, or
seed. Values resolve in this order (later wins):

1. defaults in this module
2. ``configs/config.yaml``
3. environment variables (only for the keys explicitly mapped below)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import CONFIG_DIR

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitConfig:
    """Temporal split parameters."""

    holdout_fraction: float = 0.20
    validation_fraction: float = 0.20  # of the post-holdout training period
    n_cv_folds: int = 5
    purge_days: int = 7
    strategy: str = "temporal"


@dataclass(frozen=True)
class FeatureConfig:
    """Feature-engineering switches."""

    keep_all_v_columns: bool = True
    velocity_windows_hours: tuple[int, ...] = (1, 24, 168)
    frequency_encode_min_count: int = 2
    anchor_d_columns: bool = True
    #: Suffixes dropped from the model input; used for feature ablations.
    exclude_feature_suffixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainConfig:
    """Training / selection parameters."""

    primary_metric: str = "pr_auc"
    seed: int = 42
    n_jobs: int = -1
    optuna_trials: int = 25
    optuna_timeout_seconds: int = 1800
    calibrate: bool = True


@dataclass(frozen=True)
class ServingConfig:
    """Risk banding for the API response."""

    risk_threshold_medium: float = 0.30
    risk_threshold_high: float = 0.70


@dataclass(frozen=True)
class Config:
    """Root configuration object."""

    split: SplitConfig = field(default_factory=SplitConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        """Load config from YAML, then apply environment overrides."""
        cfg_path = path or (CONFIG_DIR / "config.yaml")
        raw: dict[str, Any] = {}
        if cfg_path.is_file():
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            logger.debug("Loaded config from %s", cfg_path)
        else:
            logger.warning("No config file at %s — using defaults", cfg_path)

        cfg = cls(
            split=SplitConfig(**raw.get("split", {})),
            features=_feature_config(raw.get("features", {})),
            train=TrainConfig(**raw.get("train", {})),
            serving=ServingConfig(**raw.get("serving", {})),
        )
        return cfg._with_env_overrides()

    def _with_env_overrides(self) -> Config:
        serving = self.serving
        medium = os.getenv("RISK_THRESHOLD_MEDIUM")
        high = os.getenv("RISK_THRESHOLD_HIGH")
        if medium or high:
            serving = ServingConfig(
                risk_threshold_medium=float(medium) if medium else serving.risk_threshold_medium,
                risk_threshold_high=float(high) if high else serving.risk_threshold_high,
            )
        train = self.train
        if seed := os.getenv("RANDOM_SEED"):
            train = replace(train, seed=int(seed))
        return replace(self, serving=serving, train=train)


def _feature_config(raw: dict[str, Any]) -> FeatureConfig:
    """Build FeatureConfig, coercing the YAML list of windows to a tuple."""
    data = dict(raw)
    if "velocity_windows_hours" in data:
        data["velocity_windows_hours"] = tuple(data["velocity_windows_hours"])
    if "exclude_feature_suffixes" in data:
        data["exclude_feature_suffixes"] = tuple(data["exclude_feature_suffixes"])
    return FeatureConfig(**data)


def load_config(path: Path | None = None) -> Config:
    """Module-level convenience wrapper around :meth:`Config.load`."""
    return Config.load(path)
