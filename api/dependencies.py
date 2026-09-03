"""Application state and the request -> feature-frame conversion.

The model is loaded **once**, at startup, and reused for every request. Loading
per request would re-read a 52 MB artifact and add hundreds of milliseconds.

On velocity features: the model uses trailing-window counts per card
(``entity_card_txn_count_1h`` and friends). A stateless service has no
transaction history, so these are computed from the request batch alone — a
single transaction correctly yields a count of 0, meaning "no prior activity
known to this service". That is honest rather than fabricated, and the model
handles it natively. A production deployment would back these with a feature
store; see the README's Limitations section.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException, status

from api.settings import Settings, get_settings
from src.data.schema import KEY, RAW_CATEGORICAL, TIME_COL
from src.models.artifact import ModelArtifact
from src.utils.paths import ROOT

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    """Process-wide state populated during startup."""

    settings: Settings
    artifact: ModelArtifact | None = None
    started_at: float = field(default_factory=time.time)
    request_count: int = 0

    @property
    def model_version(self) -> str:
        return self.settings.model_version

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.started_at


state = AppState(settings=get_settings())


def load_artifact(settings: Settings) -> ModelArtifact:
    """Load the model bundle from disk."""
    return ModelArtifact.load(ROOT / settings.model_artifact_path)


def get_state() -> AppState:
    """Application state accessor."""
    return state


def require_artifact() -> ModelArtifact:
    """Return the loaded model, or 503 if startup did not complete."""
    if state.artifact is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Train one with: python scripts/train.py",
        )
    return state.artifact


def build_prepared_frame(
    records: list[dict[str, Any]], artifact: ModelArtifact, default_timestamp: int
) -> pd.DataFrame:
    """Turn raw request records into a frame the feature pipeline accepts.

    Materialises every raw column the pipeline reads, filling anything the caller
    omitted with NaN — a real capability rather than a shortcut, since the model
    is a LightGBM trained on data that is 43% missing across the V block and
    routes missing values natively.
    """
    columns = artifact.metadata.raw_input_columns or []
    if not columns:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model artifact declares no raw_input_columns; retrain with scripts/train.py",
        )

    categorical = set(RAW_CATEGORICAL)
    frame: dict[str, Any] = {}

    for column in columns:
        if column == KEY:
            continue
        if column in categorical:
            frame[column] = pd.Series(
                [_string_or_none(record.get(column)) for record in records], dtype="object"
            )
        elif column == TIME_COL:
            frame[column] = pd.Series(
                [int(record.get(TIME_COL) or default_timestamp) for record in records],
                dtype="int64",
            )
        else:
            frame[column] = pd.Series(
                [_float_or_nan(record.get(column)) for record in records], dtype="float32"
            )

    df = pd.DataFrame(frame)
    # A synthetic id is required as the velocity grouping key; it is never a
    # model input (schema.EXCLUDED_FROM_FEATURES drops it).
    base_id = int(time.time_ns() // 1000)
    df.insert(0, KEY, np.arange(base_id, base_id + len(records), dtype="int64"))
    df = df.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)

    # velocity_frame=None -> computed from this batch only (see module docstring).
    return artifact.feature_pipeline.prepare(df)


def count_supplied_features(record: dict[str, Any]) -> int:
    """How many non-null raw features the caller actually provided."""
    return sum(1 for value in record.values() if value is not None and value == value)


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
    except TypeError:
        pass
    return str(value)


def _float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
