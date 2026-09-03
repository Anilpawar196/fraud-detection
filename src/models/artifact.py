"""The serving artifact: everything needed to turn a transaction into a decision.

A bare model file is not deployable. Reproducing a training-time prediction also
requires the fitted feature vocabulary, the calibration map, the decision
threshold chosen on validation, and the feature order. Bundling them in one
versioned object removes the most common production failure in ML systems —
training/serving skew from a preprocessing step that drifted out of sync with
the model.

The bundle is what gets logged to MLflow and loaded by the API at startup.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.preprocessing import LinearPreprocessor
from src.evaluation.calibration import ProbabilityCalibrator
from src.features.pipeline import FeaturePipeline

logger = logging.getLogger(__name__)

ARTIFACT_FILENAME = "model_artifact.pkl"
METADATA_FILENAME = "model_metadata.json"


@dataclass
class ArtifactMetadata:
    """Provenance and performance record travelling with the model."""

    model_name: str
    trained_at: str
    seed: int
    n_features: int
    n_train_rows: int
    dataset_rows_total: int
    holdout_cut_dt: int
    feature_config: dict[str, Any] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    holdout_metrics: dict[str, Any] = field(default_factory=dict)
    calibrated: bool = False
    library_versions: dict[str, str] = field(default_factory=dict)
    #: Raw dataset columns the feature pipeline reads. The API builds its input
    #: frame from exactly this list, filling anything the caller omits with NaN —
    #: which is a real capability rather than a shortcut, because LightGBM routes
    #: missing values natively and the model was trained on data that is 43%
    #: missing across the V block.
    raw_input_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelArtifact:
    """Model + feature pipeline + calibrator + threshold, versioned together."""

    model: Any
    feature_pipeline: FeaturePipeline
    metadata: ArtifactMetadata
    calibrator: ProbabilityCalibrator | None = None
    linear_preprocessor: LinearPreprocessor | None = None
    decision_threshold: float = 0.5

    # --- scoring ----------------------------------------------------------

    def raw_probability(self, prepared: pd.DataFrame) -> np.ndarray:
        """Uncalibrated model output for a prepared frame."""
        X = self.feature_pipeline.transform(prepared)
        if self.linear_preprocessor is not None:
            return self.model.predict_proba(self.linear_preprocessor.transform(X))[:, 1]
        return self.model.predict_proba(X)[:, 1]

    def predict_proba(self, prepared: pd.DataFrame) -> np.ndarray:
        """Calibrated fraud probability for a prepared frame.

        Args:
            prepared: Frame that has already been through
                :meth:`FeaturePipeline.prepare` — i.e. stateless features plus
                velocity columns. The fitted encoders are applied here.
        """
        probabilities = self.raw_probability(prepared)
        if self.calibrator is not None:
            probabilities = self.calibrator.transform(probabilities)
        return probabilities

    def transform_features(self, prepared: pd.DataFrame) -> pd.DataFrame:
        """Final model-input matrix — needed by the SHAP endpoint."""
        return self.feature_pipeline.transform(prepared)

    def risk_level(self, probability: float, medium: float, high: float) -> str:
        """Band a probability into low / medium / high."""
        if probability >= high:
            return "high"
        if probability >= medium:
            return "medium"
        return "low"

    # --- persistence ------------------------------------------------------

    def save(self, directory: Path) -> Path:
        """Persist the bundle plus a human-readable metadata sidecar."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / ARTIFACT_FILENAME
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        (directory / METADATA_FILENAME).write_text(
            json.dumps(self.metadata.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        logger.info("Saved model artifact to %s (%.1f MB)", path, path.stat().st_size / 1024**2)
        return path

    @staticmethod
    def load(path: Path) -> ModelArtifact:
        """Load a bundle from a file or a directory containing one."""
        target = path / ARTIFACT_FILENAME if path.is_dir() else path
        if not target.is_file():
            raise FileNotFoundError(f"No model artifact at {target}")
        with target.open("rb") as handle:
            artifact = pickle.load(handle)
        if not isinstance(artifact, ModelArtifact):
            raise TypeError(f"{target} does not contain a ModelArtifact")
        logger.info(
            "Loaded %s artifact trained %s (%d features)",
            artifact.metadata.model_name,
            artifact.metadata.trained_at,
            artifact.metadata.n_features,
        )
        return artifact


def library_versions() -> dict[str, str]:
    """Record the versions that produced an artifact, for reproducibility."""
    import lightgbm
    import shap
    import sklearn

    return {
        "python": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}"
        f".{__import__('sys').version_info.micro}",
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "shap": shap.__version__,
    }


def utc_now_iso() -> str:
    """Timezone-aware UTC timestamp string."""
    return datetime.now(UTC).isoformat(timespec="seconds")
