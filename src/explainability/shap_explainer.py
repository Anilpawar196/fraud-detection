"""SHAP explanations for the LightGBM model.

Serves two distinct audiences:

* **Global** (offline): which features drive the model overall, used to decide
  whether the 339 V columns actually earn their place — the question we
  deliberately deferred rather than answering by assumption.
* **Local** (online): why *this* transaction scored as it did, returned by
  ``POST /explain``. A fraud analyst cannot action "0.87"; they can action
  "0.87, driven by 9 transactions on this card in the last hour and a device
  never seen before".

``TreeExplainer`` is exact for tree ensembles and needs no background dataset,
so a single-row explanation costs milliseconds — which is what makes the online
endpoint viable at all. Kernel/Permutation explainers would need hundreds of
model evaluations per row and could not meet an API latency budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


@dataclass
class FeatureContribution:
    """One feature's contribution to one prediction."""

    feature: str
    value: float | str | None
    shap_value: float

    @property
    def direction(self) -> str:
        return "increases_risk" if self.shap_value > 0 else "decreases_risk"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "value": self.value,
            "shap_value": round(float(self.shap_value), 6),
            "direction": self.direction,
        }


class ShapExplainer:
    """Wraps ``shap.TreeExplainer`` with the plumbing this project needs."""

    def __init__(self, model: Any, feature_names: list[str]) -> None:
        self.explainer = shap.TreeExplainer(model)
        self.feature_names = list(feature_names)
        self._base_value: float | None = None

    @property
    def base_value(self) -> float:
        """Model output for an average input, in log-odds space."""
        if self._base_value is None:
            expected = self.explainer.expected_value
            self._base_value = float(
                expected[-1] if isinstance(expected, list | np.ndarray) else expected
            )
        return self._base_value

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP values for the positive class, shaped ``(n_rows, n_features)``.

        LightGBM binary models return a single matrix in newer SHAP versions but
        a two-element list in older ones; both shapes are normalised here so
        callers do not have to care.
        """
        values = self.explainer.shap_values(X, check_additivity=False)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            # (n_rows, n_features, n_classes) -> positive class
            values = values[:, :, -1]
        return values

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Mean absolute SHAP value per feature, descending.

        Mean |SHAP| is preferred over LightGBM's built-in split-count or gain
        importance because it is in units of model output and is consistent
        between the global and local views the API exposes — the same number
        explains the ranking and the individual prediction.
        """
        values = self.shap_values(X)
        importance = pd.DataFrame(
            {
                "feature": X.columns,
                "mean_abs_shap": np.abs(values).mean(axis=0),
                "mean_shap": values.mean(axis=0),
            }
        )
        return importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    def explain_row(
        self, X: pd.DataFrame, row: int = 0, top_n: int = 10
    ) -> list[FeatureContribution]:
        """Top contributing features for a single transaction, by magnitude."""
        if len(X) == 0:
            raise ValueError("Cannot explain an empty frame")
        single = X.iloc[[row]]
        values = self.shap_values(single)[0]
        order = np.argsort(-np.abs(values))[:top_n]

        contributions: list[FeatureContribution] = []
        for index in order:
            name = str(X.columns[index])
            raw = single.iloc[0, index]
            contributions.append(
                FeatureContribution(
                    feature=name,
                    value=_jsonable_value(raw),
                    shap_value=float(values[index]),
                )
            )
        return contributions

    def summary_plot(self, X: pd.DataFrame, out_path: Path, max_display: int = 20) -> Path:
        """Write a SHAP beeswarm summary plot."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        values = self.shap_values(X)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure()
        shap.summary_plot(values, X, max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close("all")
        logger.info("Wrote SHAP summary plot to %s", out_path)
        return out_path

    def bar_plot(self, importance: pd.DataFrame, out_path: Path, top_n: int = 25) -> Path:
        """Write a horizontal bar chart of global importance."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = importance.head(top_n).iloc[::-1]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(9, max(4, 0.32 * len(top))))
        plt.barh(top["feature"], top["mean_abs_shap"], color="#c2410c")
        plt.xlabel("mean |SHAP value|")
        plt.title(f"Global feature importance (top {top_n})")
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close("all")
        logger.info("Wrote SHAP importance bar plot to %s", out_path)
        return out_path


def _jsonable_value(value: Any) -> float | str | None:
    """Coerce a cell value into something JSON-serialisable."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.integer | int):
        return int(value)
    if isinstance(value, np.floating | float):
        return None if np.isnan(value) else round(float(value), 6)
    return str(value)


def sample_for_explanation(X: pd.DataFrame, n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Subsample rows for global SHAP.

    Exact TreeSHAP over 472k rows x ~528 features is unnecessary for a stable
    global ranking — a few thousand rows converges to the same ordering — and
    would otherwise dominate the training script's runtime.
    """
    if len(X) <= n:
        return X
    return X.sample(n=n, random_state=seed)
