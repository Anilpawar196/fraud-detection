"""Drift detection.

Two complementary statistics, because they answer different questions:

* **PSI (Population Stability Index)** — bins the reference distribution and
  compares mass per bin. It is the industry standard in credit/fraud risk
  precisely because it is interpretable on a fixed scale and does not care about
  sample size, so a large monitoring window does not manufacture alarm.
* **Kolmogorov–Smirnov** — a proper hypothesis test on the largest CDF gap. It is
  sensitive to shape changes PSI's binning can smooth over, but its p-value goes
  to zero for *any* difference once n is large, which is why it is reported
  alongside PSI rather than used as the trigger.

Conventional PSI thresholds are used and stated explicitly rather than left
implicit: **< 0.10 stable, 0.10–0.25 moderate shift, > 0.25 significant shift.**
These are heuristics from risk modelling practice, not properties of this
dataset, and are stated here rather than left implicit.

Bin edges come from the **reference** distribution and are reused for the current
window. Re-binning per window would compare two different binnings and produce a
meaningless number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

PSI_STABLE = 0.10
PSI_MODERATE = 0.25

DriftVerdict = Literal["stable", "moderate", "significant"]

#: Added to bin proportions so an empty bin does not produce infinite PSI.
_EPSILON = 1e-6


@dataclass
class FeatureDrift:
    """Drift statistics for one feature."""

    feature: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    reference_missing_rate: float
    current_missing_rate: float
    missing_rate_delta: float
    verdict: DriftVerdict

    def to_dict(self) -> dict[str, object]:
        return {
            "feature": self.feature,
            "psi": round(self.psi, 6),
            "ks_statistic": round(self.ks_statistic, 6),
            "ks_p_value": round(self.ks_p_value, 8),
            "reference_missing_rate": round(self.reference_missing_rate, 6),
            "current_missing_rate": round(self.current_missing_rate, 6),
            "missing_rate_delta": round(self.missing_rate_delta, 6),
            "verdict": self.verdict,
        }


def classify_psi(psi: float) -> DriftVerdict:
    """Map a PSI value to a verdict using the documented thresholds."""
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_MODERATE:
        return "moderate"
    return "significant"


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, n_bins: int = 10
) -> float:
    """PSI between a reference and a current sample of one numeric feature.

    Quantile bins are derived from the reference. Features whose values are
    heavily tied (many zeros — common for the velocity counts) collapse to fewer
    usable bins, which is handled by deduplicating edges rather than by forcing
    ``n_bins``.
    """
    reference = np.asarray(reference, dtype="float64")
    current = np.asarray(current, dtype="float64")
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if reference.size == 0 or current.size == 0:
        return float("nan")

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 3:
        # Degenerate (near-constant) feature: compare means of the two samples
        # as a crude proxy rather than reporting a fabricated PSI.
        return 0.0 if np.allclose(reference.mean(), current.mean()) else float(PSI_MODERATE)

    interior = edges[1:-1]
    reference_counts = np.bincount(
        np.digitize(reference, interior, right=True), minlength=edges.size - 1
    ).astype("float64")
    current_counts = np.bincount(
        np.digitize(current, interior, right=True), minlength=edges.size - 1
    ).astype("float64")

    reference_share = reference_counts / max(reference_counts.sum(), 1.0) + _EPSILON
    current_share = current_counts / max(current_counts.sum(), 1.0) + _EPSILON

    return float(
        np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    )


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    """PSI for a categorical feature, over observed category shares."""
    reference_share = reference.value_counts(normalize=True, dropna=False)
    current_share = current.value_counts(normalize=True, dropna=False)
    categories = reference_share.index.union(current_share.index)
    r = reference_share.reindex(categories).fillna(0.0).to_numpy() + _EPSILON
    c = current_share.reindex(categories).fillna(0.0).to_numpy() + _EPSILON
    return float(np.sum((c - r) * np.log(c / r)))


def feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str] | None = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute drift statistics for every shared feature.

    Args:
        reference: Baseline sample (the training distribution).
        current: Sample to test (recent traffic, or the unlabeled test period).
        features: Restrict to these columns; defaults to the shared columns.
        n_bins: PSI bin count.

    Returns:
        One row per feature, sorted by PSI descending.
    """
    shared = features or [c for c in reference.columns if c in current.columns]
    results: list[FeatureDrift] = []

    for name in shared:
        reference_column = reference[name]
        current_column = current[name]
        reference_missing = float(reference_column.isna().mean())
        current_missing = float(current_column.isna().mean())

        if pd.api.types.is_numeric_dtype(reference_column) and pd.api.types.is_numeric_dtype(
            current_column
        ):
            psi = population_stability_index(
                reference_column.to_numpy(), current_column.to_numpy(), n_bins=n_bins
            )
            reference_values = reference_column.dropna().to_numpy()
            current_values = current_column.dropna().to_numpy()
            if reference_values.size > 1 and current_values.size > 1:
                ks = stats.ks_2samp(reference_values, current_values)
                ks_statistic, ks_p = float(ks.statistic), float(ks.pvalue)
            else:
                ks_statistic, ks_p = float("nan"), float("nan")
        else:
            psi = categorical_psi(reference_column, current_column)
            ks_statistic, ks_p = float("nan"), float("nan")

        results.append(
            FeatureDrift(
                feature=name,
                psi=psi,
                ks_statistic=ks_statistic,
                ks_p_value=ks_p,
                reference_missing_rate=reference_missing,
                current_missing_rate=current_missing,
                missing_rate_delta=current_missing - reference_missing,
                verdict=classify_psi(psi) if np.isfinite(psi) else "stable",
            )
        )

    table = pd.DataFrame([r.to_dict() for r in results])
    if table.empty:
        return table
    return table.sort_values("psi", ascending=False).reset_index(drop=True)


def prediction_drift(
    reference_scores: np.ndarray, current_scores: np.ndarray, n_bins: int = 10
) -> dict[str, float | str]:
    """Drift in the model's own output distribution.

    Score drift is the earliest available warning in a fraud system, because it
    needs no labels — chargeback outcomes arrive weeks later, so waiting for
    measured PR-AUC to move means detecting degradation far too late.
    """
    psi = population_stability_index(reference_scores, current_scores, n_bins=n_bins)
    reference_scores = np.asarray(reference_scores, dtype="float64")
    current_scores = np.asarray(current_scores, dtype="float64")
    return {
        "psi": round(psi, 6),
        "verdict": classify_psi(psi) if np.isfinite(psi) else "stable",
        "reference_mean": round(float(np.nanmean(reference_scores)), 6),
        "current_mean": round(float(np.nanmean(current_scores)), 6),
        "reference_p99": round(float(np.nanquantile(reference_scores, 0.99)), 6),
        "current_p99": round(float(np.nanquantile(current_scores, 0.99)), 6),
    }


def summarise(table: pd.DataFrame, top_n: int = 20) -> dict[str, object]:
    """Aggregate a drift table into a compact report payload."""
    if table.empty:
        return {"n_features": 0, "significant": 0, "moderate": 0, "stable": 0, "top": []}
    counts = table["verdict"].value_counts().to_dict()
    return {
        "n_features": int(len(table)),
        "significant": int(counts.get("significant", 0)),
        "moderate": int(counts.get("moderate", 0)),
        "stable": int(counts.get("stable", 0)),
        "top": table.head(top_n).to_dict(orient="records"),
    }
