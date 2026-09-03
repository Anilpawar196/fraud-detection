"""Tests for PSI/KS drift detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    classify_psi,
    feature_drift,
    population_stability_index,
    prediction_drift,
    summarise,
)


class TestPSI:
    def test_identical_distributions_give_near_zero_psi(self) -> None:
        generator = np.random.default_rng(5)
        sample = generator.normal(size=5000)
        other = generator.normal(size=5000)
        assert population_stability_index(sample, other) < 0.1

    def test_shifted_distribution_is_detected(self) -> None:
        generator = np.random.default_rng(6)
        reference = generator.normal(0, 1, size=5000)
        current = generator.normal(3, 1, size=5000)
        psi = population_stability_index(reference, current)
        assert psi > 0.25
        assert classify_psi(psi) == "significant"

    def test_psi_is_finite_when_a_bin_is_empty(self) -> None:
        reference = np.concatenate([np.zeros(1000), np.ones(1000)])
        current = np.zeros(1000)
        assert np.isfinite(population_stability_index(reference, current))

    def test_near_constant_feature_does_not_explode(self) -> None:
        assert np.isfinite(population_stability_index(np.ones(100), np.ones(100)))

    def test_empty_input_returns_nan(self) -> None:
        assert np.isnan(population_stability_index(np.array([]), np.array([1.0])))

    def test_thresholds_map_to_verdicts(self) -> None:
        assert classify_psi(0.05) == "stable"
        assert classify_psi(0.15) == "moderate"
        assert classify_psi(0.40) == "significant"


class TestFeatureDrift:
    def test_detects_drift_in_one_column_only(self) -> None:
        generator = np.random.default_rng(7)
        reference = pd.DataFrame(
            {
                "stable": generator.normal(size=3000),
                "drifting": generator.normal(size=3000),
                "category": generator.choice(["a", "b"], size=3000),
            }
        )
        current = pd.DataFrame(
            {
                "stable": generator.normal(size=3000),
                "drifting": generator.normal(5, 1, size=3000),
                "category": generator.choice(["a", "b"], size=3000),
            }
        )
        table = feature_drift(reference, current)
        assert table.iloc[0]["feature"] == "drifting"
        assert table.iloc[0]["verdict"] == "significant"
        stable_row = table[table["feature"] == "stable"].iloc[0]
        assert stable_row["verdict"] == "stable"

    def test_reports_missing_rate_change(self) -> None:
        reference = pd.DataFrame({"x": [1.0] * 100})
        current = pd.DataFrame({"x": [1.0] * 50 + [np.nan] * 50})
        table = feature_drift(reference, current)
        assert table.iloc[0]["current_missing_rate"] == pytest.approx(0.5)
        assert table.iloc[0]["missing_rate_delta"] == pytest.approx(0.5)

    def test_summarise_counts_verdicts(self) -> None:
        generator = np.random.default_rng(8)
        reference = pd.DataFrame({"a": generator.normal(size=500)})
        current = pd.DataFrame({"a": generator.normal(size=500)})
        summary = summarise(feature_drift(reference, current))
        assert summary["n_features"] == 1
        assert summary["stable"] + summary["moderate"] + summary["significant"] == 1


class TestPredictionDrift:
    def test_detects_a_score_distribution_shift(self) -> None:
        generator = np.random.default_rng(9)
        reference = generator.beta(1, 20, size=3000)
        current = generator.beta(3, 5, size=3000)
        report = prediction_drift(reference, current)
        assert report["verdict"] == "significant"
        assert report["current_mean"] > report["reference_mean"]
