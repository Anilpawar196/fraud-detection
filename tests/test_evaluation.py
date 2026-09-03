"""Tests for metrics, calibration and the comparison table."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.calibration import ProbabilityCalibrator
from src.evaluation.compare import build_comparison, metrics_row
from src.evaluation.metrics import (
    calibration_table,
    compute_metrics,
    expected_calibration_error,
    find_best_threshold,
    precision_at_alert_budget,
)


@pytest.fixture
def imbalanced_predictions() -> tuple[np.ndarray, np.ndarray]:
    """3.5% prevalence with a genuinely informative score."""
    generator = np.random.default_rng(11)
    n = 5000
    y = (generator.random(n) < 0.035).astype("int8")
    scores = generator.beta(1, 12, size=n)
    # Push positives upward so the ranking carries signal.
    scores[y == 1] = np.clip(
        scores[y == 1] + generator.uniform(0.2, 0.6, size=(y == 1).sum()), 0, 1
    )
    return y, scores


class TestComputeMetrics:
    def test_reports_pr_auc_lift_against_prevalence(self, imbalanced_predictions) -> None:
        y, scores = imbalanced_predictions
        metrics = compute_metrics(y, scores)
        assert 0.0 < metrics.pr_auc <= 1.0
        assert metrics.pr_auc_lift == pytest.approx(metrics.pr_auc / metrics.prevalence, rel=1e-6)
        assert metrics.pr_auc_lift > 1.0, "informative scores must beat the no-skill floor"

    def test_confusion_counts_sum_to_sample_size(self, imbalanced_predictions) -> None:
        y, scores = imbalanced_predictions
        metrics = compute_metrics(y, scores)
        total = (
            metrics.true_negatives
            + metrics.false_positives
            + metrics.false_negatives
            + metrics.true_positives
        )
        assert total == metrics.n_samples == len(y)

    def test_accuracy_is_not_reported(self, imbalanced_predictions) -> None:
        # Accuracy is deliberately absent: a constant predictor scores 96.5% here.
        y, scores = imbalanced_predictions
        assert "accuracy" not in compute_metrics(y, scores).to_flat_dict()

    def test_rejects_single_class_input(self) -> None:
        with pytest.raises(ValueError, match="single class"):
            compute_metrics(np.zeros(10, dtype="int8"), np.linspace(0, 1, 10))

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_metrics(np.array([0, 1, 0]), np.array([0.1, 0.2]))

    def test_perfect_ranking_gives_pr_auc_one(self) -> None:
        y = np.array([0, 0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
        assert compute_metrics(y, scores).pr_auc == pytest.approx(1.0)

    def test_explicit_threshold_is_respected(self, imbalanced_predictions) -> None:
        y, scores = imbalanced_predictions
        metrics = compute_metrics(y, scores, threshold=0.5)
        assert metrics.threshold == 0.5
        assert metrics.true_positives == int(((scores >= 0.5) & (y == 1)).sum())


class TestAlertBudget:
    def test_precision_at_budget_uses_the_top_slice(self) -> None:
        y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        scores = np.array([0.99, 0.98, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        precision, recall = precision_at_alert_budget(y, scores, 0.2)
        assert precision == pytest.approx(1.0)
        assert recall == pytest.approx(1.0)

    def test_budget_recall_cannot_exceed_one(self, imbalanced_predictions) -> None:
        y, scores = imbalanced_predictions
        _, recall = precision_at_alert_budget(y, scores, 0.5)
        assert 0.0 <= recall <= 1.0


class TestThresholdSelection:
    def test_returns_a_threshold_within_score_range(self, imbalanced_predictions) -> None:
        y, scores = imbalanced_predictions
        threshold = find_best_threshold(y, scores)
        assert scores.min() <= threshold <= scores.max()


class TestCalibration:
    def test_isotonic_reduces_calibration_error_on_inflated_scores(self) -> None:
        generator = np.random.default_rng(3)
        n = 4000
        true_probability = generator.beta(1.5, 20, size=n)
        y = (generator.random(n) < true_probability).astype("int8")
        # Simulate scale_pos_weight inflation: monotone but systematically high.
        inflated = np.clip(true_probability * 6.0, 0, 1)

        calibrator = ProbabilityCalibrator().fit(y, inflated)
        assert calibrator.improved
        assert calibrator.ece_after < calibrator.ece_before
        calibrated = calibrator.transform(inflated)
        assert calibrated.min() >= 0.0 and calibrated.max() <= 1.0

    def test_calibration_preserves_ranking(self) -> None:
        generator = np.random.default_rng(4)
        y = (generator.random(2000) < 0.05).astype("int8")
        scores = generator.random(2000)
        calibrator = ProbabilityCalibrator().fit(y, scores)
        calibrated = calibrator.transform(scores)
        # Isotonic is monotone, so ordering must be preserved (ties allowed).
        order = np.argsort(scores)
        assert np.all(np.diff(calibrated[order]) >= -1e-9)

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="before fit"):
            ProbabilityCalibrator().transform(np.array([0.5]))

    def test_calibration_table_bins_are_consistent(self, imbalanced_predictions) -> None:
        y, scores = imbalanced_predictions
        table = calibration_table(y, scores, n_bins=5)
        assert len(table["mean_predicted"]) == len(table["observed_rate"]) == len(table["count"])
        assert sum(table["count"]) == len(y)
        assert expected_calibration_error(y, scores) >= 0.0


class TestComparisonTable:
    def test_sorted_by_pr_auc_within_evaluation(self, imbalanced_predictions) -> None:
        y, scores = imbalanced_predictions
        strong = compute_metrics(y, scores)
        weak = compute_metrics(y, np.random.default_rng(0).random(len(y)))
        table = build_comparison(
            [
                metrics_row("weak", "holdout_final", weak),
                metrics_row("strong", "holdout_final", strong),
            ]
        )
        assert table.iloc[0]["model"] == "strong"
        assert "training_time_seconds" in table.columns
        assert "n_train_rows" in table.columns
