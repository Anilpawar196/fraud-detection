"""Evaluation metrics for a highly imbalanced binary problem.

Accuracy is deliberately absent. At the measured 3.4993% prevalence, predicting
"never fraud" scores 96.5007% accuracy, so accuracy cannot distinguish a useful
model from a constant and reporting it would be actively misleading.

**PR-AUC is the selection metric; ROC-AUC is reported for comparability.**
ROC-AUC is built from TPR against FPR, and FPR's denominator holds 569,877
negatives — so thousands of extra false positives barely move it, even though
false positives are the cost that dominates fraud review. PR-AUC uses precision,
whose denominator is the predicted-positive set, making it directly sensitive to
analyst workload. Their no-skill baselines differ too: ROC-AUC is 0.5 regardless
of prevalence, while PR-AUC's baseline is the prevalence itself (~0.035 here), so
PR-AUC should always be read as a multiple of that floor.

``precision_at_alert_budget`` is included because it is the metric an operations
team actually negotiates: given capacity to review N transactions per day, what
fraction of those alerts are real fraud?
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class ClassificationMetrics:
    """Metric bundle for one model on one dataset."""

    roc_auc: float
    pr_auc: float
    pr_auc_lift: float
    precision: float
    recall: float
    f1: float
    threshold: float
    brier: float
    prevalence: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    n_samples: int
    precision_at_budget: dict[str, float] = field(default_factory=dict)
    recall_at_budget: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten nested budget dicts for CSV / MLflow logging."""
        flat: dict[str, Any] = {k: v for k, v in asdict(self).items() if not isinstance(v, dict)}
        for label, value in self.precision_at_budget.items():
            flat[f"precision_at_{label}"] = value
        for label, value in self.recall_at_budget.items():
            flat[f"recall_at_{label}"] = value
        return flat

    def summary(self) -> str:
        return (
            f"PR-AUC {self.pr_auc:.4f} ({self.pr_auc_lift:.1f}x baseline) | "
            f"ROC-AUC {self.roc_auc:.4f} | P {self.precision:.4f} | "
            f"R {self.recall:.4f} | F1 {self.f1:.4f} @ thr {self.threshold:.4f}"
        )


def precision_at_alert_budget(
    y_true: np.ndarray, y_prob: np.ndarray, budget_fraction: float
) -> tuple[float, float]:
    """Precision and recall when only the top ``budget_fraction`` are alerted.

    This is the operational framing: a review team has finite capacity, so the
    model's job is to rank, and the question is what the top slice contains.

    Returns:
        ``(precision, recall)`` over the top-ranked slice.
    """
    n_alerts = max(1, int(round(len(y_prob) * budget_fraction)))
    top = np.argpartition(-y_prob, n_alerts - 1)[:n_alerts]
    caught = float(y_true[top].sum())
    total_positives = float(y_true.sum())
    precision = caught / n_alerts
    recall = caught / total_positives if total_positives else float("nan")
    return precision, recall


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1") -> float:
    """Pick a decision threshold by sweeping candidate values.

    Chosen on *validation* data only, then applied unchanged to the holdout —
    tuning the threshold on the holdout would make the holdout a validation set.
    """
    candidates = np.unique(np.quantile(y_prob, np.linspace(0.80, 0.9999, 200)))
    best_threshold, best_score = 0.5, -1.0
    for threshold in candidates:
        predicted = (y_prob >= threshold).astype("int8")
        if predicted.sum() == 0:
            continue
        score = (
            f1_score(y_true, predicted, zero_division=0)
            if metric == "f1"
            else precision_score(y_true, predicted, zero_division=0)
        )
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = None,
    budgets: tuple[float, ...] = (0.001, 0.01, 0.05),
) -> ClassificationMetrics:
    """Compute the full metric bundle.

    Args:
        y_true: Binary ground truth.
        y_prob: Predicted probability of the positive class.
        threshold: Decision threshold. If ``None``, the F1-optimal threshold is
            found on *this* data — appropriate for validation, never for a
            holdout, where the validation-derived threshold must be passed in.
        budgets: Alert-budget fractions to report precision/recall at.
    """
    y_true = np.asarray(y_true).astype("int8")
    y_prob = np.asarray(y_prob, dtype="float64")

    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true contains a single class — metrics are undefined")

    prevalence = float(y_true.mean())
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))

    if threshold is None:
        threshold = find_best_threshold(y_true, y_prob)
    predicted = (y_prob >= threshold).astype("int8")

    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()

    precision_budget: dict[str, float] = {}
    recall_budget: dict[str, float] = {}
    for budget in budgets:
        p, r = precision_at_alert_budget(y_true, y_prob, budget)
        label = f"top_{budget * 100:g}pct"
        precision_budget[label] = p
        recall_budget[label] = r

    return ClassificationMetrics(
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        pr_auc_lift=pr_auc / prevalence if prevalence else float("nan"),
        precision=float(precision_score(y_true, predicted, zero_division=0)),
        recall=float(recall_score(y_true, predicted, zero_division=0)),
        f1=float(f1_score(y_true, predicted, zero_division=0)),
        threshold=float(threshold),
        brier=float(brier_score_loss(y_true, y_prob)),
        prevalence=prevalence,
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
        n_samples=int(y_true.size),
        precision_at_budget=precision_budget,
        recall_at_budget=recall_budget,
    )


def calibration_table(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict[str, list[float]]:
    """Reliability data: mean predicted vs observed rate per probability bin.

    Calibration matters operationally here because the API returns a probability
    that drives a risk band. A model can rank perfectly (high PR-AUC) while its
    probabilities are systematically inflated — which is exactly what
    ``scale_pos_weight`` does — so thresholds set on those numbers would be wrong.
    Quantile bins are used rather than equal-width because predictions cluster
    near zero at this prevalence.
    """
    y_true = np.asarray(y_true).astype("float64")
    y_prob = np.asarray(y_prob, dtype="float64")

    edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:
        raise ValueError("Not enough distinct predictions to build calibration bins")
    bin_ids = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, edges.size - 2)

    mean_predicted, observed_rate, counts = [], [], []
    for b in range(edges.size - 1):
        mask = bin_ids == b
        if not mask.any():
            continue
        mean_predicted.append(float(y_prob[mask].mean()))
        observed_rate.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))

    return {
        "mean_predicted": mean_predicted,
        "observed_rate": observed_rate,
        "count": [float(c) for c in counts],
    }


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Weighted mean absolute gap between predicted and observed rates."""
    table = calibration_table(y_true, y_prob, n_bins=n_bins)
    predicted = np.array(table["mean_predicted"])
    observed = np.array(table["observed_rate"])
    counts = np.array(table["count"])
    return float((counts * np.abs(predicted - observed)).sum() / counts.sum())


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "pr_auc",
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Percentile bootstrap confidence interval for a ranking metric.

    A holdout PR-AUC is a single point estimate over one finite sample, so on
    its own it cannot be compared honestly against a cross-validation band. The
    bootstrap answers the missing question: how much of the gap between holdout
    and CV is sampling noise rather than genuine degradation?

    Resampling is **stratified** — positives and negatives are resampled within
    their own strata — so every resample holds the prevalence fixed. Resampling
    the pooled rows instead would let the fraud rate wander between draws, and
    PR-AUC moves with prevalence by construction, which would widen the interval
    with an artefact of the procedure rather than uncertainty about the model.

    Args:
        y_true: Binary labels.
        y_prob: Predicted scores.
        metric: ``"pr_auc"`` or ``"roc_auc"``.
        n_resamples: Bootstrap draws.
        alpha: Two-sided level; 0.05 gives a 95% interval.
        seed: Seed, so the interval is reproducible.

    Returns:
        Point estimate, interval bounds, bootstrap mean and standard error.
    """
    scorer = average_precision_score if metric == "pr_auc" else roc_auc_score
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    positive_idx = np.flatnonzero(y_true == 1)
    negative_idx = np.flatnonzero(y_true == 0)
    rng = np.random.default_rng(seed)

    draws = np.empty(n_resamples, dtype="float64")
    for i in range(n_resamples):
        take = np.concatenate(
            (
                rng.choice(positive_idx, size=positive_idx.size, replace=True),
                rng.choice(negative_idx, size=negative_idx.size, replace=True),
            )
        )
        draws[i] = scorer(y_true[take], y_prob[take])

    lower, upper = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {
        "metric": metric,
        "point": float(scorer(y_true, y_prob)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "bootstrap_mean": float(draws.mean()),
        "standard_error": float(draws.std(ddof=1)),
        "n_resamples": int(n_resamples),
        "level": float(1 - alpha),
    }
