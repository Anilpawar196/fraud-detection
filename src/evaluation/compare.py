"""Model comparison tables.

Produces ``reports/model_comparison.csv``. Two columns exist specifically to
keep the comparison honest rather than flattering:

* ``n_train_rows`` / ``subsampled`` — the dense-matrix models (Logistic
  Regression, Random Forest) may be fitted on capped row counts because a
  590k x ~550 dense float32 matrix does not fit in this machine's headroom.
  Hiding that would misrepresent the comparison, so it is a column.
* ``pr_auc_lift`` — PR-AUC divided by prevalence. An absolute PR-AUC of 0.6
  means nothing without the ~0.035 no-skill floor next to it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import ClassificationMetrics
from src.models.training import CVResult
from src.utils.paths import REPORTS_DIR, ensure_dir

logger = logging.getLogger(__name__)

#: Column order for the comparison table.
COMPARISON_COLUMNS = (
    "model",
    "evaluation",
    "roc_auc",
    "pr_auc",
    "pr_auc_std",
    "pr_auc_lift",
    "precision",
    "recall",
    "f1",
    "brier",
    "precision_at_top_1pct",
    "recall_at_top_1pct",
    "threshold",
    "n_features",
    "n_train_rows",
    "subsampled",
    "training_time_seconds",
)


def cv_result_row(result: CVResult) -> dict[str, object]:
    """One row summarising a cross-validation result."""
    last_fold = result.fold_results[-1] if result.fold_results else None
    mean_precision_at_1pct = (
        sum(
            f.metrics.precision_at_budget.get("top_1pct", float("nan")) for f in result.fold_results
        )
        / len(result.fold_results)
        if result.fold_results
        else float("nan")
    )
    mean_recall_at_1pct = (
        sum(f.metrics.recall_at_budget.get("top_1pct", float("nan")) for f in result.fold_results)
        / len(result.fold_results)
        if result.fold_results
        else float("nan")
    )
    return {
        "model": result.model_name,
        "evaluation": "cv_temporal_mean",
        "roc_auc": result.mean_metric("roc_auc"),
        "pr_auc": result.mean_metric("pr_auc"),
        "pr_auc_std": result.std_metric("pr_auc"),
        "pr_auc_lift": result.mean_metric("pr_auc_lift"),
        "precision": result.mean_metric("precision"),
        "recall": result.mean_metric("recall"),
        "f1": result.mean_metric("f1"),
        "brier": result.mean_metric("brier"),
        "precision_at_top_1pct": mean_precision_at_1pct,
        "recall_at_top_1pct": mean_recall_at_1pct,
        "threshold": result.mean_metric("threshold"),
        "n_features": last_fold.n_features if last_fold else None,
        "n_train_rows": result.n_train_rows_used,
        "subsampled": result.subsampled,
        "training_time_seconds": round(result.total_train_seconds, 1),
    }


def metrics_row(
    model_name: str,
    evaluation: str,
    metrics: ClassificationMetrics,
    n_features: int | None = None,
    n_train_rows: int | None = None,
    training_time_seconds: float | None = None,
    subsampled: bool = False,
) -> dict[str, object]:
    """One row summarising a single-dataset evaluation (e.g. the holdout)."""
    return {
        "model": model_name,
        "evaluation": evaluation,
        "roc_auc": metrics.roc_auc,
        "pr_auc": metrics.pr_auc,
        "pr_auc_std": float("nan"),
        "pr_auc_lift": metrics.pr_auc_lift,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "brier": metrics.brier,
        "precision_at_top_1pct": metrics.precision_at_budget.get("top_1pct", float("nan")),
        "recall_at_top_1pct": metrics.recall_at_budget.get("top_1pct", float("nan")),
        "threshold": metrics.threshold,
        "n_features": n_features,
        "n_train_rows": n_train_rows,
        "subsampled": subsampled,
        "training_time_seconds": (
            round(training_time_seconds, 1) if training_time_seconds is not None else None
        ),
    }


def build_comparison(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Assemble rows into the canonical comparison table, best PR-AUC first."""
    table = pd.DataFrame(rows)
    for column in COMPARISON_COLUMNS:
        if column not in table.columns:
            table[column] = None
    table = table[list(COMPARISON_COLUMNS)]
    return table.sort_values(["evaluation", "pr_auc"], ascending=[True, False]).reset_index(
        drop=True
    )


def save_comparison(table: pd.DataFrame, path: Path | None = None) -> Path:
    """Write the comparison table to CSV."""
    out = path or (ensure_dir(REPORTS_DIR) / "model_comparison.csv")
    table.to_csv(out, index=False, float_format="%.6f")
    logger.info("Wrote %s (%d rows)", out, len(table))
    return out
