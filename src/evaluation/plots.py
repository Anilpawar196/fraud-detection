"""Evaluation figures.

Matplotlib is forced onto the Agg backend because these run in scripts and CI,
where no display exists.

The precision-recall curve is plotted with the prevalence baseline drawn on it:
a PR curve without that reference line invites reading 0.6 as "not great", when
here it is roughly 17x the no-skill floor.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.evaluation.metrics import calibration_table  # noqa: E402

logger = logging.getLogger(__name__)


def precision_recall_plot(
    y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, label: str = "model"
) -> Path:
    """Precision-recall curve with the prevalence baseline."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    prevalence = float(np.mean(y_true))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color="#c2410c", lw=2, label=f"{label} (PR-AUC {pr_auc:.4f})")
    plt.axhline(
        prevalence,
        color="#64748b",
        ls="--",
        lw=1.2,
        label=f"no-skill baseline ({prevalence:.4f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall — {pr_auc / prevalence:.1f}x baseline")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close("all")
    logger.info("Wrote %s", out_path)
    return out_path


def roc_plot(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, label: str = "model") -> Path:
    """ROC curve with the chance diagonal."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#1d4ed8", lw=2, label=f"{label} (ROC-AUC {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#64748b", ls="--", lw=1.2, label="chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close("all")
    logger.info("Wrote %s", out_path)
    return out_path


def calibration_plot(
    y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, n_bins: int = 10
) -> Path:
    """Reliability diagram against the perfect-calibration diagonal."""
    table = calibration_table(y_true, y_prob, n_bins=n_bins)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], color="#64748b", ls="--", lw=1.2, label="perfectly calibrated")
    plt.plot(
        table["mean_predicted"],
        table["observed_rate"],
        marker="o",
        color="#047857",
        lw=1.8,
        label="model",
    )
    upper = max(max(table["mean_predicted"]), max(table["observed_rate"])) * 1.1 or 1.0
    plt.xlim(0, upper)
    plt.ylim(0, upper)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraud rate")
    plt.title("Calibration (quantile bins)")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close("all")
    logger.info("Wrote %s", out_path)
    return out_path


def confusion_matrix_plot(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, out_path: Path
) -> Path:
    """Confusion matrix at the operating threshold."""
    predicted = (np.asarray(y_prob) >= threshold).astype("int8")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predicted,
        display_labels=["legit", "fraud"],
        cmap="Oranges",
        colorbar=False,
        ax=axis,
    )
    axis.set_title(f"Confusion matrix @ threshold {threshold:.4f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close("all")
    logger.info("Wrote %s", out_path)
    return out_path


def score_distribution_plot(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> Path:
    """Predicted-score distributions for each class, on a log y scale."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    bins = np.linspace(0, 1, 51)
    plt.hist(y_prob[y_true == 0], bins=bins, alpha=0.65, label="legit", color="#1d4ed8")
    plt.hist(y_prob[y_true == 1], bins=bins, alpha=0.75, label="fraud", color="#c2410c")
    # Log scale because legitimate transactions outnumber fraud ~27:1; on a
    # linear axis the fraud histogram is invisible.
    plt.yscale("log")
    plt.xlabel("Predicted fraud probability")
    plt.ylabel("Count (log scale)")
    plt.title("Score distribution by true class")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close("all")
    logger.info("Wrote %s", out_path)
    return out_path


def all_evaluation_plots(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    figures_dir: Path,
    prefix: str = "holdout",
) -> list[Path]:
    """Produce the full evaluation figure set."""
    return [
        precision_recall_plot(y_true, y_prob, figures_dir / f"{prefix}_precision_recall.png"),
        roc_plot(y_true, y_prob, figures_dir / f"{prefix}_roc.png"),
        calibration_plot(y_true, y_prob, figures_dir / f"{prefix}_calibration.png"),
        confusion_matrix_plot(
            y_true, y_prob, threshold, figures_dir / f"{prefix}_confusion_matrix.png"
        ),
        score_distribution_plot(y_true, y_prob, figures_dir / f"{prefix}_score_distribution.png"),
    ]
