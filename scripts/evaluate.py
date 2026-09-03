"""Evaluate a trained artifact and produce the figure set.

Separate from ``train.py`` so figures and reports can be regenerated without
retraining, and so the holdout evaluation is an explicit, deliberate action
rather than a side effect.

Note on discipline: ``train.py`` already scores the holdout once and records the
result in the artifact metadata. Re-running this script re-scores the same rows.
That is fine for regenerating plots, but it must not become a loop of
"evaluate, tweak, evaluate" — model selection belongs to the cross-validation
folds. ``--partition validation`` exists for exactly that reason.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --partition modelling
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import TARGET  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    bootstrap_metric_ci,
    calibration_table,
    compute_metrics,
    expected_calibration_error,
)
from src.evaluation.plots import all_evaluation_plots  # noqa: E402
from src.models.artifact import ARTIFACT_FILENAME, ModelArtifact  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402
from src.utils.paths import (  # noqa: E402
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    ensure_dir,
)

logger = logging.getLogger("evaluate")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the trained model.")
    parser.add_argument("--partition", default="holdout", choices=["holdout", "modelling"])
    parser.add_argument("--artifact", type=Path, default=MODELS_DIR / ARTIFACT_FILENAME)
    parser.add_argument("--sample", type=int, default=0, help="0 = use all rows.")
    args = parser.parse_args()

    setup_logging()

    if not args.artifact.is_file():
        logger.error("No artifact at %s — run scripts/train.py first", args.artifact)
        return 1

    path = PROCESSED_DIR / f"{args.partition}_prepared.parquet"
    if not path.is_file():
        logger.error("%s missing — run scripts/build_dataset.py first", path)
        return 1

    artifact = ModelArtifact.load(args.artifact)
    df = pd.read_parquet(path)
    if args.sample and len(df) > args.sample:
        df = df.sample(n=args.sample, random_state=artifact.metadata.seed).sort_index()
    logger.info("Evaluating %s on %s: %s", artifact.metadata.model_name, args.partition, df.shape)

    if TARGET not in df.columns:
        logger.error("%s has no labels — cannot evaluate", path.name)
        return 1

    y_true = df[TARGET].to_numpy()
    y_prob = artifact.predict_proba(df)

    # The threshold comes from the artifact — chosen on validation during
    # training. Re-optimising it here would make the holdout a validation set.
    threshold = artifact.decision_threshold
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    logger.info("%s: %s", args.partition.upper(), metrics.summary())
    logger.info(
        "Alert budgets — precision@0.1%%=%.4f, @1%%=%.4f, @5%%=%.4f",
        metrics.precision_at_budget.get("top_0.1pct", float("nan")),
        metrics.precision_at_budget.get("top_1pct", float("nan")),
        metrics.precision_at_budget.get("top_5pct", float("nan")),
    )

    pr_ci = bootstrap_metric_ci(y_true, y_prob, metric="pr_auc")
    roc_ci = bootstrap_metric_ci(y_true, y_prob, metric="roc_auc")
    logger.info(
        "PR-AUC %.4f  95%% CI [%.4f, %.4f]  (SE %.4f, %d stratified resamples)",
        pr_ci["point"],
        pr_ci["ci_lower"],
        pr_ci["ci_upper"],
        pr_ci["standard_error"],
        pr_ci["n_resamples"],
    )
    logger.info(
        "ROC-AUC %.4f  95%% CI [%.4f, %.4f]",
        roc_ci["point"],
        roc_ci["ci_lower"],
        roc_ci["ci_upper"],
    )

    ece = expected_calibration_error(y_true, y_prob)
    logger.info(
        "Expected calibration error: %.6f (calibrated=%s)", ece, artifact.metadata.calibrated
    )

    figures = all_evaluation_plots(
        y_true, y_prob, threshold, ensure_dir(FIGURES_DIR), prefix=args.partition
    )

    report = {
        "partition": args.partition,
        "n_rows": int(len(df)),
        "model_name": artifact.metadata.model_name,
        "trained_at": artifact.metadata.trained_at,
        "calibrated": artifact.metadata.calibrated,
        "decision_threshold": threshold,
        "expected_calibration_error": ece,
        "metrics": metrics.to_flat_dict(),
        "bootstrap_ci": {"pr_auc": pr_ci, "roc_auc": roc_ci},
        "calibration_table": calibration_table(y_true, y_prob),
        "figures": [str(p.relative_to(REPORTS_DIR.parent)) for p in figures],
    }
    out = ensure_dir(REPORTS_DIR) / f"evaluation_{args.partition}.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s and %d figures", out.name, len(figures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
