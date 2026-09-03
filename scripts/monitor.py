"""Drift monitoring: training distribution vs a later period.

**On honesty about the traffic.** This project has no production users, so nothing
here is production traffic. But the comparison is not synthetic either: the
default is the training period against the **real, unlabeled IEEE-CIS test period,
which begins 30 days after training ends**. That is genuine covariate shift on
genuine data, which is a far better demonstration than perturbing a copy of the
training set.

Two statistics, because they answer different questions:

* **PSI** bins the reference distribution and compares mass per bin. Interpretable
  on a fixed scale and insensitive to sample size, which is why it is the trigger.
* **KS** tests the largest CDF gap, catching shape changes PSI's binning smooths
  over — but its p-value goes to zero for *any* difference once n is large, so it
  is reported alongside rather than used as the alarm.

Per-feature missing-rate deltas come out of the same pass, which is why a separate
data-quality module is unnecessary.

Outputs (``reports/monitoring/``):
    feature_drift.csv        PSI, KS and missing-rate delta per feature
    monitoring_summary.json  verdict counts, prediction drift, top drifted features

Usage:
    python scripts/monitor.py                    # train vs test period
    python scripts/monitor.py --current holdout  # train vs holdout period
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
from src.models.artifact import ARTIFACT_FILENAME, ModelArtifact  # noqa: E402
from src.monitoring.drift import feature_drift, prediction_drift, summarise  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402
from src.utils.paths import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_dir  # noqa: E402

logger = logging.getLogger("monitor")

MONITORING_DIR = REPORTS_DIR / "monitoring"

#: Rows sampled per period. PSI is stable well below the full 472k rows, and
#: sampling keeps the report runnable in constrained memory.
SAMPLE_ROWS = 60_000


def _load(name: str, sample: int, seed: int) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}_prepared.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} missing. Run: python scripts/build_dataset.py"
            + (" --with-test" if name == "test" else "")
        )
    df = pd.read_parquet(path)
    if len(df) > sample:
        df = df.sample(n=sample, random_state=seed).sort_index()
    logger.info("Loaded %s: %s", name, df.shape)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PSI/KS drift monitoring.")
    parser.add_argument("--current", default="test", choices=["test", "holdout"])
    parser.add_argument("--sample", type=int, default=SAMPLE_ROWS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    setup_logging()
    ensure_dir(MONITORING_DIR)

    reference = _load("modelling", args.sample, args.seed)
    try:
        current = _load(args.current, args.sample, args.seed)
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 1

    # Compare only model-input features: internal keys and the target are not
    # what the model sees, so drift in them is not actionable.
    exclude = {TARGET, "TransactionID", "TransactionDT"}
    features = [
        c
        for c in reference.columns
        if c in current.columns and c not in exclude and not c.startswith("_")
    ]
    logger.info("Comparing %d features: modelling vs %s", len(features), args.current)

    # --- feature drift (PSI + KS + missing-rate delta) ----------------------
    drift_table = feature_drift(reference[features], current[features])
    drift_path = MONITORING_DIR / "feature_drift.csv"
    drift_table.to_csv(drift_path, index=False)
    drift_summary = summarise(drift_table, top_n=args.top)
    logger.info(
        "Feature drift: %d significant, %d moderate, %d stable (of %d)",
        drift_summary["significant"],
        drift_summary["moderate"],
        drift_summary["stable"],
        drift_summary["n_features"],
    )
    logger.info(
        "Most drifted: %s",
        ", ".join(f"{r['feature']} (PSI {r['psi']:.3f})" for r in drift_summary["top"][:8]),
    )

    # --- prediction drift ---------------------------------------------------
    prediction_report: dict[str, object] = {"status": "model_unavailable"}
    artifact_path = MODELS_DIR / ARTIFACT_FILENAME
    if artifact_path.is_file():
        artifact = ModelArtifact.load(artifact_path)
        reference_scores = artifact.predict_proba(reference)
        current_scores = artifact.predict_proba(current)
        prediction_report = prediction_drift(reference_scores, current_scores)
        prediction_report["n_reference"] = int(len(reference_scores))
        prediction_report["n_current"] = int(len(current_scores))
        logger.info(
            "Prediction drift: PSI %.4f (%s) | mean %.4f -> %.4f",
            prediction_report["psi"],
            prediction_report["verdict"],
            prediction_report["reference_mean"],
            prediction_report["current_mean"],
        )
    else:
        logger.warning("No model artifact — skipping prediction drift")

    report = {
        "traffic_source": (
            "real unlabeled IEEE-CIS test period (starts 30 days after training ends)"
            if args.current == "test"
            else "held-out tail of the training period"
        ),
        "disclaimer": (
            "Not production traffic. This project has no live users; the comparison "
            "uses real dataset periods, not synthetic perturbations."
        ),
        "reference": "modelling_prepared",
        "current": f"{args.current}_prepared",
        "n_features_compared": len(features),
        "feature_drift": drift_summary,
        "prediction_drift": prediction_report,
    }
    summary_path = MONITORING_DIR / "monitoring_summary.json"
    summary_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s and %s", drift_path.name, summary_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
