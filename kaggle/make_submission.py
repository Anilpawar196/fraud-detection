"""Score the Kaggle test set and write a submission file.

Kept out of ``scripts/`` deliberately: nothing in the production pipeline
depends on this, and the leaderboard is not how this project judges itself.

Two choices worth stating, because both cost leaderboard rank on purpose:

1. **Uncalibrated scores.** IEEE-CIS is judged on ROC-AUC, which only reads the
   *ordering* of predictions. The isotonic calibrator is a monotone step
   function, so it cannot improve that ordering and its flat segments create
   ties that can only hurt. Calibration stays in the served model, where
   probabilities must mean something; it is skipped here.

2. **Encoders fitted on train only.** Most leaderboard solutions fit frequency
   and target encodings over train+test together, which leaks test distribution
   into training and is worth real rank. This pipeline refuses to, because the
   model is meant to work on transactions that have not happened yet.

Usage:
    python kaggle/make_submission.py
    python kaggle/make_submission.py --artifact models/model_artifact.pkl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.artifact import ModelArtifact  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402
from src.utils.paths import PROCESSED_DIR, ROOT  # noqa: E402

logger = logging.getLogger("kaggle")

ID = "TransactionID"
TARGET = "isFraud"


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a Kaggle submission CSV.")
    parser.add_argument("--artifact", type=Path, default=ROOT / "models" / "model_artifact.pkl")
    parser.add_argument("--test", type=Path, default=PROCESSED_DIR / "test_prepared.parquet")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "submission.csv")
    parser.add_argument(
        "--calibrated",
        action="store_true",
        help="Use calibrated probabilities (expected to score slightly worse on ROC-AUC).",
    )
    args = parser.parse_args()

    setup_logging()

    if not args.test.is_file():
        logger.error("Missing %s - run `python scripts/build_dataset.py` first.", args.test)
        return 1
    if not args.artifact.is_file():
        logger.error("Missing %s - run `python scripts/train.py` first.", args.artifact)
        return 1

    artifact = ModelArtifact.load(args.artifact)
    test = pd.read_parquet(args.test)
    logger.info("Scoring %s rows with %s", f"{len(test):,}", artifact.metadata.model_name)

    if args.calibrated:
        scores = artifact.predict_proba(test)
        logger.info("Using CALIBRATED scores (ties may cost ROC-AUC)")
    else:
        scores = artifact.raw_probability(test)
        logger.info("Using raw uncalibrated scores (correct for a ranking metric)")

    submission = pd.DataFrame({ID: test[ID].to_numpy(), TARGET: scores})

    # Fail loudly rather than upload a file Kaggle will silently reject.
    if submission[ID].duplicated().any():
        logger.error("Duplicate %s values in submission", ID)
        return 1
    if submission[TARGET].isna().any():
        logger.error("NaN scores in submission")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)

    logger.info(
        "Wrote %s (%s rows) | score range %.6f - %.6f | mean %.4f",
        args.out,
        f"{len(submission):,}",
        submission[TARGET].min(),
        submission[TARGET].max(),
        submission[TARGET].mean(),
    )
    logger.info("Training-period fraud rate was 3.51%% - compare against the mean above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
