"""Leakage-safe temporal splitting.

Why not random stratified K-fold — three measured reasons, not a preference:

1. Train spans days 1–183 and the Kaggle test set spans days 213–396: strictly
   disjoint, with a 30.0-day gap and zero shared timestamps. Random CV measures
   within-window interpolation; deployment requires forward extrapolation.
2. Fraud prevalence is non-stationary across 30-day blocks (2.4762% → 4.0373%
   → 4.0319% → 3.9265% → 3.4723% → 3.4013% → 4.1795%), so randomly drawn folds
   are not exchangeable samples from one distribution.
3. Entities (cards, devices) recur across rows. Random folds place the same
   entity on both sides of the boundary, letting the model memorise the entity
   rather than learn fraud.

Chronological splitting was verified feasible before being adopted:
``TransactionDT`` is complete, non-negative, already monotonic in file order,
covers 182 contiguous days with no missing day and no gap wider than 1.15 h.
5.746% of rows share a timestamp with another row (max 8), so split boundaries
are snapped to timestamp edges — a tie group is never cut in half.

``random_stratified_folds`` is provided deliberately, not as an alternative but
as the control for a documented experiment quantifying how optimistic random CV
is on this data.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schema import SECONDS_PER_DAY, TARGET, TIME_COL
from src.utils.paths import PROCESSED_DIR, ensure_dir

logger = logging.getLogger(__name__)


class SplitError(ValueError):
    """Raised when a requested split cannot be produced safely."""


@dataclass(frozen=True)
class SplitMetadata:
    """Everything needed to reproduce and audit a split."""

    strategy: str
    n_total: int
    n_train: int
    n_validation: int
    n_holdout: int
    holdout_cut_dt: int
    validation_cut_dt: int
    train_fraud_rate: float
    validation_fraud_rate: float
    holdout_fraud_rate: float
    train_day_range: tuple[int, int]
    validation_day_range: tuple[int, int]
    holdout_day_range: tuple[int, int]


@dataclass
class TemporalSplit:
    """Positional indices for the three partitions, plus audit metadata."""

    train_idx: np.ndarray
    validation_idx: np.ndarray
    holdout_idx: np.ndarray
    metadata: SplitMetadata


def find_time_cut(timestamps: pd.Series, fraction: float) -> int:
    """Find a timestamp cut placing ``fraction`` of rows at or before it.

    The cut is snapped to a timestamp boundary so that all rows sharing a
    timestamp land on the same side. Without this, a tie group straddling the
    boundary would put near-simultaneous transactions — plausibly the same card
    -testing burst — in both partitions.

    Returns:
        The largest timestamp ``t`` for which ``P(ts <= t) <= fraction``.
    """
    if not 0.0 < fraction < 1.0:
        raise SplitError(f"fraction must be in (0, 1), got {fraction}")

    values, counts = np.unique(timestamps.to_numpy(), return_counts=True)
    cumulative = np.cumsum(counts) / counts.sum()
    eligible = np.nonzero(cumulative <= fraction)[0]
    if eligible.size == 0:
        raise SplitError(
            f"No timestamp boundary yields <= {fraction:.3f} of rows; the first "
            "timestamp group is larger than the requested fraction."
        )
    return int(values[eligible[-1]])


def temporal_split(
    df: pd.DataFrame,
    holdout_fraction: float = 0.20,
    validation_fraction: float = 0.20,
) -> TemporalSplit:
    """Split chronologically into train / validation / holdout.

    Layout along the time axis::

        |<---------- training period ---------->|<-- holdout -->|
        |<---- train ---->|<-- validation -->|  |               |
        day 1                                                day 182

    The holdout is the final ``holdout_fraction`` of rows by time and is scored
    exactly once, by one model, at the end. Validation is the final
    ``validation_fraction`` of what remains and drives all model selection.

    Args:
        df: Chronologically sorted frame containing ``TransactionDT`` and the target.
        holdout_fraction: Tail fraction reserved as the untouched holdout.
        validation_fraction: Fraction of the remaining period used for validation.
    """
    if TIME_COL not in df.columns:
        raise SplitError(f"Missing {TIME_COL}")
    ts = df[TIME_COL]
    if not ts.is_monotonic_increasing:
        raise SplitError(f"{TIME_COL} must be sorted ascending before splitting")

    holdout_cut = find_time_cut(ts, 1.0 - holdout_fraction)
    is_holdout = (ts > holdout_cut).to_numpy()

    train_period = ts[~is_holdout]
    validation_cut = find_time_cut(train_period, 1.0 - validation_fraction)
    is_validation = (~is_holdout) & (ts > validation_cut).to_numpy()
    is_train = (~is_holdout) & (~is_validation)

    train_idx = np.nonzero(is_train)[0]
    validation_idx = np.nonzero(is_validation)[0]
    holdout_idx = np.nonzero(is_holdout)[0]

    for name, idx in (
        ("train", train_idx),
        ("validation", validation_idx),
        ("holdout", holdout_idx),
    ):
        if idx.size == 0:
            raise SplitError(f"{name} partition is empty — check the fractions")

    day = (ts // SECONDS_PER_DAY).to_numpy()
    target = df[TARGET].to_numpy() if TARGET in df.columns else None

    def rate(idx: np.ndarray) -> float:
        return float(target[idx].mean()) if target is not None else float("nan")

    def days(idx: np.ndarray) -> tuple[int, int]:
        return int(day[idx].min()), int(day[idx].max())

    metadata = SplitMetadata(
        strategy="temporal_tail_holdout",
        n_total=len(df),
        n_train=int(train_idx.size),
        n_validation=int(validation_idx.size),
        n_holdout=int(holdout_idx.size),
        holdout_cut_dt=holdout_cut,
        validation_cut_dt=validation_cut,
        train_fraud_rate=rate(train_idx),
        validation_fraud_rate=rate(validation_idx),
        holdout_fraud_rate=rate(holdout_idx),
        train_day_range=days(train_idx),
        validation_day_range=days(validation_idx),
        holdout_day_range=days(holdout_idx),
    )

    # Hard guarantee: no timestamp appears in more than one partition.
    _assert_disjoint_in_time(ts.to_numpy(), train_idx, validation_idx, holdout_idx)

    logger.info(
        "Temporal split | train %d (days %d-%d, fraud %.4f%%) | "
        "val %d (days %d-%d, fraud %.4f%%) | holdout %d (days %d-%d, fraud %.4f%%)",
        metadata.n_train,
        *metadata.train_day_range,
        metadata.train_fraud_rate * 100,
        metadata.n_validation,
        *metadata.validation_day_range,
        metadata.validation_fraud_rate * 100,
        metadata.n_holdout,
        *metadata.holdout_day_range,
        metadata.holdout_fraud_rate * 100,
    )
    return TemporalSplit(train_idx, validation_idx, holdout_idx, metadata)


def _assert_disjoint_in_time(ts: np.ndarray, *partitions: np.ndarray) -> None:
    """Assert no timestamp value is shared between any two partitions."""
    seen: set[int] = set()
    for idx in partitions:
        values = set(np.unique(ts[idx]).tolist())
        overlap = values & seen
        if overlap:
            raise SplitError(
                f"{len(overlap)} timestamps appear in more than one partition — "
                "a tie group was cut across a boundary"
            )
        seen |= values


class PurgedForwardChainingCV:
    """Forward-chaining cross-validation with a purge gap.

    Fold ``i`` trains on everything up to the start of validation block
    ``i + 1``, minus a purge gap, and validates on block ``i + 1``::

        fold 0:  [train][gap][val]
        fold 1:  [    train   ][gap][val]
        fold 2:  [        train     ][gap][val]

    The purge gap matters specifically because of the velocity features: a
    trailing 168-hour aggregate computed for a row at the start of the
    validation block reaches back into training rows. Without a gap at least as
    wide as the longest look-back window, the feature values themselves would
    straddle the boundary even though the *rows* do not.

    Blocks are equal-count (quantile-based) rather than equal-duration, so each
    fold's validation set has a comparable number of positives — important at
    3.5% prevalence.
    """

    def __init__(self, n_splits: int = 5, purge_days: int = 7) -> None:
        if n_splits < 2:
            raise SplitError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.purge_seconds = purge_days * SECONDS_PER_DAY

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: ARG002
        """scikit-learn compatibility."""
        return self.n_splits

    def split(
        self,
        X: pd.DataFrame,
        y=None,  # noqa: ARG002 - unused; required by the scikit-learn splitter protocol
        groups=None,  # noqa: ARG002 - unused; required by the scikit-learn splitter protocol
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, validation_idx)`` positional index pairs."""
        if TIME_COL not in X.columns:
            raise SplitError(f"{TIME_COL} required for temporal CV")
        ts = X[TIME_COL].to_numpy()
        if not np.all(np.diff(ts) >= 0):
            raise SplitError(f"{TIME_COL} must be sorted ascending")

        boundaries = self._block_boundaries(X[TIME_COL])
        for i in range(self.n_splits):
            val_start, val_end = boundaries[i], boundaries[i + 1]
            val_mask = (ts > val_start) & (ts <= val_end)
            train_mask = ts <= (val_start - self.purge_seconds)

            train_idx = np.nonzero(train_mask)[0]
            val_idx = np.nonzero(val_mask)[0]
            if train_idx.size == 0 or val_idx.size == 0:
                raise SplitError(
                    f"Fold {i} is degenerate (train={train_idx.size}, val={val_idx.size}). "
                    f"Reduce n_splits ({self.n_splits}) or purge_days ({self.purge_days})."
                )
            logger.debug(
                "fold %d | train %d rows (<= %d) | val %d rows (%d, %d]",
                i,
                train_idx.size,
                val_start - self.purge_seconds,
                val_idx.size,
                val_start,
                val_end,
            )
            yield train_idx, val_idx

    def _block_boundaries(self, timestamps: pd.Series) -> list[int]:
        """Equal-count time boundaries defining ``n_splits + 1`` blocks."""
        fractions = [(i + 1) / (self.n_splits + 1) for i in range(self.n_splits)]
        boundaries = [find_time_cut(timestamps, f) for f in fractions]
        boundaries = [int(timestamps.min()) - 1, *boundaries, int(timestamps.max())]
        # Drop the synthetic lower bound: blocks are (b[i], b[i+1]].
        return boundaries[1:]


def random_stratified_folds(
    y: pd.Series, n_splits: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Random stratified folds — the *control* for the leakage experiment.

    Used once, to measure how much more optimistic random CV is than the
    time-aware scheme on this dataset. Never used for model selection.
    """
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y))


def save_split(split: TemporalSplit, out_dir: Path | None = None) -> Path:
    """Persist split indices and metadata so every run uses identical partitions."""
    directory = ensure_dir(out_dir or PROCESSED_DIR)
    np.savez_compressed(
        directory / "split_indices.npz",
        train=split.train_idx,
        validation=split.validation_idx,
        holdout=split.holdout_idx,
    )
    meta_path = directory / "split_metadata.json"
    meta_path.write_text(json.dumps(asdict(split.metadata), indent=2), encoding="utf-8")
    logger.info("Saved split indices and metadata to %s", directory)
    return meta_path


def load_split(out_dir: Path | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """Load persisted split indices and metadata."""
    directory = out_dir or PROCESSED_DIR
    with np.load(directory / "split_indices.npz") as data:
        indices = {k: data[k] for k in ("train", "validation", "holdout")}
    metadata = json.loads((directory / "split_metadata.json").read_text(encoding="utf-8"))
    return indices, metadata


def save_folds(
    folds: list[tuple[np.ndarray, np.ndarray]], name: str, out_dir: Path | None = None
) -> Path:
    """Persist CV fold indices so all models are compared on identical folds."""
    directory = ensure_dir(out_dir or PROCESSED_DIR)
    payload: dict[str, np.ndarray] = {}
    for i, (train_idx, val_idx) in enumerate(folds):
        payload[f"fold{i}_train"] = train_idx
        payload[f"fold{i}_val"] = val_idx
    path = directory / f"folds_{name}.npz"
    np.savez_compressed(path, n_folds=np.array([len(folds)]), **payload)
    logger.info("Saved %d %s folds to %s", len(folds), name, path.name)
    return path


def load_folds(name: str, out_dir: Path | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load persisted CV fold indices."""
    directory = out_dir or PROCESSED_DIR
    with np.load(directory / f"folds_{name}.npz") as data:
        n = int(data["n_folds"][0])
        return [(data[f"fold{i}_train"], data[f"fold{i}_val"]) for i in range(n)]
