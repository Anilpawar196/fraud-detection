"""Fitted (stateful) encoders — the leakage-critical part of the feature layer.

Unlike :mod:`src.features.velocity`, everything here is a **population
statistic**: a count over rows, or a per-entity mean. Those quantities are
contaminated the moment they are computed over data the model will be evaluated
on, because they smuggle information about the future population into a past
row.

Two rules are therefore enforced by the interface rather than by discipline:

1. ``fit`` is only ever called on a training partition. ``transform`` is applied
   to validation/holdout/serving data as a pure lookup.
2. Nothing here reads the target. Frequency encoding is preferred over target
   encoding precisely because it cannot leak the label at all. Target encoding
   is not implemented: given the temporal folds and recurring entities, its
   fold-internal variant would add substantial complexity for a gain that
   frequency encoding largely already captures.

Note on the deliberate choice *not* to fit on train+test combined: fitting
frequencies over the union is a well-known Kaggle score-booster and it is
transductive leakage — it assumes the scoring population is known while
training. A deployed API scores one transaction at a time and cannot see the
future population, so counts come from the training window only. This costs
leaderboard points and is the correct engineering decision for a serving system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FrequencyEncoder:
    """Replaces a categorical value with how often it occurred in training.

    Mechanism: rare cards, rare device fingerprints and rare email domains are
    disproportionately fraudulent, while the very highest-frequency values tend
    to be shared infrastructure. Count is a monotone proxy for both, needs no
    label, and handles high cardinality without one-hot width.

    Unseen values map to 0 — literally true ("never observed in the training
    window") and distinguishable by a tree from any positive count.
    """

    columns: list[str] = field(default_factory=list)
    counts: dict[str, dict] = field(default_factory=dict)
    min_count: int = 1
    suffix: str = "_freq"

    def fit(self, df: pd.DataFrame, columns: list[str]) -> FrequencyEncoder:
        self.columns = [c for c in columns if c in df.columns]
        self.counts = {}
        for col in self.columns:
            counts = df[col].value_counts(dropna=True)
            if self.min_count > 1:
                counts = counts[counts >= self.min_count]
            self.counts[col] = counts.to_dict()
        logger.info("FrequencyEncoder fitted on %d columns", len(self.columns))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only the new frequency columns, indexed like ``df``.

        Returning new columns instead of mutating the input keeps this a pure
        function of its argument — the caller concatenates. That avoids both
        pandas' SettingWithCopy ambiguity on sliced frames and the quadratic
        fragmentation cost of inserting columns one at a time into a 500-column
        frame.
        """
        output: dict[str, pd.Series] = {}
        for col in self.columns:
            if col not in df.columns:
                raise KeyError(f"FrequencyEncoder: column {col!r} missing at transform")
            mapped = df[col].map(self.counts[col])
            output[f"{col.lstrip('_')}{self.suffix}"] = (
                pd.to_numeric(mapped, errors="coerce").fillna(0).astype("float32")
            )
        return pd.DataFrame(output, index=df.index)

    def fit_transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        self.fit(df, columns)
        return pd.concat([df, self.transform(df)], axis=1)


@dataclass
class EntityAmountAggregator:
    """Per-entity amount statistics, and each row's deviation from them.

    Mechanism: the measured absolute difference between fraud and legitimate
    amounts is small (mean 149.24 vs 134.51; median 75.00 vs 68.50), so absolute
    amount is weak on its own. What is informative is a transaction that is
    anomalous *for that account* — a card whose history is £10 coffees suddenly
    charging £400. That requires a per-entity baseline, which is exactly what
    this fits.

    Unseen entities yield NaN rather than a global fallback: "no history for this
    card" is a distinct and genuinely informative state, and trees handle it as
    a missing branch.
    """

    entity_columns: list[str] = field(default_factory=list)
    amount_column: str = "log_amount"
    stats: dict[str, pd.DataFrame] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, entity_columns: list[str]) -> EntityAmountAggregator:
        self.entity_columns = [c for c in entity_columns if c in df.columns]
        self.stats = {}
        for col in self.entity_columns:
            grouped = df.groupby(col, observed=True)[self.amount_column].agg(
                ["mean", "std", "count"]
            )
            self.stats[col] = grouped.astype("float32")
        logger.info(
            "EntityAmountAggregator fitted on %d entity keys (%s)",
            len(self.entity_columns),
            ", ".join(f"{c}:{len(self.stats[c]):,} groups" for c in self.entity_columns),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only the new deviation columns, indexed like ``df``."""
        values = df[self.amount_column].to_numpy(dtype="float32")
        output: dict[str, np.ndarray] = {}
        for col in self.entity_columns:
            stats = self.stats[col]
            mean = df[col].map(stats["mean"]).to_numpy(dtype="float32")
            std = df[col].map(stats["std"]).to_numpy(dtype="float32")
            name = col.lstrip("_")

            output[f"{name}_amt_mean_hist"] = mean
            output[f"{name}_amt_diff_from_mean"] = (values - mean).astype("float32")
            with np.errstate(invalid="ignore", divide="ignore"):
                # std == 0 means the entity has a single historical amount; a
                # z-score is undefined there, so leave it missing rather than
                # emitting inf.
                safe_std = np.where((std > 0) & np.isfinite(std), std, np.nan)
                output[f"{name}_amt_zscore"] = ((values - mean) / safe_std).astype("float32")
        return pd.DataFrame(output, index=df.index)

    def fit_transform(self, df: pd.DataFrame, entity_columns: list[str]) -> pd.DataFrame:
        self.fit(df, entity_columns)
        return pd.concat([df, self.transform(df)], axis=1)
