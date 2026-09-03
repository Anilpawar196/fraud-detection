"""Feature pipeline orchestration.

The pipeline has a deliberately unusual three-phase shape, and the shape *is*
the leakage control:

``prepare(df)``
    Stateless + causal features (row-local transforms, past-only velocity).
    Safe to run on the entire frame before splitting, because nothing is fitted
    and nothing reads a future row.

``fit(train_df)``
    Learns population statistics — frequency counts, per-entity amount
    baselines, categorical vocabularies — from the **training partition only**.

``transform(df)``
    Pure lookup. Applied identically to validation, holdout, and live serving
    traffic.

Splitting these apart makes it structurally impossible to accidentally fit an
encoder on validation data: the only method that learns anything takes the
training frame as its argument.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data.preprocessing import CategoricalCodeEncoder, _is_categorical_like
from src.data.schema import EXCLUDED_FROM_FEATURES, KEY, TIME_COL
from src.features.aggregations import EntityAmountAggregator, FrequencyEncoder
from src.features.builders import (
    AGGREGATION_KEY_COLUMNS,
    DAY_COL,
    ENTITY_KEY_COLUMNS,
    UID_COLUMN,
    add_uid_entity_key,
    add_v_nan_group_features,
    build_stateless_features,
    v_nan_groups,
)
from src.features.velocity import add_velocity_features

logger = logging.getLogger(__name__)

#: Columns frequency-encoded. High-cardinality identifiers where "how common is
#: this value" is more useful to a tree than the value itself.
FREQUENCY_COLUMNS: tuple[str, ...] = (
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
    "id_30",
    "id_31",
    "id_33",
    "_entity_card",
    "_entity_card_addr",
    "_entity_card_full",
    # How many transactions this synthetic account has. Fitted on train only,
    # so unseen accounts map to 0 rather than leaking a test-period count.
    "_entity_uid",
)

#: Helper columns used to build features, then dropped. Absolute time and raw
#: entity keys must never reach the model (audit §7.6, §7.7).
INTERNAL_PREFIX = "_"


def _ensure_uid(df: pd.DataFrame) -> pd.DataFrame:
    """Add the uid entity key if the frame predates it.

    ``_entity_uid`` is produced by :func:`build_stateless_features`, so freshly
    built datasets already carry it. Parquet written before the key existed does
    not, and rebuilding the whole dataset to add one row-local column would be
    wasteful. The key depends only on values already present in the frame, so
    recomputing it here is identical to having built it earlier.
    """
    if UID_COLUMN in df.columns:
        return df
    return add_uid_entity_key(df.copy())


@dataclass
class FeaturePipeline:
    """Fit/transform feature pipeline with explicit leakage boundaries."""

    velocity_windows_hours: tuple[int, ...] = (1, 24, 168)
    anchor_d_columns: bool = True
    #: Feature-name suffixes dropped from the model input. Empty by default, so
    #: the pipeline is unchanged unless a run opts in. Exists to make feature
    #: ablations a config flag rather than a code edit — see the D*_anchored
    #: ablation in the README, where drift showed those features harm temporal
    #: generalisation despite ranking highly by SHAP.
    exclude_feature_suffixes: tuple[str, ...] = ()
    frequency_min_count: int = 1
    frequency_encoder: FrequencyEncoder = field(default_factory=FrequencyEncoder)
    entity_aggregator: EntityAmountAggregator = field(default_factory=EntityAmountAggregator)
    categorical_encoder: CategoricalCodeEncoder = field(default_factory=CategoricalCodeEncoder)
    #: V missingness blocks learned from the training partition. Fitted rather
    #: than derived per frame: the grouping depends on which rows are present,
    #: so recomputing it at transform time would change the feature set and let
    #: validation rows influence their own encoding.
    v_nan_groups: dict[int, list[str]] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    is_fitted: bool = False

    # --- phase 1: stateless + causal -------------------------------------

    def prepare(self, df: pd.DataFrame, velocity_frame: pd.DataFrame | None = None) -> pd.DataFrame:
        """Row-local and past-only features. No fitting, safe before splitting.

        Args:
            df: Frame (or time-contiguous partition of one) to enrich.
            velocity_frame: Optional precomputed velocity columns indexed by
                ``TransactionID``, from
                :func:`src.features.velocity.compute_velocity_frame`. Supplying
                it is what makes per-partition processing possible: velocity
                needs the whole timeline, but only these ~30 columns do, so they
                are computed once globally and joined per partition. Passing
                ``None`` computes velocity from ``df`` alone, which is only
                correct when ``df`` *is* the whole timeline.
        """
        df = build_stateless_features(df, anchor_d=self.anchor_d_columns)
        if velocity_frame is None:
            return add_velocity_features(
                df,
                entity_columns=ENTITY_KEY_COLUMNS,
                windows_hours=self.velocity_windows_hours,
            )
        missing = set(df[KEY]) - set(velocity_frame.index)
        if missing:
            raise KeyError(
                f"{len(missing)} rows have no precomputed velocity row "
                "(velocity_frame must cover the full timeline)"
            )
        joined = df.join(velocity_frame, on=KEY)
        if len(joined) != len(df):
            raise RuntimeError("velocity join changed row count")
        return joined

    # --- phase 2: fit on training partition only --------------------------

    def fit(self, train_df: pd.DataFrame) -> FeaturePipeline:
        """Learn population statistics from the training partition."""
        if DAY_COL not in train_df.columns:
            raise RuntimeError("fit() requires a frame returned by prepare()")

        train_df = _ensure_uid(train_df)
        self.v_nan_groups = v_nan_groups(train_df)
        train_df = add_v_nan_group_features(train_df.copy(), self.v_nan_groups)
        self.frequency_encoder.min_count = self.frequency_min_count
        self.frequency_encoder.fit(train_df, list(FREQUENCY_COLUMNS))
        self.entity_aggregator.fit(train_df, list(AGGREGATION_KEY_COLUMNS))

        # Determine the final feature list by transforming a small slice, so the
        # column set is derived from real output rather than predicted.
        sample = self._apply_fitted(train_df.head(min(1000, len(train_df))).copy())
        candidate = self._select_feature_columns(sample)

        self.categorical_encoder.fit(
            sample[candidate], [c for c in candidate if _is_categorical_like(sample[c])]
        )
        # Refit the categorical vocabulary on the full training partition — the
        # 1000-row sample is only used to discover the column list.
        full = self._apply_fitted(train_df.copy())
        self.categorical_encoder.fit(
            full[candidate], [c for c in candidate if _is_categorical_like(full[c])]
        )
        self.feature_names = candidate
        self.categorical_features = list(self.categorical_encoder.columns)
        self.is_fitted = True
        logger.info(
            "FeaturePipeline fitted: %d features (%d categorical)",
            len(self.feature_names),
            len(self.categorical_features),
        )
        return self

    # --- phase 3: pure lookup --------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted encoders and return the final feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("FeaturePipeline.transform called before fit")
        out = self._apply_fitted(df)
        missing = [c for c in self.feature_names if c not in out.columns]
        if missing:
            raise RuntimeError(
                f"{len(missing)} expected features absent at transform: {missing[:5]}"
            )
        out = out[self.feature_names]
        return self.categorical_encoder.transform(out)

    def _apply_fitted(self, df: pd.DataFrame) -> pd.DataFrame:
        """Concatenate fitted-encoder output onto ``df`` without mutating it."""
        df = _ensure_uid(df)
        if self.v_nan_groups:
            df = add_v_nan_group_features(df.copy(), self.v_nan_groups)
        return pd.concat(
            [
                df,
                self.frequency_encoder.transform(df),
                self.entity_aggregator.transform(df),
            ],
            axis=1,
        )

    def _select_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Everything except hard exclusions, helper columns and opt-in suffixes."""
        excluded = set(EXCLUDED_FROM_FEATURES)
        selected = [
            c
            for c in df.columns
            if c not in excluded and not c.startswith(INTERNAL_PREFIX) and c != TIME_COL
        ]
        if self.exclude_feature_suffixes:
            before = len(selected)
            selected = [c for c in selected if not c.endswith(tuple(self.exclude_feature_suffixes))]
            logger.info(
                "Excluded %d features by suffix %s (%d -> %d)",
                before - len(selected),
                list(self.exclude_feature_suffixes),
                before,
                len(selected),
            )
        return selected

    # --- persistence ------------------------------------------------------

    def save(self, path: Path) -> Path:
        """Pickle the fitted pipeline for reuse at serving time."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved feature pipeline to %s (%.1f KB)", path, path.stat().st_size / 1024)
        return path

    @staticmethod
    def load(path: Path) -> FeaturePipeline:
        """Load a fitted pipeline."""
        with path.open("rb") as handle:
            pipeline = pickle.load(handle)
        if not isinstance(pipeline, FeaturePipeline):
            raise TypeError(f"{path} does not contain a FeaturePipeline")
        return pipeline
