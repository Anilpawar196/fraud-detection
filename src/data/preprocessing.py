"""Model-input preparation.

Two distinct paths, because the two model families want opposite things:

* **Tree path** (LightGBM, Random Forest): categoricals become pandas
  ``category`` with a *train-fitted* vocabulary; NaN is preserved. Imputing
  would destroy real signal here — missingness in this dataset is structural
  (``C1``–``C14`` are never null, ``M1``–``M9`` are 29–59% null, and the
  ``V39``–``V52`` block appears and vanishes as a unit), so "absent" is a fact
  about the transaction, not a defect. LightGBM learns a default direction per
  split and uses that fact directly.

* **Linear path** (Logistic Regression): cannot accept NaN, so numeric columns
  are median-imputed **with an explicit missing indicator** for every column
  that has any nulls, then scaled. Without the indicator the linear baseline
  would be handicapped relative to the trees for a reason unrelated to model
  capacity, making the comparison unfair.

Everything is fit on the training partition only. The fitted vocabulary and
statistics are the artifact that gets persisted and reused at serving time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CategoricalCodeEncoder:
    """Maps categorical columns to a fixed, train-fitted vocabulary.

    Values unseen at fit time become NaN rather than a new code, which is the
    honest representation: at serving time an unknown browser string carries no
    learned information, and NaN routes it down the model's missing branch.
    """

    columns: list[str] = field(default_factory=list)
    vocabularies: dict[str, list[Any]] = field(default_factory=dict)
    max_categories: int | None = None

    def fit(self, df: pd.DataFrame, columns: list[str] | None = None) -> CategoricalCodeEncoder:
        self.columns = (
            columns
            if columns is not None
            else [c for c in df.columns if _is_categorical_like(df[c])]
        )
        self.vocabularies = {}
        for col in self.columns:
            counts = df[col].value_counts(dropna=True)
            if self.max_categories is not None and len(counts) > self.max_categories:
                counts = counts.head(self.max_categories)
            self.vocabularies[col] = counts.index.tolist()
        logger.info(
            "Fitted categorical encoder on %d columns (%d total categories)",
            len(self.columns),
            sum(len(v) for v in self.vocabularies.values()),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.columns:
            if col not in out.columns:
                raise KeyError(f"Column {col!r} missing at transform time")
            out[col] = pd.Categorical(out[col], categories=self.vocabularies[col])
        return out

    def fit_transform(self, df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        return self.fit(df, columns).transform(df)

    def n_unseen(self, df: pd.DataFrame) -> dict[str, int]:
        """Count values per column that are not in the fitted vocabulary.

        Reused by the monitoring layer as a data-quality signal: a spike in
        unseen categories is drift the model cannot represent.
        """
        result: dict[str, int] = {}
        for col in self.columns:
            if col in df.columns:
                known = set(self.vocabularies[col])
                values = df[col].dropna()
                result[col] = int((~values.isin(known)).sum())
        return result


@dataclass
class LinearPreprocessor:
    """Median-impute + missing-indicate + standardise, plus one-hot categoricals.

    Produces a float32 dense matrix. Categoricals are one-hot encoded with a
    minimum-frequency floor so that high-cardinality columns (``DeviceInfo``
    exceeds 1,000 distinct values) do not explode the width; rare levels
    collapse into one "infrequent" column, which is also the honest treatment
    given they carry almost no estimable signal in a linear model.
    """

    min_frequency: int = 50
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    indicator_columns: list[str] = field(default_factory=list)
    medians: pd.Series | None = None
    means: np.ndarray | None = None
    scales: np.ndarray | None = None
    onehot_vocab: dict[str, list[Any]] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> LinearPreprocessor:
        self.numeric_columns = [c for c in df.columns if not _is_categorical_like(df[c])]
        self.categorical_columns = [c for c in df.columns if _is_categorical_like(df[c])]
        numeric = df[self.numeric_columns]

        # An all-NaN column yields a NaN median, which would then propagate into
        # the output matrix. Resolving it here means transform() needs no
        # full-matrix NaN sweep (see the note there).
        self.medians = numeric.median(numeric_only=False).fillna(0.0)
        # Only columns that actually have nulls get an indicator — an all-present
        # column's indicator would be a constant zero feature.
        candidates = [c for c in self.numeric_columns if numeric[c].isna().any()]
        # ...and identical missing patterns are collapsed to one column. On this
        # dataset 361 candidates carry only 58 distinct patterns, because the V
        # block appears and vanishes in groups (one pattern is shared by 50
        # columns). Duplicates are exactly redundant for a linear model, and
        # dropping them cuts the dense matrix width by ~24%, which is the
        # difference between fitting in memory and not.
        seen: dict[bytes, str] = {}
        for column in candidates:
            key = numeric[column].isna().to_numpy().tobytes()
            seen.setdefault(key, column)
        self.indicator_columns = list(seen.values())
        if len(candidates) != len(self.indicator_columns):
            logger.info(
                "Missingness indicators: %d columns -> %d distinct patterns",
                len(candidates),
                len(self.indicator_columns),
            )

        self.onehot_vocab = {}
        for col in self.categorical_columns:
            counts = df[col].value_counts(dropna=True)
            self.onehot_vocab[col] = counts[counts >= self.min_frequency].index.tolist()

        filled = numeric.fillna(self.medians)
        self.means = filled.mean().to_numpy(dtype="float32")
        scales = filled.std(ddof=0).to_numpy(dtype="float32")
        # Guard against zero-variance columns producing inf on division.
        scales[scales == 0] = 1.0
        self.scales = scales

        self.feature_names = (
            list(self.numeric_columns)
            + [f"{c}__isna" for c in self.indicator_columns]
            + [
                f"{c}__{value}"
                for c in self.categorical_columns
                for value in [*self.onehot_vocab[c], "__other__"]
            ]
        )
        logger.info(
            "Fitted linear preprocessor: %d numeric + %d indicators + %d one-hot = %d features",
            len(self.numeric_columns),
            len(self.indicator_columns),
            len(self.feature_names) - len(self.numeric_columns) - len(self.indicator_columns),
            len(self.feature_names),
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Build the dense matrix by filling one preallocated array in place.

        The obvious implementation builds each block, then ``np.hstack``es them,
        which holds the blocks *and* the result alive at once — roughly double the
        peak. At 133,979 x 1,270 float32 that second copy is 649 MiB, and it
        failed on this machine even with ~4.9 GB free, because a single
        contiguous block that large is hard to place once the address space has
        been fragmented by repeated per-fold allocations.

        Writing each block straight into its column slice of one preallocated
        array avoids the duplicate entirely, and lets each temporary be released
        as soon as it is copied.
        """
        if self.medians is None or self.means is None or self.scales is None:
            raise RuntimeError("LinearPreprocessor.transform called before fit")

        n_rows = len(df)
        matrix = np.zeros((n_rows, len(self.feature_names)), dtype="float32")
        numeric = df[self.numeric_columns]
        cursor = 0

        # --- scaled numeric block ------------------------------------------
        scaled = (
            (numeric.fillna(self.medians).to_numpy(dtype="float32") - self.means) / self.scales
        ).astype("float32")
        # Guard the only block that could carry NaN or inf. Medians are NaN-free
        # by construction and zero-variance scales are clamped to 1.0 at fit, so
        # this is a safety net rather than a routine correction.
        np.nan_to_num(scaled, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        matrix[:, cursor : cursor + scaled.shape[1]] = scaled
        cursor += scaled.shape[1]
        del scaled

        # --- missingness indicators ----------------------------------------
        for column in self.indicator_columns:
            matrix[:, cursor] = numeric[column].isna().to_numpy(dtype="float32")
            cursor += 1

        # --- one-hot blocks -------------------------------------------------
        for column in self.categorical_columns:
            vocab = self.onehot_vocab[column]
            values = df[column]
            first = cursor
            for value in vocab:
                matrix[:, cursor] = (values == value).to_numpy(dtype="float32")
                cursor += 1
            # Final column absorbs rare + unseen + missing levels.
            matrix[:, cursor] = (matrix[:, first:cursor].sum(axis=1) == 0).astype("float32")
            cursor += 1

        if cursor != len(self.feature_names):
            raise RuntimeError(
                f"Width mismatch: filled {cursor}, expected {len(self.feature_names)}"
            )
        return matrix

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


def _is_categorical_like(series: pd.Series) -> bool:
    """True for category / object / string dtypes."""
    return (
        isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
    )


def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 -> float32 and int64 -> int32 in place where safe."""
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == "float64":
            df[col] = df[col].astype("float32")
        elif dtype == "int64" and df[col].abs().max() < np.iinfo("int32").max:
            df[col] = df[col].astype("int32")
    return df


def memory_mb(df: pd.DataFrame) -> float:
    """Resident size of a frame in MB, including object/category overhead."""
    return float(df.memory_usage(deep=True).sum() / 1024**2)
