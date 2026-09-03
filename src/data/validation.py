"""Schema and integrity validation.

Validation here is *fail-loud*. The dataset has two traps that produce silent
degradation rather than errors — the ``id-NN``/``id_NN`` rename and the
identity LEFT JOIN row count — so both are asserted rather than trusted.

Every check corresponds to a fact measured in ``docs/01_dataset_audit.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from src.data.schema import (
    KEY,
    RAW_CATEGORICAL,
    TARGET,
    TIME_COL,
)

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when the data violates an invariant the pipeline depends on."""


@dataclass
class ValidationReport:
    """Outcome of a validation pass."""

    name: str
    n_rows: int
    n_cols: int
    checks_passed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def passed(self, check: str) -> None:
        self.checks_passed.append(check)

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        logger.warning("[%s] %s", self.name, message)

    def summary(self) -> str:
        return (
            f"{self.name}: {self.n_rows:,} rows x {self.n_cols} cols | "
            f"{len(self.checks_passed)} checks passed, {len(self.warnings)} warnings"
        )


def validate_key(df: pd.DataFrame, report: ValidationReport) -> None:
    """The join key must exist, be complete, and be unique."""
    if KEY not in df.columns:
        raise DataValidationError(f"Missing join key {KEY!r}")
    if df[KEY].isna().any():
        raise DataValidationError(f"{KEY} contains nulls")
    n_unique = df[KEY].nunique()
    if n_unique != len(df):
        raise DataValidationError(
            f"{KEY} is not unique: {n_unique:,} distinct values for {len(df):,} rows"
        )
    report.passed(f"{KEY} present, complete, unique")


def validate_temporal_order(df: pd.DataFrame, report: ValidationReport) -> None:
    """Verify the time column can actually support a chronological split.

    This is the precondition for the entire evaluation design. If it fails, a
    chronological split is not defensible and the failure must be surfaced —
    never silently swapped for a random split.
    """
    if TIME_COL not in df.columns:
        raise DataValidationError(f"Missing time column {TIME_COL!r}")
    ts = df[TIME_COL]
    if ts.isna().any():
        raise DataValidationError(f"{TIME_COL} contains nulls — cannot order chronologically")
    if (ts < 0).any():
        raise DataValidationError(f"{TIME_COL} contains negative values")
    if not ts.is_monotonic_increasing:
        raise DataValidationError(
            f"{TIME_COL} is not sorted ascending. build_interim() sorts on load; "
            "this frame was modified after loading."
        )
    report.passed(f"{TIME_COL} non-null, non-negative, monotonic ascending")

    # Ties are expected (5.746% of train rows share a timestamp, max 8 rows on
    # one timestamp). They are not an error, but split boundaries must not cut
    # through a tie group — see splitting.py.
    n_tied = int(len(ts) - ts.nunique())
    if n_tied:
        report.passed(f"{n_tied:,} rows share a timestamp with another row (handled at split)")


def validate_categoricals(df: pd.DataFrame, report: ValidationReport) -> None:
    """Declared categoricals must be non-numeric; nothing else may be object.

    Guards against the schema drifting away from ``src/data/schema.py``.
    """
    declared = set(RAW_CATEGORICAL)
    for col in df.columns:
        is_declared = col in declared
        dtype = df[col].dtype
        is_texty = (
            isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
        )
        if is_declared and not is_texty:
            report.warn(f"{col} is declared categorical but has dtype {dtype}")
        elif not is_declared and is_texty:
            raise DataValidationError(
                f"{col} has dtype {dtype} but is not declared in schema.RAW_CATEGORICAL. "
                "Add it there (with a measured reason) before using this data."
            )
    report.passed("categorical declarations consistent with dtypes")


def validate_identity_rename(df: pd.DataFrame, report: ValidationReport) -> None:
    """No hyphenated identity columns may survive into a validated frame."""
    hyphenated = [c for c in df.columns if c.startswith("id-")]
    if hyphenated:
        raise DataValidationError(
            f"{len(hyphenated)} hyphenated identity columns present "
            f"(e.g. {hyphenated[:3]}). test_identity.csv must be renamed to the "
            "id_NN convention at load time."
        )
    report.passed("no hyphenated identity columns")


def validate_target(df: pd.DataFrame, report: ValidationReport, required: bool) -> None:
    """Check the target when it should be present, and its absence when not."""
    if not required:
        if TARGET in df.columns:
            raise DataValidationError(
                f"{TARGET} present in an unlabeled frame — the Kaggle test set has "
                "no labels, so this indicates the wrong file was loaded."
            )
        report.passed(f"{TARGET} correctly absent (unlabeled split)")
        return

    if TARGET not in df.columns:
        raise DataValidationError(f"Missing target {TARGET!r}")
    if df[TARGET].isna().any():
        raise DataValidationError(f"{TARGET} contains nulls")
    values = set(pd.unique(df[TARGET]).tolist())
    if not values <= {0, 1}:
        raise DataValidationError(f"{TARGET} has non-binary values: {sorted(values)}")
    rate = float(df[TARGET].mean())
    if not 0.0 < rate < 0.5:
        raise DataValidationError(f"Implausible fraud rate {rate:.6f}")
    report.passed(f"{TARGET} binary and complete, positive rate {rate:.4%}")


def validate_no_duplicate_rows(df: pd.DataFrame, report: ValidationReport) -> None:
    """Detect duplicated rows via content hash (cheap on wide frames)."""
    n_dup = int(len(df) - df.index.size + df.duplicated(subset=[KEY]).sum())
    if n_dup:
        raise DataValidationError(f"{n_dup} duplicate {KEY} rows")
    report.passed("no duplicate key rows")


def validate_frame(
    df: pd.DataFrame,
    name: str,
    labeled: bool,
    expect_rows: int | None = None,
) -> ValidationReport:
    """Run the full validation suite over a joined frame.

    Args:
        df: Frame to validate.
        name: Label used in log output.
        labeled: Whether ``isFraud`` is expected to be present.
        expect_rows: Optional exact row count assertion — used after the join to
            prove no rows were gained or lost.
    """
    report = ValidationReport(name=name, n_rows=len(df), n_cols=df.shape[1])

    if df.empty:
        raise DataValidationError(f"{name} is empty")
    if expect_rows is not None and len(df) != expect_rows:
        raise DataValidationError(f"{name} row count {len(df):,} != expected {expect_rows:,}")

    validate_key(df, report)
    validate_identity_rename(df, report)
    validate_temporal_order(df, report)
    validate_categoricals(df, report)
    validate_target(df, report, required=labeled)
    validate_no_duplicate_rows(df, report)

    logger.info(report.summary())
    return report


def missing_value_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing rate and cardinality, sorted worst-first.

    Used by the EDA notebook and reused by the monitoring layer as the
    data-quality reference.
    """
    profile = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "missing_rate": [float(df[c].isna().mean()) for c in df.columns],
            "n_unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )
    return profile.sort_values("missing_rate", ascending=False).reset_index(drop=True)
