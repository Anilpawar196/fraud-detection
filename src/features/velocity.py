"""Past-only velocity features.

These are the features that actually detect card testing and account takeover:
a single transaction viewed in isolation cannot reveal that the same card was
charged eleven times in the last four minutes.

**Why this is not leakage.** Each value uses only rows *strictly earlier in
time* than the row it describes. At inference the same information is available
— those earlier transactions have already happened. So velocity can be computed
over the full chronologically-sorted frame *before* splitting, and a holdout row
legitimately sees earlier holdout rows. What would leak is any statistic drawn
from later rows, and none is used here. Contrast with
:mod:`src.features.aggregations`, whose counts *are* population statistics and
therefore must be fit on the training partition only.

**Implementation.** Naively this is a groupby-apply over ~100k entity groups,
which is slow in Python. Instead each entity's timestamps are offset into a
disjoint numeric band, making the whole frame one globally sorted array; a
single vectorised ``searchsorted`` then resolves every window for every entity
at once. The offset stride exceeds the timestamp span plus the widest window, so
a search can never cross an entity boundary.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.schema import TIME_COL

logger = logging.getLogger(__name__)

#: Offset stride per entity group. Must exceed (max timestamp + widest window).
#: Measured max TransactionDT is 34,214,345 and the widest window is 604,800 s,
#: so 10^9 leaves three orders of magnitude of headroom while staying far inside
#: int64 for the ~10^6 entity groups involved.
_GROUP_STRIDE = 1_000_000_000


def add_velocity_features(
    df: pd.DataFrame,
    entity_columns: tuple[str, ...],
    windows_hours: tuple[int, ...] = (1, 24, 168),
    amount_column: str = "TransactionAmt",
) -> pd.DataFrame:
    """Add trailing-window counts, amount sums and inter-arrival gaps.

    For each entity key and each window, produces:

    * ``{entity}_txn_count_{w}h`` — prior transactions in the window. Bursts are
      the signal; card testing produces many small charges quickly.
    * ``{entity}_amt_sum_{w}h`` — prior amount in the window. Separates "many
      tiny probes" from "one large cash-out".
    * ``{entity}_amt_mean_{w}h`` — the ratio of the two, so the model does not
      have to learn a division.
    * ``{entity}_seconds_since_prev`` — inter-arrival gap. A very short gap is
      automation rather than a human shopping.

    Args:
        df: Frame sorted ascending by ``TransactionDT``.
        entity_columns: Entity key columns to compute velocity per.
        windows_hours: Trailing window widths in hours.
        amount_column: Column to sum within each window.
    """
    if not df[TIME_COL].is_monotonic_increasing:
        raise ValueError(f"{TIME_COL} must be sorted ascending before velocity features")

    timestamps = df[TIME_COL].to_numpy(dtype="int64")
    amounts = df[amount_column].to_numpy(dtype="float64")

    for entity in entity_columns:
        if entity not in df.columns:
            logger.warning("Entity column %s absent — skipping velocity", entity)
            continue
        _add_for_entity(df, entity, timestamps, amounts, windows_hours)

    return df


def _add_for_entity(
    df: pd.DataFrame,
    entity: str,
    timestamps: np.ndarray,
    amounts: np.ndarray,
    windows_hours: tuple[int, ...],
) -> None:
    """Compute all velocity features for one entity key, vectorised."""
    # Integer group codes. NaN entities get their own code and are handled like
    # any other group; they are pre-filled with a sentinel upstream.
    codes = pd.factorize(df[entity], use_na_sentinel=False)[0].astype("int64")

    # Order by (entity, time). Stable sort preserves the existing time order
    # inside each group, which the offset trick relies on.
    order = np.lexsort((timestamps, codes))
    sorted_codes = codes[order]
    sorted_ts = timestamps[order]
    sorted_amt = amounts[order]

    # Project each group into a disjoint numeric band so one global
    # searchsorted resolves every group independently.
    offset_ts = sorted_codes * _GROUP_STRIDE + sorted_ts

    positions = np.arange(sorted_ts.size)
    prefix_amount = np.concatenate([[0.0], np.cumsum(sorted_amt)])

    short_name = entity.lstrip("_")
    for hours in windows_hours:
        window = hours * 3600
        left = np.searchsorted(offset_ts, offset_ts - window, side="left")
        count = (positions - left).astype("float32")
        amount_sum = (prefix_amount[positions] - prefix_amount[left]).astype("float32")

        with np.errstate(invalid="ignore", divide="ignore"):
            amount_mean = np.where(count > 0, amount_sum / count, np.nan).astype("float32")

        df[f"{short_name}_txn_count_{hours}h"] = _restore(count, order)
        df[f"{short_name}_amt_sum_{hours}h"] = _restore(amount_sum, order)
        df[f"{short_name}_amt_mean_{hours}h"] = _restore(amount_mean, order)

    # Inter-arrival gap: difference to the previous row of the same entity.
    gap = np.full(sorted_ts.size, np.nan, dtype="float32")
    same_group = sorted_codes[1:] == sorted_codes[:-1]
    diffs = (sorted_ts[1:] - sorted_ts[:-1]).astype("float32")
    gap[1:] = np.where(same_group, diffs, np.nan)
    df[f"{short_name}_seconds_since_prev"] = _restore(gap, order)


def compute_velocity_frame(
    narrow: pd.DataFrame,
    entity_columns: tuple[str, ...],
    windows_hours: tuple[int, ...] = (1, 24, 168),
    key_column: str = "TransactionID",
    amount_column: str = "TransactionAmt",
) -> pd.DataFrame:
    """Compute velocity features from a narrow frame, keyed for later joining.

    Velocity is the one feature family that genuinely needs the *whole* timeline
    at once: a validation row's trailing 7-day count legitimately includes
    training rows. Materialising every engineered column for all 590,540 rows
    simultaneously to get it would cost well over a gigabyte of RAM.

    Instead this takes only the handful of columns velocity actually depends on
    (time, amount, entity keys), computes the ~30 velocity columns for the full
    timeline, and returns them indexed by ``TransactionID``. Each partition then
    joins the rows it needs. Identical values, a fraction of the peak memory.

    Args:
        narrow: Frame with the key, time, amount and entity key columns, sorted
            ascending by time.
        entity_columns: Entity key columns to compute velocity per.
        windows_hours: Trailing window widths in hours.
        key_column: Column to index the result by.
        amount_column: Column to sum within each window.

    Returns:
        Velocity columns only, indexed by ``key_column``.
    """
    before = set(narrow.columns)
    enriched = add_velocity_features(
        narrow.copy(),
        entity_columns=entity_columns,
        windows_hours=windows_hours,
        amount_column=amount_column,
    )
    new_columns = [c for c in enriched.columns if c not in before]
    result = enriched[[key_column, *new_columns]].set_index(key_column)
    logger.info(
        "Computed %d velocity columns for %d rows (%.1f MB)",
        len(new_columns),
        len(result),
        result.memory_usage(deep=True).sum() / 1024**2,
    )
    return result


def _restore(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Invert the (entity, time) permutation back to original row order."""
    out = np.empty_like(values)
    out[order] = values
    return out
