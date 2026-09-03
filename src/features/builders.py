"""Stateless feature builders.

Every function here is *stateless and causal*: its output for a row depends only
on that row (or on strictly earlier rows). Nothing learns a statistic from the
data, so these can be applied to the whole frame before splitting without
leaking — there is no fitted quantity that could carry information from
validation or holdout back into training.

Stateful transformations (frequency maps, entity aggregates) live in
:mod:`src.features.aggregations` and are fit on the training partition only.

Each feature carries a short note on *why* it should help. Features without a
mechanism are not added.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from src.data.schema import (
    C_COLUMNS,
    D_COLUMNS,
    ID_COLUMNS,
    M_COLUMNS,
    SECONDS_PER_DAY,
    TIME_COL,
    V_COLUMNS,
)

logger = logging.getLogger(__name__)

#: Internal helper column: absolute day index. Used to build features and folds,
#: then dropped — absolute time cannot be a model input because train and test
#: day ranges are disjoint (days 1-183 vs 213-396), so a tree would route every
#: test row into the rightmost leaf.
DAY_COL = "_day_index"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical time features derived from ``TransactionDT``.

    Only *cyclical* and *relative* time is usable. Hour-of-day and day-of-week
    repeat identically in the test period, so they transfer across the 30-day
    gap; an absolute day index does not.

    Mechanism: fraud rates vary by time of day (automated card testing runs at
    off-peak hours when review staffing is thin).
    """
    seconds = df[TIME_COL].to_numpy(dtype="int64")
    df[DAY_COL] = (seconds // SECONDS_PER_DAY).astype("int32")
    df["hour_of_day"] = ((seconds // 3600) % 24).astype("int8")
    df["day_of_week"] = ((seconds // SECONDS_PER_DAY) % 7).astype("int8")
    # Business-hours flag: coarse but interpretable, and trees can ignore it if
    # hour_of_day already carries the signal.
    df["is_night"] = ((df["hour_of_day"] < 6) | (df["hour_of_day"] >= 22)).astype("int8")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    return df


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transaction-amount transformations.

    * ``log_amount`` — the raw amount is strongly right-skewed (mean 135.03,
      std 239.16, max 31,937.39). The linear baseline needs this to be usable at
      all; trees are indifferent but unharmed.
    * ``amount_cents`` / ``is_round_amount`` — the fractional part. Card-testing
      transactions cluster on round values, and some fraud patterns reuse an
      exact amount repeatedly. The decimal part is also a fingerprint of
      currency conversion, which distinguishes cross-border traffic.
    * ``amount_log_decimal_digits`` — number of decimal places actually used.
    """
    amount = df["TransactionAmt"].to_numpy(dtype="float64")
    df["log_amount"] = np.log1p(amount).astype("float32")
    cents = np.round(amount - np.floor(amount), 4)
    df["amount_cents"] = cents.astype("float32")
    df["is_round_amount"] = (cents == 0).astype("int8")
    return df


def add_email_features(df: pd.DataFrame) -> pd.DataFrame:
    """Purchaser/recipient email-domain features.

    * provider and TLD split — ``gmail.com`` and ``gmail.co.uk`` share a
      provider but differ in geography; splitting lets the model use each.
    * ``email_domains_match`` — a purchaser/recipient domain mismatch is a
      classic mule/reshipping indicator.
    * missing flags — ``R_emaildomain`` is 76.75% null, so its absence is a
      far more common state than any single value it takes.
    """
    for col, prefix in (("P_emaildomain", "p_email"), ("R_emaildomain", "r_email")):
        if col not in df.columns:
            continue
        values = df[col].astype("string")
        parts = values.str.split(".", n=1, expand=True)
        provider = parts[0] if parts.shape[1] > 0 else values
        suffix = parts[1] if parts.shape[1] > 1 else pd.Series(pd.NA, index=df.index)
        df[f"{prefix}_provider"] = provider.astype("category")
        df[f"{prefix}_suffix"] = suffix.astype("category")
        df[f"{prefix}_is_missing"] = values.isna().astype("int8")

    if {"P_emaildomain", "R_emaildomain"} <= set(df.columns):
        p = df["P_emaildomain"].astype("string")
        r = df["R_emaildomain"].astype("string")
        both_present = (p.notna() & r.notna()).to_numpy()
        # Comparing two nullable string columns yields NA wherever either side is
        # missing, which cannot cast to an integer — so resolve NA to False
        # first and encode "not comparable" as -1 via the mask below.
        equal = (p == r).fillna(False).to_numpy()
        df["email_domains_match"] = np.where(both_present, equal.astype("int8"), -1).astype("int8")
    return df


def add_device_features(df: pd.DataFrame) -> pd.DataFrame:
    """Device, OS, browser and screen features parsed out of free-text columns.

    ``DeviceInfo`` has >1,000 distinct values in raw form (``SAMSUNG SM-G892A
    Build/NRD90M``). Used verbatim it would overfit to individual handsets; the
    vendor token generalises. Same argument for splitting ``id_30``/``id_31``
    into family + version: version-as-a-number lets the model express "outdated
    client", which an opaque string cannot.
    """
    if "DeviceInfo" in df.columns:
        info = df["DeviceInfo"].astype("string")
        df["device_vendor"] = (
            info.str.split(r"[ /]", n=1, regex=True).str[0].str.lower().astype("category")
        )
        df["device_info_is_missing"] = info.isna().astype("int8")

    if "id_30" in df.columns:
        os_values = df["id_30"].astype("string")
        df["os_family"] = (
            os_values.str.extract(r"^([A-Za-z]+)", expand=False).str.lower().astype("category")
        )
        df["os_version_major"] = pd.to_numeric(
            os_values.str.extract(r"(\d+)", expand=False), errors="coerce"
        ).astype("float32")

    if "id_31" in df.columns:
        browser = df["id_31"].astype("string")
        df["browser_family"] = (
            browser.str.extract(r"^([a-zA-Z ]+)", expand=False)
            .str.strip()
            .str.lower()
            .astype("category")
        )
        df["browser_version_major"] = pd.to_numeric(
            browser.str.extract(r"(\d+)", expand=False), errors="coerce"
        ).astype("float32")

    if "id_33" in df.columns:
        res = df["id_33"].astype("string").str.extract(r"^(\d+)x(\d+)$")
        width = pd.to_numeric(res[0], errors="coerce")
        height = pd.to_numeric(res[1], errors="coerce")
        df["screen_width"] = width.astype("float32")
        df["screen_height"] = height.astype("float32")
        df["screen_pixels"] = (width * height).astype("float32")
        # Aspect ratio separates phones from desktops independently of size.
        df["screen_aspect"] = (width / height.replace(0, np.nan)).astype("float32")
    return df


def add_missingness_features(df: pd.DataFrame) -> pd.DataFrame:
    """Explicit missingness counts.

    Missingness in this dataset is structural and non-random: identity is
    resolved for only 24.42% of transactions, ``M1``-``M9`` are 29-59% null, and
    ``V`` blocks appear/vanish as units. Counting them per row gives the model a
    compact summary of "how much do we actually know about this transaction",
    which is itself a strong fraud signal — thin records are riskier.
    """
    families = {
        "n_missing_v": [c for c in V_COLUMNS if c in df.columns],
        "n_missing_d": [c for c in D_COLUMNS if c in df.columns],
        "n_missing_m": [c for c in M_COLUMNS if c in df.columns],
        "n_missing_c": [c for c in C_COLUMNS if c in df.columns],
        "n_missing_id": [c for c in ID_COLUMNS if c in df.columns],
    }
    total = np.zeros(len(df), dtype="int32")
    for name, columns in families.items():
        if not columns:
            continue
        counts = df[columns].isna().sum(axis=1).to_numpy(dtype="int32")
        df[name] = counts
        total += counts
    df["n_missing_total"] = total
    return df


def add_anchored_d_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time-anchor the ``D`` columns.

    ``D1``-``D15`` behave like day-deltas relative to a moving reference (``D1``
    spans [0, 640]; ``D4``, ``D6``, ``D11``, ``D12``, ``D14``, ``D15`` go
    negative). Raw, their values shift with absolute time, so their train
    distribution does not match the test period 30 days later.

    ``day_index - D_n`` converts a moving delta into a fixed calendar anchor
    (e.g. "the day this card was first seen"), which is stable across the gap.
    Both forms are kept and the model decides; §7.8 of the audit flagged this as
    a hypothesis, and the validation comparison in
    ``reports/feature_ablation.csv`` is what settles it.
    """
    if DAY_COL not in df.columns:
        raise KeyError(f"{DAY_COL} required — call add_time_features first")
    day = df[DAY_COL].to_numpy(dtype="float32")
    for col in D_COLUMNS:
        if col in df.columns:
            df[f"{col}_anchored"] = (day - df[col].to_numpy(dtype="float32")).astype("float32")
    return df


UID_COLUMN = "_entity_uid"

#: Columns the synthetic account id is built from. All are row-local.
UID_COMPONENTS = (
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
)


def v_nan_groups(df: pd.DataFrame) -> dict[int, list[str]]:
    """Group V columns by identical null count.

    The 339 V columns are anonymised and highly redundant, but their *missing*
    patterns are structured: columns sharing a null count belong to the same
    underlying block. Returned keyed by null count and sorted deterministically
    by the caller.

    This is a **population statistic** — the groups depend on which rows the
    frame contains, so a 30k slice yields 13 groups where the full training
    partition yields 14. It must therefore be fitted on the training partition
    and reused verbatim at transform time; deriving it per frame would make the
    feature set differ between fit and transform, and would also let the
    validation rows influence their own encoding.
    """
    v_columns = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
    groups: dict[int, list[str]] = {}
    if not v_columns:
        return groups
    nulls = df[v_columns].isna().sum()
    for col in v_columns:
        count = int(nulls[col])
        if count > 0:
            groups.setdefault(count, []).append(col)
    return groups


def add_v_nan_group_features(df: pd.DataFrame, groups: dict[int, list[str]]) -> pd.DataFrame:
    """Summarise each fitted V missingness block to a per-row mean and std.

    Reducing a block of correlated columns to its level and spread gives the
    model a compact view without feeding it 339 columns. Purely row-wise given
    the group definition: each output depends only on values in the same row.

    Blocks are summarised one at a time so peak memory is a single block (the
    largest is 46 columns) rather than the full V matrix.
    """
    if not groups:
        return df
    for index, count in enumerate(sorted(groups)):
        block = [c for c in groups[count] if c in df.columns]
        if not block:
            continue
        values = df[block].to_numpy(dtype="float32")
        # A row can be entirely null within a block; nanmean/nanstd correctly
        # return NaN there and warn about it. NaN is the answer we want -- the
        # boosters route missing values natively -- so the warning is noise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            df[f"v_nan_group_{index}_mean"] = np.nanmean(values, axis=1).astype("float32")
            df[f"v_nan_group_{index}_std"] = np.nanstd(values, axis=1).astype("float32")
    return df


def add_uid_entity_key(df: pd.DataFrame) -> pd.DataFrame:
    """Build a synthetic account id from the card fields plus ``D1 - day_index``.

    ``D1`` is a day-delta against a moving reference, so ``D1 - day_index`` is
    *constant for a given card* — it encodes when the account was first seen.
    Combined with the card attributes it approximates an account identifier far
    more precisely than ``card1 + addr1 + card2`` does on its own.

    This is the same quantity as :func:`add_anchored_d_features` produces, used
    differently and deliberately so. As a *numeric feature* the anchor is
    excluded from the model, because its value drifts completely against the
    deployment period (KS 1.000 — see the drift report). As a *grouping key* that
    drift is irrelevant: two transactions from one card fall in the same group
    whatever period they occur in, because the key is compared for equality, not
    magnitude. Identity is stable even when the number that encodes it is not.

    Row-local, so this belongs in the prepare phase: nothing is learned from
    other rows and no future information is touched. The aggregates computed
    *over* these groups are fitted on the training partition only, as with every
    other encoder here.
    """
    if DAY_COL not in df.columns:
        raise KeyError(f"{DAY_COL} required — call add_time_features first")
    missing = [c for c in UID_COMPONENTS if c not in df.columns]
    if missing:
        raise KeyError(f"add_uid_entity_key needs {missing}")

    # Account-open anchor. NaN D1 yields NaN, which str() renders consistently,
    # so those rows group together rather than each becoming a unique id.
    anchor = df["D1"].to_numpy(dtype="float64") - df[DAY_COL].to_numpy(dtype="float64")

    parts = [df[c].astype("string").fillna("na") for c in UID_COMPONENTS]
    key = parts[0].str.cat(parts[1:], sep="_")
    key = key.str.cat(pd.Series(anchor, index=df.index).astype("string"), sep="_")

    # Deterministic across processes and partitions, unlike factorize codes.
    df[UID_COLUMN] = pd.util.hash_pandas_object(key, index=False).astype("uint64")
    return df


def add_entity_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Composite entity keys approximating a card/account.

    No true account id exists in this dataset. ``card1`` (0% missing) combined
    with ``addr1`` (11.13% missing) and ``card2`` approximates one closely enough
    for per-entity history. These are *keys*, not model inputs — the velocity and
    aggregation layers consume them, then the pipeline drops them.

    The keys are built as **deterministic integer arithmetic** rather than
    concatenated strings or factorized codes, for two reasons:

    * Memory: three string columns over 590,540 rows cost hundreds of MB;
      int64 costs 4.7 MB each.
    * Correctness: partitions are prepared separately, and ``factorize`` codes
      are assigned per-call, so the same card would get different codes in the
      training and holdout files. Arithmetic on the raw values is stable
      everywhere, including at serving time for a single transaction.

    Collision-freedom depends only on the two *low-order* components staying
    inside their positional slots: ``addr1`` and ``card2`` must each be < 1000.
    ``card1`` occupies the high-order slot, so its magnitude is irrelevant to
    uniqueness and is bounded only by int64 overflow.

    That distinction was learned the hard way. An earlier version also asserted
    ``card1 <= 18396`` — its maximum in *train* — and the build failed on the test
    split, where ``card1`` reaches 18,397. The assertion was over-strict rather
    than the data being wrong: one extra card id cannot cause a collision. The
    check now constrains exactly what correctness requires, which is also why the
    fix needed no retraining — every previously computed key is unchanged.

    Measured ranges: ``card1`` ∈ [1000, 18397], ``addr1`` ∈ [100, 540],
    ``card2`` ∈ [100, 600] across both train and test. Missing values map to 0,
    which cannot collide because each column's observed minimum is ≥ 100.
    """
    # Low-order slots: exceeding these would overflow into the next component
    # and silently merge two different entities.
    for column, exclusive_maximum in (("addr1", 1_000), ("card2", 1_000)):
        observed = df[column].max()
        if pd.notna(observed) and observed >= exclusive_maximum:
            raise ValueError(
                f"{column} max {observed} reaches the {exclusive_maximum} positional slot "
                "assumed by the entity-key encoding; widen the multipliers in "
                "add_entity_keys (and rebuild all derived features) before proceeding."
            )

    # High-order slot: only int64 overflow is a real constraint here.
    card1_max = df["card1"].max()
    if pd.notna(card1_max) and card1_max > 9_000_000_000:
        raise ValueError(f"card1 max {card1_max} risks int64 overflow in the entity-key encoding.")

    card1 = df["card1"].fillna(0).astype("int64")
    addr1 = df["addr1"].fillna(0).astype("int64")
    card2 = df["card2"].fillna(0).astype("int64")

    df["_entity_card"] = card1
    df["_entity_card_addr"] = card1 * 1_000 + addr1
    df["_entity_card_full"] = card1 * 1_000_000 + addr1 * 1_000 + card2
    return df


ENTITY_KEY_COLUMNS = ("_entity_card", "_entity_card_addr", "_entity_card_full")

#: Keys the fitted aggregators group over. Wider than ENTITY_KEY_COLUMNS, which
#: also drives velocity: velocity is computed once over the whole timeline
#: during dataset build, so adding a key there means rebuilding the dataset,
#: whereas aggregation keys are fitted per fold and cost nothing to extend.
AGGREGATION_KEY_COLUMNS = (*ENTITY_KEY_COLUMNS, UID_COLUMN)


def build_stateless_features(df: pd.DataFrame, anchor_d: bool = True) -> pd.DataFrame:
    """Apply every stateless builder in dependency order."""
    n_before = df.shape[1]
    df = add_time_features(df)
    df = add_amount_features(df)
    df = add_email_features(df)
    df = add_device_features(df)
    df = add_missingness_features(df)
    if anchor_d:
        df = add_anchored_d_features(df)
    df = add_entity_keys(df)
    df = add_uid_entity_key(df)
    logger.info("Stateless features: %d -> %d columns", n_before, df.shape[1])
    return df
