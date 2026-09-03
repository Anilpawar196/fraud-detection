"""Canonical schema facts for the IEEE-CIS dataset.

Every list here was *derived from measurement* by ``scripts/inspect_dataset.py``
(see ``docs/01_dataset_audit.md``), not from assumption. The lists are declared
explicitly so they are reviewable, and :mod:`src.data.validation` asserts them
against the real files at load time — if the data ever disagrees with this
module, loading fails loudly instead of degrading silently.
"""

from __future__ import annotations

from typing import Final

TARGET: Final[str] = "isFraud"
KEY: Final[str] = "TransactionID"
TIME_COL: Final[str] = "TransactionDT"

#: Seconds in the TransactionDT unit. Verified: 182 contiguous days, no gaps
#: larger than 1.15 h, no nulls, no negatives.
SECONDS_PER_DAY: Final[int] = 86_400

# --- Categorical columns (measured: non-numeric dtype in the raw CSVs) -------

TRANSACTION_CATEGORICAL: Final[tuple[str, ...]] = (
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
)

IDENTITY_CATEGORICAL: Final[tuple[str, ...]] = (
    "id_12",
    "id_15",
    "id_16",
    "id_23",
    "id_27",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_33",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
    "DeviceType",
    "DeviceInfo",
)

RAW_CATEGORICAL: Final[tuple[str, ...]] = TRANSACTION_CATEGORICAL + IDENTITY_CATEGORICAL

# --- Column families --------------------------------------------------------

C_COLUMNS: Final[tuple[str, ...]] = tuple(f"C{i}" for i in range(1, 15))
D_COLUMNS: Final[tuple[str, ...]] = tuple(f"D{i}" for i in range(1, 16))
M_COLUMNS: Final[tuple[str, ...]] = tuple(f"M{i}" for i in range(1, 10))
V_COLUMNS: Final[tuple[str, ...]] = tuple(f"V{i}" for i in range(1, 340))
CARD_COLUMNS: Final[tuple[str, ...]] = tuple(f"card{i}" for i in range(1, 7))
ADDR_COLUMNS: Final[tuple[str, ...]] = ("addr1", "addr2")
DIST_COLUMNS: Final[tuple[str, ...]] = ("dist1", "dist2")
ID_COLUMNS: Final[tuple[str, ...]] = tuple(f"id_{i:02d}" for i in range(1, 39))
EMAIL_COLUMNS: Final[tuple[str, ...]] = ("P_emaildomain", "R_emaildomain")

#: Columns used to approximate a card/account entity for past-only aggregates.
#: card1 has 0% missing; addr1 has 11.13%. Combined they form a usable, if
#: imperfect, client proxy. Validated for cardinality in the feature layer.
ENTITY_KEYS: Final[tuple[str, ...]] = ("card1", "addr1", "card2", "P_emaildomain")

# --- Exclusions (see docs/01_dataset_audit.md §8) ---------------------------

#: Never fed to a model. Each has a measured reason:
#:   isFraud       — the target
#:   TransactionID — monotonic (corr 0.998 with TransactionDT) ⇒ absolute-time proxy
#:   TransactionDT — absolute time; train and test ranges are disjoint
#:   V107          — constant in test_transaction.csv (zero variance at scoring)
EXCLUDED_FROM_FEATURES: Final[tuple[str, ...]] = (
    TARGET,
    KEY,
    TIME_COL,
    "V107",
)


def identity_rename_map(columns: list[str]) -> dict[str, str]:
    """Map hyphenated identity columns to the underscored train convention.

    ``test_identity.csv`` ships ``id-01``…``id-38`` while
    ``train_identity.csv`` ships ``id_01``…``id_38``. All 38 differ. Left
    unfixed, a model trained on train and scored on test sees 38 all-null
    columns and degrades silently rather than raising.
    """
    return {c: c.replace("-", "_") for c in columns if c.startswith("id-")}


def feature_columns(all_columns: list[str]) -> list[str]:
    """All model-input columns: everything except the hard exclusions."""
    excluded = set(EXCLUDED_FROM_FEATURES)
    return [c for c in all_columns if c not in excluded]
