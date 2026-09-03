"""Raw data loading: CSV -> typed Parquet -> joined interim frame.

Design constraints that shaped this module:

* ``train_transaction.csv`` is 651 MB and would occupy ~1.9 GB as pandas
  float64. Explicit dtypes (float32 / int32 / category) cut that to ~0.9 GB.
* The conversion is **streamed** through ``pyarrow.parquet.ParquetWriter``, so
  the full frame is never resident during conversion. That keeps the pipeline
  runnable on a machine with a couple of GB free.
* ``test_identity.csv`` uses ``id-NN`` where ``train_identity.csv`` uses
  ``id_NN``. Renaming happens here, once, at the boundary.

Parquet is the interchange format for everything downstream: it preserves
dtypes (so categoricals do not silently become objects), supports column
projection, and reloads roughly an order of magnitude faster than the CSV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.schema import (
    KEY,
    RAW_CATEGORICAL,
    TARGET,
    TIME_COL,
    identity_rename_map,
)
from src.utils.paths import INTERIM_DIR, RAW_DIR, ensure_dir

logger = logging.getLogger(__name__)

Split = Literal["train", "test"]
Kind = Literal["transaction", "identity"]

#: Chunk size for the streaming CSV -> Parquet conversion.
DEFAULT_CHUNKSIZE = 100_000


def raw_csv_path(split: Split, kind: Kind, raw_dir: Path | None = None) -> Path:
    """Path to a raw Kaggle CSV."""
    return (raw_dir or RAW_DIR) / f"{split}_{kind}.csv"


def parquet_path(split: Split, kind: Kind, interim_dir: Path | None = None) -> Path:
    """Path to the typed Parquet mirror of a raw CSV."""
    return (interim_dir or INTERIM_DIR) / f"{split}_{kind}.parquet"


def joined_path(split: Split, interim_dir: Path | None = None) -> Path:
    """Path to the transaction+identity joined frame."""
    return (interim_dir or INTERIM_DIR) / f"{split}_joined.parquet"


def build_dtypes(columns: list[str]) -> dict[str, str]:
    """Choose a compact dtype per column, without assuming column names.

    Rules, in order:
      * declared categorical (measured non-numeric)  -> ``category``
      * the join key and the time column             -> ``int32``
      * the target                                   -> ``int8``
      * everything else                              -> ``float32``

    float32 is safe for this data: the widest numeric range measured is
    ``TransactionAmt`` at 31,937.39 and ``id_02`` in the hundreds of thousands,
    both far inside float32's ~7 significant digits.
    """
    categorical = set(RAW_CATEGORICAL)
    dtypes: dict[str, str] = {}
    for col in columns:
        if col in categorical:
            dtypes[col] = "category"
        elif col in (KEY, TIME_COL):
            dtypes[col] = "int32"
        elif col == TARGET:
            dtypes[col] = "int8"
        else:
            dtypes[col] = "float32"
    return dtypes


def read_csv_header(path: Path) -> list[str]:
    """Read only the header row of a CSV."""
    return list(pd.read_csv(path, nrows=0).columns)


def csv_to_parquet(
    csv_path: Path,
    out_path: Path,
    chunksize: int = DEFAULT_CHUNKSIZE,
    compression: str = "snappy",
) -> Path:
    """Stream a raw CSV into a typed Parquet file.

    Categorical columns are written as plain strings rather than dictionary
    type: category *codes* are per-chunk and would be inconsistent across row
    groups. Category alignment happens later, once, against a train-fitted
    vocabulary.
    """
    columns = read_csv_header(csv_path)
    columns = [identity_rename_map(columns).get(c, c) for c in columns]
    dtypes = build_dtypes(columns)
    # Read categoricals as string here; convert to category after alignment.
    read_dtypes = {c: ("string" if d == "category" else d) for c, d in dtypes.items()}

    ensure_dir(out_path.parent)
    writer: pq.ParquetWriter | None = None
    n_rows = 0
    try:
        reader = pd.read_csv(
            csv_path,
            chunksize=chunksize,
            dtype=read_dtypes,
            low_memory=False,
        )
        for chunk in reader:
            chunk = chunk.rename(columns=identity_rename_map(list(chunk.columns)))
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression=compression)
            writer.write_table(table)
            n_rows += len(chunk)
    finally:
        if writer is not None:
            writer.close()

    size_mb = out_path.stat().st_size / 1024**2
    logger.info(
        "%s -> %s (%d rows, %d cols, %.1f MB parquet)",
        csv_path.name,
        out_path.name,
        n_rows,
        len(columns),
        size_mb,
    )
    return out_path


def load_parquet(
    path: Path, columns: list[str] | None = None, categorical: bool = True
) -> pd.DataFrame:
    """Read a Parquet file, restoring ``category`` dtype on declared columns.

    Args:
        path: Parquet file to read.
        columns: Optional column projection — the main lever for keeping
            memory down when only a few columns are needed.
        categorical: Whether to restore ``category`` dtype.
    """
    df = pd.read_parquet(path, columns=columns)
    if categorical:
        for col in df.columns:
            if col in RAW_CATEGORICAL and not isinstance(df[col].dtype, pd.CategoricalDtype):
                df[col] = df[col].astype("category")
    return df


def join_transaction_identity(transaction: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    """LEFT JOIN identity onto transaction on ``TransactionID``.

    LEFT, not INNER: identity covers only 24.42% of train rows and 28.01% of
    test rows (measured). An inner join would discard 75.6% of training data,
    and because coverage itself differs between train and test it would also
    bias the sample. Identity absence is captured as a feature instead.

    Raises:
        ValueError: if the join changed the row count, which would mean the key
            is not unique on the identity side.
    """
    n_before = len(transaction)
    merged = transaction.merge(identity, on=KEY, how="left", validate="one_to_one")
    if len(merged) != n_before:
        raise ValueError(
            f"LEFT JOIN changed row count: {n_before} -> {len(merged)}. "
            f"{KEY} is not unique in the identity frame."
        )
    identity_cols = [c for c in identity.columns if c != KEY]
    merged["identity_present"] = merged[identity_cols].notna().any(axis=1).astype("int8")
    coverage = merged["identity_present"].mean()
    logger.info(
        "Joined %d transaction rows; identity coverage %.4f%%",
        len(merged),
        coverage * 100,
    )
    return merged


def build_interim(
    split: Split,
    raw_dir: Path | None = None,
    interim_dir: Path | None = None,
    chunksize: int = DEFAULT_CHUNKSIZE,
    force: bool = False,
) -> Path:
    """Convert both raw CSVs for ``split`` to Parquet and write the joined frame.

    Returns:
        Path to the joined Parquet file.
    """
    out = joined_path(split, interim_dir)
    if out.is_file() and not force:
        logger.info("%s already exists — skipping (pass force=True to rebuild)", out.name)
        return out

    for kind in ("transaction", "identity"):
        pq_path = parquet_path(split, kind, interim_dir)  # type: ignore[arg-type]
        if force or not pq_path.is_file():
            csv_to_parquet(
                raw_csv_path(split, kind, raw_dir),  # type: ignore[arg-type]
                pq_path,
                chunksize=chunksize,
            )

    transaction = load_parquet(parquet_path(split, "transaction", interim_dir))
    identity = load_parquet(parquet_path(split, "identity", interim_dir))
    merged = join_transaction_identity(transaction, identity)
    del transaction, identity

    # Sort by time so every downstream consumer sees chronological order.
    # Verified: the raw file order is already monotonic in TransactionDT, so
    # this is a cheap no-op guard rather than a reshuffle — but relying on file
    # order would be an undocumented assumption.
    merged = merged.sort_values([TIME_COL, KEY], kind="mergesort").reset_index(drop=True)

    ensure_dir(out.parent)
    merged.to_parquet(out, index=False, compression="snappy")
    logger.info(
        "Wrote %s (%d rows x %d cols, %.1f MB)",
        out.name,
        len(merged),
        merged.shape[1],
        out.stat().st_size / 1024**2,
    )
    return out
