"""Build the modelling dataset: raw CSV -> validated join -> prepared features.

Pipeline stages, and why they are ordered this way:

1. **Join + validate.** LEFT JOIN identity onto transaction, then assert the
   invariants that would otherwise fail silently (row count, key uniqueness,
   chronological order, no surviving ``id-NN`` columns).
2. **Velocity once, globally.** Trailing-window features need the whole
   timeline, so they are computed from a narrow projection (key, time, amount,
   entity keys) and cached keyed by ``TransactionID``.
3. **Split by time.** Holdout is the final 20% of rows by timestamp; the rest is
   the modelling period. Purged forward-chaining folds are generated over the
   modelling period.
4. **Prepare each partition separately.** Stateless + causal features only.
   Stateful encoders are deliberately *not* fitted here — they are fitted inside
   each CV fold by the training code, which is what makes the cross-validation
   genuinely leakage-free rather than approximately so.

Outputs (``data/processed/``):
    modelling_prepared.parquet   train+validation period, labelled
    holdout_prepared.parquet     final 20% by time, labelled, scored once
    split_metadata.json          partition sizes, cut timestamps, fraud rates
    folds_temporal.npz           purged forward-chaining fold indices
    folds_random.npz             random stratified folds (control experiment only)

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --force --with-test
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loading import build_interim, joined_path, load_parquet  # noqa: E402
from src.data.schema import KEY, TARGET, TIME_COL  # noqa: E402
from src.data.splitting import (  # noqa: E402
    PurgedForwardChainingCV,
    random_stratified_folds,
    save_folds,
    save_split,
    temporal_split,
)
from src.data.validation import validate_frame  # noqa: E402
from src.features.builders import ENTITY_KEY_COLUMNS, add_entity_keys  # noqa: E402
from src.features.pipeline import FeaturePipeline  # noqa: E402
from src.features.velocity import compute_velocity_frame  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402
from src.utils.paths import INTERIM_DIR, PROCESSED_DIR, ensure_dir  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = logging.getLogger("build_dataset")

VELOCITY_SOURCE_COLUMNS = (KEY, TIME_COL, "TransactionAmt", "card1", "addr1", "card2")

#: Rows per streaming batch when preparing a partition. Bounds peak memory at
#: roughly chunk x ~514 columns x 4 bytes (~205 MB at 100k).
DEFAULT_CHUNK_ROWS = 100_000


def build_velocity_cache(split: str, windows_hours: tuple[int, ...], force: bool) -> pd.DataFrame:
    """Compute (or load) the global velocity frame for a split."""
    cache = INTERIM_DIR / f"{split}_velocity.parquet"
    if cache.is_file() and not force:
        logger.info("Loading cached velocity frame %s", cache.name)
        return pd.read_parquet(cache)

    narrow = load_parquet(
        joined_path(split), columns=list(VELOCITY_SOURCE_COLUMNS), categorical=False
    )
    narrow = narrow.sort_values([TIME_COL, KEY], kind="mergesort").reset_index(drop=True)
    narrow = add_entity_keys(narrow)
    velocity = compute_velocity_frame(
        narrow, entity_columns=ENTITY_KEY_COLUMNS, windows_hours=windows_hours
    )
    velocity.to_parquet(cache, compression="snappy")
    logger.info("Cached velocity frame -> %s (%.1f MB)", cache.name, cache.stat().st_size / 1024**2)
    return velocity


def prepare_partition(
    split: str,
    dt_low: int | None,
    dt_high: int | None,
    velocity: pd.DataFrame,
    pipeline: FeaturePipeline,
    out_path: Path,
    force: bool = False,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> tuple[int, int]:
    """Read one time-contiguous partition, apply causal features, write Parquet.

    Reading by timestamp predicate rather than row index keeps only one
    partition resident at a time — the difference between a ~1 GB peak and a
    ~2.5 GB one on this dataset.
    """
    if out_path.is_file() and not force:
        # Rebuilding an existing partition is pure cost, and on a memory-tight
        # machine the transient copy is enough to fail the whole run. Adding
        # --with-test should not re-do the train partitions.
        existing = pq.read_metadata(out_path)
        logger.info(
            "%s already exists (%d rows x %d cols) — skipping; pass --force to rebuild",
            out_path.name,
            existing.num_rows,
            existing.num_columns,
        )
        return existing.num_rows, existing.num_columns

    started = time.perf_counter()
    ensure_dir(out_path.parent)

    # Streamed in row batches rather than prepared as one frame. With `velocity`
    # supplied, prepare() is purely row-local — stateless features plus a lookup
    # join — so chunking is exactly equivalent to a single pass. Doing it in one
    # pass is not: joining the velocity columns onto a 434-column frame forces
    # pandas to consolidate blocks, which for the 506,691-row test split means a
    # single 773 MiB allocation on top of everything already resident. Streaming
    # bounds the peak at roughly chunk_size x n_columns instead.
    reader = pq.ParquetFile(joined_path(split))
    writer: pq.ParquetWriter | None = None
    n_rows = 0
    n_cols = 0
    previous_max_dt: int | None = None

    try:
        for batch in reader.iter_batches(batch_size=chunk_rows):
            chunk = batch.to_pandas()
            if dt_low is not None:
                chunk = chunk[chunk[TIME_COL] > dt_low]
            if dt_high is not None:
                chunk = chunk[chunk[TIME_COL] <= dt_high]
            if chunk.empty:
                continue

            # build_interim() writes in chronological order and batches preserve
            # it, so this is a cheap assertion rather than a re-sort.
            if not chunk[TIME_COL].is_monotonic_increasing:
                chunk = chunk.sort_values([TIME_COL, KEY], kind="mergesort")
            batch_min = int(chunk[TIME_COL].iloc[0])
            if previous_max_dt is not None and batch_min < previous_max_dt:
                raise RuntimeError(
                    "Parquet batches are not in chronological order; velocity joins "
                    "would be misaligned."
                )
            previous_max_dt = int(chunk[TIME_COL].iloc[-1])

            prepared = pipeline.prepare(chunk.reset_index(drop=True), velocity_frame=velocity)
            # Category dtypes carry per-chunk vocabularies, which would give each
            # row group a different Arrow schema. Strings keep the schema stable;
            # load_parquet() restores category on read.
            for column in prepared.columns:
                if isinstance(prepared[column].dtype, pd.CategoricalDtype):
                    prepared[column] = prepared[column].astype("string")

            table = pa.Table.from_pandas(prepared, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
                n_cols = prepared.shape[1]
            writer.write_table(table)
            n_rows += len(prepared)
            del chunk, prepared, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()

    if n_rows == 0:
        raise RuntimeError(f"No rows selected for {out_path.name}")

    logger.info(
        "Wrote %s (%d rows x %d cols, %.1f MB, %.1fs)",
        out_path.name,
        n_rows,
        n_cols,
        out_path.stat().st_size / 1024**2,
        time.perf_counter() - started,
    )
    return n_rows, n_cols


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the modelling dataset.")
    parser.add_argument("--force", action="store_true", help="Rebuild cached artifacts.")
    parser.add_argument(
        "--with-test",
        action="store_true",
        help="Also build the unlabeled test split (used for drift analysis and traffic replay).",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config()
    set_seed(config.train.seed)
    started = time.perf_counter()

    # --- stage 1: join + validate -------------------------------------------
    build_interim("train", force=args.force)
    labels = (
        load_parquet(joined_path("train"), columns=[KEY, TIME_COL, TARGET], categorical=False)
        .sort_values([TIME_COL, KEY], kind="mergesort")
        .reset_index(drop=True)
    )

    audit = load_parquet(
        joined_path("train"), columns=[KEY, TIME_COL, TARGET, "card1", "ProductCD"]
    )
    audit = audit.sort_values([TIME_COL, KEY], kind="mergesort").reset_index(drop=True)
    validate_frame(audit, name="train_joined(projection)", labeled=True)
    del audit

    # --- stage 2: global velocity -------------------------------------------
    velocity = build_velocity_cache(
        "train", config.features.velocity_windows_hours, force=args.force
    )

    # --- stage 3: temporal split + folds ------------------------------------
    split = temporal_split(
        labels,
        holdout_fraction=config.split.holdout_fraction,
        validation_fraction=config.split.validation_fraction,
    )
    save_split(split)

    modelling = labels.iloc[list(split.train_idx) + list(split.validation_idx)].reset_index(
        drop=True
    )
    cv = PurgedForwardChainingCV(
        n_splits=config.split.n_cv_folds, purge_days=config.split.purge_days
    )
    temporal_folds = list(cv.split(modelling))
    save_folds(temporal_folds, "temporal")
    save_folds(
        random_stratified_folds(
            modelling[TARGET], n_splits=config.split.n_cv_folds, seed=config.train.seed
        ),
        "random",
    )
    for i, (train_idx, val_idx) in enumerate(temporal_folds):
        logger.info(
            "fold %d | train %d (fraud %.4f%%) | val %d (fraud %.4f%%)",
            i,
            train_idx.size,
            modelling[TARGET].to_numpy()[train_idx].mean() * 100,
            val_idx.size,
            modelling[TARGET].to_numpy()[val_idx].mean() * 100,
        )

    # --- stage 4: prepare partitions ---------------------------------------
    pipeline = FeaturePipeline(
        velocity_windows_hours=config.features.velocity_windows_hours,
        anchor_d_columns=config.features.anchor_d_columns,
        frequency_min_count=config.features.frequency_encode_min_count,
    )
    holdout_cut = split.metadata.holdout_cut_dt

    modelling_rows, modelling_cols = prepare_partition(
        "train",
        None,
        holdout_cut,
        velocity,
        pipeline,
        PROCESSED_DIR / "modelling_prepared.parquet",
        force=args.force,
    )
    holdout_rows, holdout_cols = prepare_partition(
        "train",
        holdout_cut,
        None,
        velocity,
        pipeline,
        PROCESSED_DIR / "holdout_prepared.parquet",
        force=args.force,
    )

    if modelling_rows + holdout_rows != len(labels):
        raise RuntimeError(
            f"Partition rows {modelling_rows} + {holdout_rows} != total {len(labels)}"
        )
    if modelling_cols != holdout_cols:
        raise RuntimeError(f"Column mismatch: {modelling_cols} vs {holdout_cols}")

    # --- optional: unlabeled test split ------------------------------------
    if args.with_test:
        build_interim("test", force=args.force)
        test_velocity = build_velocity_cache(
            "test", config.features.velocity_windows_hours, force=args.force
        )
        prepare_partition(
            "test",
            None,
            None,
            test_velocity,
            pipeline,
            PROCESSED_DIR / "test_prepared.parquet",
            force=args.force,
        )

    summary = {
        "modelling_rows": modelling_rows,
        "holdout_rows": holdout_rows,
        "prepared_columns": modelling_cols,
        "n_temporal_folds": len(temporal_folds),
        "split": json.loads((PROCESSED_DIR / "split_metadata.json").read_text(encoding="utf-8")),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    (PROCESSED_DIR / "build_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info("Build complete in %.1fs", summary["elapsed_seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
