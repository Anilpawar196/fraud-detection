"""Phase 1 — dataset inspection.

Profiles the raw IEEE-CIS CSV files without assuming anything about their
schema, and writes a machine-readable + human-readable audit to ``reports/``.

The files are large (train_transaction.csv alone is several hundred MB and
would occupy roughly 2 GB as float64), so every pass is chunked: we accumulate
streaming statistics and never hold a full frame in memory. This makes the
audit runnable on a laptop and inside CI.

Usage:
    python scripts/inspect_dataset.py
    python scripts/inspect_dataset.py --chunksize 100000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_config import setup_logging  # noqa: E402
from src.utils.paths import RAW_DIR, REPORTS_DIR, ensure_dir  # noqa: E402

logger = logging.getLogger("inspect_dataset")

EXPECTED_FILES = (
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
)

#: Above this many distinct values we stop tracking them individually. Keeps
#: memory bounded on high-cardinality columns such as TransactionID.
NUNIQUE_CAP = 1000

#: Columns whose distinct values we always want to see in full, because the
#: join strategy and target analysis depend on them.
KEY_CANDIDATES = ("TransactionID",)


@dataclass
class ColumnStats:
    """Streaming statistics for a single column."""

    name: str
    dtypes: set[str] = field(default_factory=set)
    n_total: int = 0
    n_null: int = 0
    distinct: set[Any] = field(default_factory=set)
    distinct_overflowed: bool = False
    # Numeric accumulators (Welford is unnecessary here; magnitudes are modest
    # and we only report descriptive stats).
    numeric_count: int = 0
    numeric_sum: float = 0.0
    numeric_sumsq: float = 0.0
    numeric_min: float | None = None
    numeric_max: float | None = None
    samples: list[Any] = field(default_factory=list)

    def update(self, series: pd.Series) -> None:
        self.dtypes.add(str(series.dtype))
        self.n_total += len(series)
        self.n_null += int(series.isna().sum())

        non_null = series.dropna()
        if non_null.empty:
            return

        if not self.distinct_overflowed:
            self.distinct.update(non_null.unique().tolist())
            if len(self.distinct) > NUNIQUE_CAP:
                self.distinct_overflowed = True
                self.distinct.clear()

        if len(self.samples) < 5:
            self.samples.extend(non_null.head(5 - len(self.samples)).tolist())

        if pd.api.types.is_numeric_dtype(non_null) and not pd.api.types.is_bool_dtype(non_null):
            values = non_null.astype("float64").to_numpy()
            self.numeric_count += values.size
            self.numeric_sum += float(values.sum())
            self.numeric_sumsq += float(np.square(values).sum())
            chunk_min, chunk_max = float(values.min()), float(values.max())
            self.numeric_min = (
                chunk_min if self.numeric_min is None else min(self.numeric_min, chunk_min)
            )
            self.numeric_max = (
                chunk_max if self.numeric_max is None else max(self.numeric_max, chunk_max)
            )

    @property
    def missing_rate(self) -> float:
        return self.n_null / self.n_total if self.n_total else 0.0

    @property
    def n_unique(self) -> int | None:
        """Distinct non-null values, or ``None`` if the cap was exceeded."""
        return None if self.distinct_overflowed else len(self.distinct)

    def to_dict(self) -> dict[str, Any]:
        mean = std = None
        if self.numeric_count:
            mean = self.numeric_sum / self.numeric_count
            variance = max(self.numeric_sumsq / self.numeric_count - mean**2, 0.0)
            std = float(np.sqrt(variance))
        return {
            "column": self.name,
            "dtypes": sorted(self.dtypes),
            "n_total": self.n_total,
            "n_null": self.n_null,
            "missing_rate": round(self.missing_rate, 6),
            "n_unique": self.n_unique,
            "n_unique_capped_at": NUNIQUE_CAP if self.distinct_overflowed else None,
            "is_numeric": self.numeric_count > 0,
            "min": self.numeric_min,
            "max": self.numeric_max,
            "mean": mean,
            "std": std,
            "sample_values": [_jsonable(v) for v in self.samples],
        }


@dataclass
class FileProfile:
    """Profile of one CSV file."""

    path: Path
    n_rows: int = 0
    size_bytes: int = 0
    columns: dict[str, ColumnStats] = field(default_factory=dict)
    duplicate_rows: int = 0
    key_values: set[Any] = field(default_factory=set)
    key_column: str | None = None
    target_counts: dict[str, int] = field(default_factory=dict)

    @property
    def n_cols(self) -> int:
        return len(self.columns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.path.name,
            "size_mb": round(self.size_bytes / 1024**2, 2),
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "duplicate_full_rows": self.duplicate_rows,
            "key_column": self.key_column,
            "n_unique_keys": len(self.key_values) if self.key_values else None,
            "target_counts": self.target_counts,
            "columns": [c.to_dict() for c in self.columns.values()],
        }


def _jsonable(value: Any) -> Any:
    """Coerce numpy/pandas scalars into JSON-serialisable Python objects."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None or isinstance(value, int | float | str | bool):
        return value
    return str(value)


def profile_csv(path: Path, chunksize: int, target_column: str = "isFraud") -> FileProfile:
    """Stream ``path`` and accumulate per-column and per-file statistics."""
    logger.info("Profiling %s (%.1f MB)", path.name, path.stat().st_size / 1024**2)
    profile = FileProfile(path=path, size_bytes=path.stat().st_size)
    seen_row_hashes: set[int] = set()

    reader: Iterable[pd.DataFrame] = pd.read_csv(path, chunksize=chunksize, low_memory=False)
    for i, chunk in enumerate(reader):
        profile.n_rows += len(chunk)

        for col in chunk.columns:
            stats = profile.columns.get(col)
            if stats is None:
                stats = ColumnStats(name=col)
                profile.columns[col] = stats
            stats.update(chunk[col])

        # Full-row duplicates, detected across chunk boundaries via content hash.
        hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy()
        before = len(seen_row_hashes)
        seen_row_hashes.update(hashes.tolist())
        profile.duplicate_rows += len(chunk) - (len(seen_row_hashes) - before)

        # Join key tracking.
        for candidate in KEY_CANDIDATES:
            if candidate in chunk.columns:
                profile.key_column = candidate
                profile.key_values.update(chunk[candidate].tolist())

        # Target distribution.
        if target_column in chunk.columns:
            counts = chunk[target_column].value_counts(dropna=False)
            for value, count in counts.items():
                key = str(_jsonable(value))
                profile.target_counts[key] = profile.target_counts.get(key, 0) + int(count)

        if (i + 1) % 5 == 0:
            logger.info("  ... %d rows", profile.n_rows)

    logger.info("Done: %s rows=%d cols=%d", path.name, profile.n_rows, profile.n_cols)
    return profile


def analyse_join(transaction: FileProfile, identity: FileProfile) -> dict[str, Any]:
    """Quantify the transaction <-> identity relationship on the join key."""
    t_keys, i_keys = transaction.key_values, identity.key_values
    both = t_keys & i_keys
    return {
        "transaction_file": transaction.path.name,
        "identity_file": identity.path.name,
        "join_key": transaction.key_column,
        "transaction_rows": transaction.n_rows,
        "transaction_unique_keys": len(t_keys),
        "transaction_key_is_unique": len(t_keys) == transaction.n_rows,
        "identity_rows": identity.n_rows,
        "identity_unique_keys": len(i_keys),
        "identity_key_is_unique": len(i_keys) == identity.n_rows,
        "keys_in_both": len(both),
        "identity_coverage_of_transactions": (
            round(len(both) / len(t_keys), 6) if t_keys else None
        ),
        "identity_keys_missing_from_transactions": len(i_keys - t_keys),
    }


def temporal_summary(
    profiles: dict[str, FileProfile], column: str = "TransactionDT"
) -> dict[str, Any]:
    """Report the range of the time column per file, to test for overlap."""
    out: dict[str, Any] = {"column": column, "ranges": {}, "note": ""}
    for name, profile in profiles.items():
        stats = profile.columns.get(column)
        if stats is not None:
            out["ranges"][name] = {"min": stats.numeric_min, "max": stats.numeric_max}
    train = out["ranges"].get("train_transaction.csv")
    test = out["ranges"].get("test_transaction.csv")
    if train and test and train["max"] is not None and test["min"] is not None:
        disjoint = test["min"] > train["max"]
        out["train_test_disjoint_in_time"] = disjoint
        out["note"] = (
            "Test period starts after train ends — the split is temporal, so "
            "validation must be time-aware."
            if disjoint
            else "Train and test time ranges overlap — check the split design."
        )
    return out


def flagged_columns(profile: FileProfile, missing_threshold: float = 0.90) -> dict[str, list[str]]:
    """Columns that are unusable or need care, derived from the profile."""
    all_null, constant, high_missing = [], [], []
    for stats in profile.columns.values():
        if stats.n_null == stats.n_total:
            all_null.append(stats.name)
        elif stats.n_unique == 1:
            constant.append(stats.name)
        elif stats.missing_rate >= missing_threshold:
            high_missing.append(stats.name)
    return {
        "all_null": all_null,
        "constant": constant,
        f"missing_ge_{int(missing_threshold * 100)}pct": high_missing,
    }


def write_markdown(audit: dict[str, Any], out_path: Path) -> None:
    """Render the audit as a readable markdown report."""
    lines: list[str] = ["# Dataset Audit — IEEE-CIS Fraud Detection", ""]
    lines.append("Generated by `scripts/inspect_dataset.py`. All numbers are measured.")
    lines.append("")

    lines += [
        "## File overview",
        "",
        "| file | size (MB) | rows | cols | dup rows | unique keys |",
        "|---|---|---|---|---|---|",
    ]
    for f in audit["files"]:
        keys = f"{f['n_unique_keys']:,}" if f["n_unique_keys"] is not None else "n/a"
        lines.append(
            f"| {f['file']} | {f['size_mb']} | {f['n_rows']:,} | {f['n_cols']} | "
            f"{f['duplicate_full_rows']} | {keys} |"
        )
    lines.append("")

    lines += ["## Target distribution", ""]
    for f in audit["files"]:
        if f["target_counts"]:
            total = sum(f["target_counts"].values())
            lines.append(f"**{f['file']}** (n={total:,})")
            lines.append("")
            for value, count in sorted(f["target_counts"].items()):
                lines.append(f"- `{value}`: {count:,} ({count / total:.4%})")
            lines.append("")
        else:
            lines.append(f"**{f['file']}**: no target column present.")
            lines.append("")

    lines += ["## Join analysis", ""]
    for j in audit["joins"]:
        lines.append(f"### {j['transaction_file']} <- {j['identity_file']}")
        lines.append("")
        for key, value in j.items():
            if key not in ("transaction_file", "identity_file"):
                lines.append(f"- {key}: `{value}`")
        lines.append("")

    lines += ["## Temporal structure", "", f"Column: `{audit['temporal']['column']}`", ""]
    for name, rng in audit["temporal"]["ranges"].items():
        lines.append(f"- {name}: min=`{rng['min']}` max=`{rng['max']}`")
    lines.append("")
    if audit["temporal"].get("note"):
        lines.append(f"> {audit['temporal']['note']}")
        lines.append("")

    lines += ["## Flagged columns", ""]
    for file_name, flags in audit["flagged"].items():
        lines.append(f"### {file_name}")
        lines.append("")
        for label, cols in flags.items():
            lines.append(
                f"- **{label}** ({len(cols)}): {', '.join(f'`{c}`' for c in cols) or '_none_'}"
            )
        lines.append("")

    lines += ["## Missing-value profile (worst 30 columns per file)", ""]
    for f in audit["files"]:
        lines.append(f"### {f['file']}")
        lines.append("")
        lines.append("| column | missing % | n_unique | dtype |")
        lines.append("|---|---|---|---|")
        worst = sorted(f["columns"], key=lambda c: -c["missing_rate"])[:30]
        for c in worst:
            lines.append(
                f"| {c['column']} | {c['missing_rate']:.2%} | "
                f"{c['n_unique'] if c['n_unique'] is not None else f'>{NUNIQUE_CAP}'} | "
                f"{'/'.join(c['dtypes'])} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile the raw IEEE-CIS CSV files.")
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Rows per read_csv chunk (default: 100000).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Directory holding the raw CSVs (default: data/raw).",
    )
    args = parser.parse_args()
    setup_logging()

    missing = [name for name in EXPECTED_FILES if not (args.raw_dir / name).is_file()]
    if missing:
        logger.error(
            "Missing %d expected file(s) in %s: %s",
            len(missing),
            args.raw_dir,
            ", ".join(missing),
        )
        logger.error("See data/README.md for how to obtain the dataset.")
        return 1

    profiles = {
        name: profile_csv(args.raw_dir / name, chunksize=args.chunksize) for name in EXPECTED_FILES
    }

    audit: dict[str, Any] = {
        "files": [p.to_dict() for p in profiles.values()],
        "joins": [
            analyse_join(profiles["train_transaction.csv"], profiles["train_identity.csv"]),
            analyse_join(profiles["test_transaction.csv"], profiles["test_identity.csv"]),
        ],
        "temporal": temporal_summary(profiles),
        "flagged": {name: flagged_columns(p) for name, p in profiles.items()},
        "schema_diff": {
            "in_train_transaction_not_test": sorted(
                set(profiles["train_transaction.csv"].columns)
                - set(profiles["test_transaction.csv"].columns)
            ),
            "in_test_transaction_not_train": sorted(
                set(profiles["test_transaction.csv"].columns)
                - set(profiles["train_transaction.csv"].columns)
            ),
            "in_train_identity_not_test": sorted(
                set(profiles["train_identity.csv"].columns)
                - set(profiles["test_identity.csv"].columns)
            ),
            "in_test_identity_not_train": sorted(
                set(profiles["test_identity.csv"].columns)
                - set(profiles["train_identity.csv"].columns)
            ),
        },
    }

    ensure_dir(REPORTS_DIR)
    json_path = REPORTS_DIR / "dataset_audit.json"
    json_path.write_text(json.dumps(audit, indent=2, default=_jsonable), encoding="utf-8")
    logger.info("Wrote %s", json_path)
    write_markdown(audit, REPORTS_DIR / "dataset_audit.md")

    # Console summary so the run is self-verifying.
    for f in audit["files"]:
        logger.info(
            "%-24s rows=%-8d cols=%-4d dup=%d",
            f["file"],
            f["n_rows"],
            f["n_cols"],
            f["duplicate_full_rows"],
        )
    logger.info(
        "Schema diff (train_transaction - test_transaction): %s",
        audit["schema_diff"]["in_train_transaction_not_test"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
