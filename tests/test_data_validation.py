"""Tests for schema validation and temporal splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.schema import identity_rename_map
from src.data.splitting import (
    PurgedForwardChainingCV,
    SplitError,
    find_time_cut,
    temporal_split,
)
from src.data.validation import (
    DataValidationError,
    missing_value_profile,
    validate_frame,
)


class TestValidateFrame:
    def test_accepts_a_well_formed_labeled_frame(self, raw_frame: pd.DataFrame) -> None:
        report = validate_frame(raw_frame, name="synthetic", labeled=True)
        assert report.n_rows == len(raw_frame)
        assert any("unique" in check for check in report.checks_passed)

    def test_rejects_duplicate_keys(self, raw_frame: pd.DataFrame) -> None:
        broken = pd.concat([raw_frame, raw_frame.iloc[[0]]], ignore_index=True)
        broken = broken.sort_values("TransactionDT").reset_index(drop=True)
        with pytest.raises(DataValidationError, match="not unique"):
            validate_frame(broken, name="dupes", labeled=True)

    def test_rejects_unsorted_time_column(self, raw_frame: pd.DataFrame) -> None:
        shuffled = raw_frame.sample(frac=1.0, random_state=1).reset_index(drop=True)
        with pytest.raises(DataValidationError, match="not sorted"):
            validate_frame(shuffled, name="unsorted", labeled=True)

    def test_rejects_hyphenated_identity_columns(self, raw_frame: pd.DataFrame) -> None:
        broken = raw_frame.rename(columns={"id_12": "id-12"})
        with pytest.raises(DataValidationError, match="hyphenated"):
            validate_frame(broken, name="hyphenated", labeled=True)

    def test_rejects_target_present_in_unlabeled_frame(self, raw_frame: pd.DataFrame) -> None:
        with pytest.raises(DataValidationError, match="present in an unlabeled frame"):
            validate_frame(raw_frame, name="should-be-unlabeled", labeled=False)

    def test_rejects_wrong_row_count(self, raw_frame: pd.DataFrame) -> None:
        with pytest.raises(DataValidationError, match="row count"):
            validate_frame(raw_frame, name="x", labeled=True, expect_rows=len(raw_frame) + 1)

    def test_missing_value_profile_is_sorted_worst_first(self, raw_frame: pd.DataFrame) -> None:
        profile = missing_value_profile(raw_frame)
        assert list(profile["missing_rate"]) == sorted(profile["missing_rate"], reverse=True)
        assert profile.loc[profile["column"] == "C1", "missing_rate"].iloc[0] == 0.0


class TestIdentityRename:
    def test_maps_hyphens_to_underscores(self) -> None:
        mapping = identity_rename_map(["id-01", "id-38", "TransactionID", "id_02"])
        assert mapping == {"id-01": "id_01", "id-38": "id_38"}


class TestFindTimeCut:
    def test_cut_keeps_tie_groups_together(self) -> None:
        # Every timestamp appears 3 times: no cut may split a group.
        timestamps = pd.Series(np.repeat([10, 20, 30, 40, 50], 3))
        cut = find_time_cut(timestamps, 0.8)
        below = (timestamps <= cut).sum()
        assert below % 3 == 0

    def test_rejects_invalid_fraction(self) -> None:
        with pytest.raises(SplitError):
            find_time_cut(pd.Series([1, 2, 3]), 1.5)


class TestTemporalSplit:
    def test_partitions_are_chronologically_ordered_and_disjoint(
        self, raw_frame: pd.DataFrame
    ) -> None:
        split = temporal_split(raw_frame, holdout_fraction=0.2, validation_fraction=0.2)
        timestamps = raw_frame["TransactionDT"].to_numpy()

        assert timestamps[split.train_idx].max() < timestamps[split.validation_idx].min()
        assert timestamps[split.validation_idx].max() < timestamps[split.holdout_idx].min()

        total = split.train_idx.size + split.validation_idx.size + split.holdout_idx.size
        assert total == len(raw_frame)
        assert not set(split.train_idx) & set(split.holdout_idx)

    def test_holdout_is_roughly_the_requested_fraction(self, raw_frame: pd.DataFrame) -> None:
        split = temporal_split(raw_frame, holdout_fraction=0.2)
        assert 0.17 <= split.holdout_idx.size / len(raw_frame) <= 0.23

    def test_rejects_unsorted_input(self, raw_frame: pd.DataFrame) -> None:
        shuffled = raw_frame.sample(frac=1.0, random_state=2).reset_index(drop=True)
        with pytest.raises(SplitError, match="sorted"):
            temporal_split(shuffled)


class TestPurgedForwardChainingCV:
    def test_train_always_precedes_validation_with_a_gap(self, raw_frame: pd.DataFrame) -> None:
        cv = PurgedForwardChainingCV(n_splits=3, purge_days=2)
        timestamps = raw_frame["TransactionDT"].to_numpy()

        for train_idx, validation_idx in cv.split(raw_frame):
            gap = timestamps[validation_idx].min() - timestamps[train_idx].max()
            assert gap > 0, "validation must start after training ends"
            # The purge gap must be at least the configured width.
            assert gap >= 2 * 86_400 - 86_400, "purge gap narrower than configured"

    def test_training_window_grows_monotonically(self, raw_frame: pd.DataFrame) -> None:
        cv = PurgedForwardChainingCV(n_splits=3, purge_days=1)
        sizes = [train_idx.size for train_idx, _ in cv.split(raw_frame)]
        assert sizes == sorted(sizes)
        assert sizes[0] > 0

    def test_folds_never_overlap_in_validation(self, raw_frame: pd.DataFrame) -> None:
        cv = PurgedForwardChainingCV(n_splits=3, purge_days=1)
        seen: set[int] = set()
        for _, validation_idx in cv.split(raw_frame):
            assert not seen & set(validation_idx.tolist())
            seen |= set(validation_idx.tolist())

    def test_rejects_degenerate_configuration(self, raw_frame: pd.DataFrame) -> None:
        # A purge gap wider than the whole dataset leaves no training rows.
        cv = PurgedForwardChainingCV(n_splits=3, purge_days=10_000)
        with pytest.raises(SplitError, match="degenerate"):
            list(cv.split(raw_frame))

    def test_requires_time_column(self) -> None:
        cv = PurgedForwardChainingCV(n_splits=2)
        with pytest.raises(SplitError, match="TransactionDT"):
            list(cv.split(pd.DataFrame({"a": [1, 2, 3]})))
