"""Tests for feature engineering, with emphasis on leakage safety."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import CategoricalCodeEncoder, LinearPreprocessor
from src.data.schema import EXCLUDED_FROM_FEATURES
from src.features.aggregations import EntityAmountAggregator, FrequencyEncoder
from src.features.builders import (
    add_amount_features,
    add_email_features,
    add_entity_keys,
    add_time_features,
    build_stateless_features,
)
from src.features.pipeline import FeaturePipeline
from src.features.velocity import add_velocity_features, compute_velocity_frame


class TestStatelessBuilders:
    def test_time_features_are_cyclical_and_in_range(self, raw_frame: pd.DataFrame) -> None:
        out = add_time_features(raw_frame.copy())
        assert out["hour_of_day"].between(0, 23).all()
        assert out["day_of_week"].between(0, 6).all()
        assert set(out["is_night"].unique()) <= {0, 1}

    def test_amount_features(self, raw_frame: pd.DataFrame) -> None:
        out = add_amount_features(raw_frame.copy())
        expected = np.log1p(raw_frame["TransactionAmt"].to_numpy(dtype="float64"))
        np.testing.assert_allclose(out["log_amount"].to_numpy(), expected, rtol=1e-5)
        assert out["amount_cents"].between(0, 1).all()

    def test_email_match_encodes_missing_as_minus_one(self) -> None:
        frame = pd.DataFrame(
            {
                "P_emaildomain": ["gmail.com", "gmail.com", None, "yahoo.com"],
                "R_emaildomain": ["gmail.com", "yahoo.com", "gmail.com", None],
            }
        )
        out = add_email_features(frame)
        assert out["email_domains_match"].tolist() == [1, 0, -1, -1]

    def test_entity_keys_are_deterministic_across_calls(self, raw_frame: pd.DataFrame) -> None:
        first = add_entity_keys(raw_frame.copy())["_entity_card_full"]
        # A different subset must produce the same key for the same row, which
        # factorize-based codes would not.
        subset = raw_frame.iloc[100:200].copy()
        second = add_entity_keys(subset)["_entity_card_full"]
        pd.testing.assert_series_equal(
            first.iloc[100:200].reset_index(drop=True),
            second.reset_index(drop=True),
            check_names=False,
        )

    def test_entity_keys_reject_low_order_slot_overflow(self, raw_frame: pd.DataFrame) -> None:
        """addr1/card2 must stay below 1000 or they overflow into the next slot."""
        broken = raw_frame.copy()
        broken.loc[0, "addr1"] = 99_999
        with pytest.raises(ValueError, match="positional slot"):
            add_entity_keys(broken)

    def test_entity_keys_allow_card1_beyond_the_training_maximum(
        self, raw_frame: pd.DataFrame
    ) -> None:
        """Regression: card1 occupies the high-order slot, so a value above the
        training maximum cannot cause a collision and must not be rejected.
        The real test split contains card1=18397 against train's 18396."""
        frame = raw_frame.copy()
        frame.loc[0, "card1"] = 18_397
        result = add_entity_keys(frame)
        assert result.loc[0, "_entity_card"] == 18_397
        # Keys must still be recoverable component-wise.
        addr1 = int(frame.loc[0, "addr1"] if pd.notna(frame.loc[0, "addr1"]) else 0)
        assert result.loc[0, "_entity_card_addr"] == 18_397 * 1_000 + addr1

    def test_build_stateless_adds_expected_families(self, raw_frame: pd.DataFrame) -> None:
        out = build_stateless_features(raw_frame.copy())
        for column in (
            "log_amount",
            "hour_of_day",
            "n_missing_total",
            "D1_anchored",
            "p_email_provider",
            "device_vendor",
        ):
            assert column in out.columns


class TestVelocityIsCausal:
    def test_first_transaction_of_an_entity_has_no_history(self) -> None:
        frame = pd.DataFrame(
            {
                "TransactionID": [1, 2, 3],
                "TransactionDT": [1000, 2000, 3000],
                "TransactionAmt": [10.0, 20.0, 30.0],
                "_entity_card": [111, 111, 222],
            }
        )
        out = add_velocity_features(frame.copy(), ("_entity_card",), windows_hours=(1,))
        assert out["entity_card_txn_count_1h"].tolist() == [0.0, 1.0, 0.0]
        assert out["entity_card_amt_sum_1h"].tolist() == [0.0, 10.0, 0.0]
        assert np.isnan(out["entity_card_seconds_since_prev"].iloc[0])
        assert out["entity_card_seconds_since_prev"].iloc[1] == 1000

    def test_counts_exclude_the_current_row(self) -> None:
        # Five transactions on one card, one minute apart.
        frame = pd.DataFrame(
            {
                "TransactionID": range(5),
                "TransactionDT": [0, 60, 120, 180, 240],
                "TransactionAmt": [1.0] * 5,
                "_entity_card": [7] * 5,
            }
        )
        out = add_velocity_features(frame.copy(), ("_entity_card",), windows_hours=(1,))
        assert out["entity_card_txn_count_1h"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_window_boundary_excludes_older_transactions(self) -> None:
        frame = pd.DataFrame(
            {
                "TransactionID": [1, 2],
                # 2 hours apart: outside a 1-hour window.
                "TransactionDT": [0, 7200],
                "TransactionAmt": [5.0, 5.0],
                "_entity_card": [1, 1],
            }
        )
        out = add_velocity_features(frame.copy(), ("_entity_card",), windows_hours=(1,))
        assert out["entity_card_txn_count_1h"].tolist() == [0.0, 0.0]

    def test_entities_do_not_leak_into_each_other(self) -> None:
        frame = pd.DataFrame(
            {
                "TransactionID": range(6),
                "TransactionDT": [0, 1, 2, 3, 4, 5],
                "TransactionAmt": [1.0] * 6,
                "_entity_card": [1, 2, 1, 2, 1, 2],
            }
        )
        out = add_velocity_features(frame.copy(), ("_entity_card",), windows_hours=(1,))
        assert out["entity_card_txn_count_1h"].tolist() == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]

    def test_requires_sorted_input(self) -> None:
        frame = pd.DataFrame(
            {
                "TransactionID": [1, 2],
                "TransactionDT": [100, 50],
                "TransactionAmt": [1.0, 1.0],
                "_entity_card": [1, 1],
            }
        )
        with pytest.raises(ValueError, match="sorted"):
            add_velocity_features(frame, ("_entity_card",))

    def test_velocity_frame_matches_inline_computation(self, raw_frame: pd.DataFrame) -> None:
        narrow = add_entity_keys(
            raw_frame[
                ["TransactionID", "TransactionDT", "TransactionAmt", "card1", "addr1", "card2"]
            ].copy()
        )
        frame_result = compute_velocity_frame(narrow.copy(), ("_entity_card",), (24,))
        inline = add_velocity_features(narrow.copy(), ("_entity_card",), (24,))
        np.testing.assert_allclose(
            frame_result.loc[narrow["TransactionID"], "entity_card_txn_count_24h"].to_numpy(),
            inline["entity_card_txn_count_24h"].to_numpy(),
        )


class TestFittedEncodersDoNotLeak:
    def test_frequency_encoder_uses_only_fitted_counts(self) -> None:
        train = pd.DataFrame({"card1": [1, 1, 1, 2, 2]})
        test = pd.DataFrame({"card1": [1, 2, 999]})
        encoder = FrequencyEncoder().fit(train, ["card1"])
        out = encoder.transform(test)
        # 999 is unseen in training -> 0, not a count derived from the test set.
        assert out["card1_freq"].tolist() == [3.0, 2.0, 0.0]

    def test_entity_aggregator_returns_nan_for_unseen_entities(self) -> None:
        train = pd.DataFrame({"_entity_card": [1, 1, 2], "log_amount": [1.0, 3.0, 5.0]})
        test = pd.DataFrame({"_entity_card": [1, 99], "log_amount": [2.0, 2.0]})
        aggregator = EntityAmountAggregator().fit(train, ["_entity_card"])
        out = aggregator.transform(test)
        assert out["entity_card_amt_mean_hist"].iloc[0] == pytest.approx(2.0)
        assert np.isnan(out["entity_card_amt_mean_hist"].iloc[1])

    def test_encoders_do_not_mutate_their_input(self) -> None:
        train = pd.DataFrame({"card1": [1, 1, 2]})
        before = list(train.columns)
        FrequencyEncoder().fit(train, ["card1"]).transform(train)
        assert list(train.columns) == before


class TestFeaturePipeline:
    def test_excluded_columns_never_reach_the_model(self, fitted_pipeline) -> None:
        for column in EXCLUDED_FROM_FEATURES:
            assert column not in fitted_pipeline.feature_names
        assert not any(name.startswith("_") for name in fitted_pipeline.feature_names)

    def test_transform_is_stable_in_shape_and_order(
        self, fitted_pipeline, prepared_frame: pd.DataFrame
    ) -> None:
        first = fitted_pipeline.transform(prepared_frame.iloc[:100])
        second = fitted_pipeline.transform(prepared_frame.iloc[100:200])
        assert list(first.columns) == list(second.columns) == fitted_pipeline.feature_names

    def test_transform_before_fit_raises(self, prepared_frame: pd.DataFrame) -> None:
        with pytest.raises(RuntimeError, match="before fit"):
            FeaturePipeline().transform(prepared_frame)

    def test_categoricals_use_the_training_vocabulary(
        self, fitted_pipeline, prepared_frame: pd.DataFrame
    ) -> None:
        out = fitted_pipeline.transform(prepared_frame.iloc[:50])
        for column in fitted_pipeline.categorical_features:
            expected = set(fitted_pipeline.categorical_encoder.vocabularies[column])
            assert set(out[column].cat.categories) == expected

    def test_save_and_load_roundtrip(self, fitted_pipeline, tmp_path) -> None:
        path = fitted_pipeline.save(tmp_path / "pipeline.pkl")
        loaded = FeaturePipeline.load(path)
        assert loaded.feature_names == fitted_pipeline.feature_names


class TestPreprocessing:
    def test_categorical_encoder_maps_unseen_to_nan(self) -> None:
        train = pd.DataFrame({"c": pd.Series(["a", "b"], dtype="category")})
        encoder = CategoricalCodeEncoder().fit(train, ["c"])
        out = encoder.transform(pd.DataFrame({"c": ["a", "zzz"]}))
        assert out["c"].iloc[0] == "a"
        assert pd.isna(out["c"].iloc[1])

    def test_linear_preprocessor_produces_finite_matrix_with_indicators(self) -> None:
        frame = pd.DataFrame(
            {
                "num": [1.0, np.nan, 3.0, 100.0],
                "cat": pd.Series(["a", "a", "b", None], dtype="category"),
            }
        )
        preprocessor = LinearPreprocessor(min_frequency=1)
        matrix = preprocessor.fit_transform(frame)
        assert np.isfinite(matrix).all()
        assert matrix.shape[0] == 4
        assert "num__isna" in preprocessor.feature_names
        assert matrix.shape[1] == len(preprocessor.feature_names)

    def test_linear_preprocessor_handles_zero_variance_columns(self) -> None:
        frame = pd.DataFrame({"constant": [5.0, 5.0, 5.0]})
        matrix = LinearPreprocessor().fit_transform(frame)
        assert np.isfinite(matrix).all()
