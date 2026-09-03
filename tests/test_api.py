"""API contract tests.

These run against a real FastAPI TestClient with a real (small) trained model, so
they exercise the full path: Pydantic validation, frame construction, the feature
pipeline, the model, calibration and SHAP.
"""

from __future__ import annotations

import pytest


class TestHealth:
    def test_reports_loaded_model_and_provenance(self, api_client) -> None:
        response = api_client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_name"] == "lightgbm"
        assert body["n_features"] > 0
        assert body["trained_at"]
        assert 0.0 <= body["decision_threshold"] <= 1.0
        assert body["uptime_seconds"] >= 0

    def test_degraded_without_a_model(self, api_client) -> None:
        """A missing model is 'degraded', not a crash — that distinction is the
        point of a health check."""
        from api import dependencies

        original = dependencies.state.artifact
        dependencies.state.artifact = None
        try:
            body = api_client.get("/health").json()
            assert body["status"] == "degraded"
            assert body["model_loaded"] is False
        finally:
            dependencies.state.artifact = original

    def test_root_points_at_docs(self, api_client) -> None:
        assert api_client.get("/").json()["docs"] == "/docs"


class TestPredict:
    def test_returns_valid_probability_and_risk_band(
        self, api_client, valid_transaction_payload
    ) -> None:
        response = api_client.post("/predict", json=valid_transaction_payload)
        assert response.status_code == 200, response.text
        body = response.json()
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert body["risk_level"] in {"low", "medium", "high"}
        assert isinstance(body["flagged"], bool)
        assert body["latency_ms"] > 0
        assert body["features_supplied"] > 0

    def test_risk_band_matches_the_probability(self, api_client, valid_transaction_payload) -> None:
        body = api_client.post("/predict", json=valid_transaction_payload).json()
        probability = body["fraud_probability"]
        expected = "high" if probability >= 0.70 else "medium" if probability >= 0.30 else "low"
        assert body["risk_level"] == expected

    def test_flagged_matches_the_threshold(self, api_client, valid_transaction_payload) -> None:
        body = api_client.post("/predict", json=valid_transaction_payload).json()
        assert body["flagged"] == (body["fraud_probability"] >= body["decision_threshold"])

    def test_works_with_only_required_fields(self, api_client) -> None:
        """Omitted features become NaN, which the model routes natively."""
        response = api_client.post("/predict", json={"transaction_amt": 50.0, "product_cd": "W"})
        assert response.status_code == 200, response.text
        assert 0.0 <= response.json()["fraud_probability"] <= 1.0

    def test_returns_503_without_a_model(self, api_client, valid_transaction_payload) -> None:
        from api import dependencies

        original = dependencies.state.artifact
        dependencies.state.artifact = None
        try:
            response = api_client.post("/predict", json=valid_transaction_payload)
            assert response.status_code == 503
            assert "not loaded" in response.json()["detail"]
        finally:
            dependencies.state.artifact = original


class TestPredictValidation:
    @pytest.mark.parametrize(
        "payload,reason",
        [
            ({}, "missing required fields"),
            ({"transaction_amt": 100.0}, "missing product_cd"),
            ({"transaction_amt": -5.0, "product_cd": "W"}, "negative amount"),
            ({"transaction_amt": 0.0, "product_cd": "W"}, "zero amount"),
            ({"transaction_amt": "abc", "product_cd": "W"}, "non-numeric amount"),
            ({"transaction_amt": 10.0, "product_cd": ""}, "empty product code"),
            ({"transaction_amt": 10.0, "product_cd": "W", "card1": -1}, "negative card1"),
            (
                {"transaction_amt": 10.0, "product_cd": "W", "unknown_field": 1},
                "unknown field rejected by extra='forbid'",
            ),
        ],
    )
    def test_invalid_payloads_return_422(self, api_client, payload, reason) -> None:
        response = api_client.post("/predict", json=payload)
        assert response.status_code == 422, f"{reason}: {response.text}"
        body = response.json()
        assert body["error_type"] == "RequestValidationError"
        assert "errors" in body


class TestExplain:
    def test_returns_ranked_shap_contributions(self, api_client, valid_transaction_payload) -> None:
        response = api_client.post("/explain", json=valid_transaction_payload)
        assert response.status_code == 200, response.text
        body = response.json()
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert body["top_factors"], "expected at least one contributing feature"

        magnitudes = [abs(f["shap_value"]) for f in body["top_factors"]]
        assert magnitudes == sorted(magnitudes, reverse=True), "must be ranked by magnitude"
        for factor in body["top_factors"]:
            assert factor["direction"] in {"increases_risk", "decreases_risk"}
            assert factor["feature"]

    def test_top_n_is_respected(self, api_client, valid_transaction_payload) -> None:
        body = api_client.post("/explain?top_n=3", json=valid_transaction_payload).json()
        assert len(body["top_factors"]) == 3

    def test_invalid_top_n_is_rejected(self, api_client, valid_transaction_payload) -> None:
        assert (
            api_client.post("/explain?top_n=999", json=valid_transaction_payload).status_code == 422
        )

    def test_explanation_agrees_with_prediction(
        self, api_client, valid_transaction_payload
    ) -> None:
        """The same transaction must score identically through both routes."""
        predicted = api_client.post("/predict", json=valid_transaction_payload).json()
        explained = api_client.post("/explain", json=valid_transaction_payload).json()
        assert predicted["fraud_probability"] == pytest.approx(
            explained["fraud_probability"], abs=1e-6
        )


class TestColdStartVelocity:
    """Velocity features have no external store; a lone transaction must yield a
    zero-history state rather than fabricated activity."""

    def test_single_transaction_has_zero_prior_activity(
        self, api_client, trained_artifact, valid_transaction_payload
    ) -> None:
        from api.dependencies import build_prepared_frame
        from api.schemas import TransactionRequest

        record = TransactionRequest(**valid_transaction_payload).to_raw_record()
        prepared = build_prepared_frame([record], trained_artifact, default_timestamp=12_000_000)
        for column in prepared.columns:
            if column.endswith(("_txn_count_1h", "_txn_count_24h", "_amt_sum_1h")):
                assert prepared[column].iloc[0] == 0.0, f"{column} should be 0 with no history"
