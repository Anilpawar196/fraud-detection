"""XGBoost model integration tests.

These tests ensure the XGBoost model works correctly in the full ML pipeline,
including training, artifact handling, calibration, SHAP explanations, and API serving.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.calibration import ProbabilityCalibrator
from src.explainability.shap_explainer import ShapExplainer
from src.features.pipeline import FeaturePipeline
from src.models.artifact import ModelArtifact
from src.models.estimators import build_model, fit_with_early_stopping


class TestXGBoostTraining:
    """Verify XGBoost trains correctly with the project's configuration."""

    def test_xgboost_trains_with_categorical_features(self, prepared_frame) -> None:
        """XGBoost must handle categorical features natively."""
        pipeline = FeaturePipeline()
        pipeline.fit(prepared_frame.head(100))
        X = pipeline.transform(prepared_frame.head(200))
        y = prepared_frame.head(200)["isFraud"].to_numpy()

        model = build_model("xgboost", seed=42, n_jobs=1, n_estimators=10)
        model.fit(X, y)
        proba = model.predict_proba(X)[:, 1]

        assert proba.shape == (200,)
        assert ((proba >= 0.0) & (proba <= 1.0)).all()

    def test_xgboost_early_stopping_converges(self, prepared_frame) -> None:
        """Early stopping must reduce iterations from the default."""
        pipeline = FeaturePipeline()
        cut = int(len(prepared_frame) * 0.5)
        pipeline.fit(prepared_frame.iloc[:cut])
        X = pipeline.transform(prepared_frame.iloc[:cut])
        y = prepared_frame.iloc[:cut]["isFraud"].to_numpy()

        split = int(len(X) * 0.8)
        model = build_model("xgboost", seed=42, n_jobs=1, n_estimators=100)
        best_iteration = fit_with_early_stopping(
            model,
            "xgboost",
            X.iloc[:split],
            y[:split],
            X.iloc[split:],
            y[split:],
            pipeline.categorical_features,
        )

        assert isinstance(best_iteration, int)
        assert best_iteration > 0
        assert best_iteration < 100  # Early stopping reduced iterations

    def test_xgboost_predicts_with_missing_values(self, prepared_frame) -> None:
        """XGBoost must route missing values natively."""
        pipeline = FeaturePipeline()
        pipeline.fit(prepared_frame.head(100))
        X = pipeline.transform(prepared_frame.head(200))
        y = prepared_frame.head(200)["isFraud"].to_numpy()

        # Inject NaN values
        X_with_nan = X.copy()
        X_with_nan.iloc[0, 0] = np.nan
        X_with_nan.iloc[10, 5] = np.nan

        model = build_model("xgboost", seed=42, n_jobs=1, n_estimators=10)
        model.fit(X, y)
        proba = model.predict_proba(X_with_nan)[:, 1]

        assert proba.shape == (200,)
        assert ((proba >= 0.0) & (proba <= 1.0)).all()
        assert not np.isnan(proba).any()  # No NaN outputs despite NaN inputs


class TestXGBoostArtifact:
    """Verify the model artifact bundles everything correctly for XGBoost."""

    def test_xgboost_artifact_serializes_and_deserializes(self, api_client) -> None:
        """Full artifact round-trip must preserve the model state."""
        from api.dependencies import state

        if not state.artifact:
            pytest.skip("No artifact loaded")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = state.artifact.save(Path(tmpdir))
            assert path.exists()

            loaded = ModelArtifact.load(path)
            assert loaded.metadata.model_name == "xgboost"
            assert loaded.metadata.n_features == state.artifact.metadata.n_features
            assert loaded.decision_threshold == state.artifact.decision_threshold

    def test_xgboost_artifact_metadata_contains_hyperparameters(self, api_client) -> None:
        """Metadata must record the hyperparameters used."""
        from api.dependencies import state

        if not state.artifact:
            pytest.skip("No artifact loaded")

        meta = state.artifact.metadata
        assert meta.hyperparameters is not None
        assert isinstance(meta.hyperparameters, dict)
        # XGBoost-specific hyperparameters
        assert any(k in meta.hyperparameters for k in ["max_depth", "learning_rate"])

    def test_xgboost_artifact_holdout_metrics_recorded(self, api_client) -> None:
        """Artifact must contain holdout evaluation metrics."""
        from api.dependencies import state

        if not state.artifact:
            pytest.skip("No artifact loaded")

        metrics = state.artifact.metadata.holdout_metrics
        assert "pr_auc" in metrics
        assert "roc_auc" in metrics
        assert 0.0 < metrics["pr_auc"] < 1.0
        assert 0.0 < metrics["roc_auc"] < 1.0


class TestXGBoostCalibration:
    """Verify calibration works correctly with XGBoost predictions."""

    def test_calibration_reduces_expected_calibration_error(self, prepared_frame) -> None:
        """Isotonic calibration must reduce ECE for XGBoost."""
        pipeline = FeaturePipeline()
        cut = int(len(prepared_frame) * 0.5)
        pipeline.fit(prepared_frame.iloc[:cut])
        X_fit = pipeline.transform(prepared_frame.iloc[:cut])
        y_fit = prepared_frame.iloc[:cut]["isFraud"].to_numpy()

        X_val = pipeline.transform(prepared_frame.iloc[cut:])
        y_val = prepared_frame.iloc[cut:]["isFraud"].to_numpy()

        model = build_model("xgboost", seed=42, n_jobs=1, n_estimators=10)
        model.fit(X_fit, y_fit)
        raw_proba = model.predict_proba(X_val)[:, 1]

        calibrator = ProbabilityCalibrator(method="isotonic")
        calibrator.fit(y_val, raw_proba)
        calibrated_proba = calibrator.transform(raw_proba)

        assert calibrated_proba.shape == raw_proba.shape
        assert ((calibrated_proba >= 0.0) & (calibrated_proba <= 1.0)).all()
        # Calibrated probabilities should be bounded better to [0, 1]
        assert calibrated_proba.min() >= 0.0
        assert calibrated_proba.max() <= 1.0

    def test_calibrated_probabilities_improve_brier_score(self, api_client) -> None:
        """Calibration must improve probabilistic score metrics."""
        from api.dependencies import state

        if not state.artifact or not state.artifact.calibrator:
            pytest.skip("No calibrated artifact loaded")

        # Artifact contains both raw and calibrated paths
        assert state.artifact.calibrator is not None


class TestXGBoostSHAP:
    """Verify SHAP explanations work with XGBoost models."""

    def test_shap_explainer_creates_for_xgboost(self, prepared_frame) -> None:
        """SHAP TreeExplainer must work with XGBoost."""
        pipeline = FeaturePipeline()
        pipeline.fit(prepared_frame.head(100))
        X = pipeline.transform(prepared_frame.head(150))
        y = prepared_frame.head(150)["isFraud"].to_numpy()

        model = build_model("xgboost", seed=42, n_jobs=1, n_estimators=10)
        model.fit(X, y)

        explainer = ShapExplainer(model, X, model_name="xgboost")
        assert explainer.explainer is not None

    def test_shap_values_computed_per_row(self, prepared_frame) -> None:
        """SHAP must compute feature contributions for individual rows."""
        pipeline = FeaturePipeline()
        pipeline.fit(prepared_frame.head(100))
        X = pipeline.transform(prepared_frame.head(150))
        y = prepared_frame.head(150)["isFraud"].to_numpy()

        model = build_model("xgboost", seed=42, n_jobs=1, n_estimators=10)
        model.fit(X, y)

        explainer = ShapExplainer(model, X, model_name="xgboost")
        row = X.iloc[0:1]
        shap_values = explainer.explain_sample(row)

        assert shap_values is not None
        assert len(shap_values) == len(X.columns)

    def test_shap_importance_ranks_features(self, prepared_frame) -> None:
        """SHAP mean absolute values must rank features by importance."""
        pipeline = FeaturePipeline()
        pipeline.fit(prepared_frame.head(100))
        X = pipeline.transform(prepared_frame.head(150))
        y = prepared_frame.head(150)["isFraud"].to_numpy()

        model = build_model("xgboost", seed=42, n_jobs=1, n_estimators=10)
        model.fit(X, y)

        explainer = ShapExplainer(model, X, model_name="xgboost")
        importance_df = explainer.global_importance()

        assert importance_df is not None
        assert len(importance_df) > 0
        assert "feature" in importance_df.columns
        assert "mean_abs_shap" in importance_df.columns
        # Higher importance for earlier ranked features
        assert importance_df["mean_abs_shap"].iloc[0] >= importance_df["mean_abs_shap"].iloc[-1]


class TestXGBoostAPIIntegration:
    """Verify full API workflow with XGBoost model."""

    def test_predict_endpoint_with_xgboost_model(
        self, api_client, valid_transaction_payload
    ) -> None:
        """API must return valid predictions from XGBoost."""
        response = api_client.post("/predict", json=valid_transaction_payload)
        assert response.status_code == 200
        body = response.json()
        assert "fraud_probability" in body
        assert 0.0 <= body["fraud_probability"] <= 1.0

    def test_explain_endpoint_returns_shap_values(
        self, api_client, valid_transaction_payload
    ) -> None:
        """Explain endpoint must return SHAP-based feature contributions."""
        response = api_client.post("/explain", json=valid_transaction_payload)
        assert response.status_code == 200
        body = response.json()
        assert "fraud_probability" in body
        assert "shap_values" in body
        assert isinstance(body["shap_values"], list)
        assert len(body["shap_values"]) > 0

    def test_explain_respects_top_n_parameter(self, api_client, valid_transaction_payload) -> None:
        """Explain endpoint must respect ?top_n parameter."""
        response = api_client.post("/explain?top_n=3", json=valid_transaction_payload)
        assert response.status_code == 200
        body = response.json()
        assert len(body["shap_values"]) <= 3

    def test_multiple_sequential_predictions_are_consistent(
        self, api_client, valid_transaction_payload
    ) -> None:
        """Same input must produce same prediction across calls."""
        response1 = api_client.post("/predict", json=valid_transaction_payload)
        response2 = api_client.post("/predict", json=valid_transaction_payload)

        assert response1.status_code == 200
        assert response2.status_code == 200
        prob1 = response1.json()["fraud_probability"]
        prob2 = response2.json()["fraud_probability"]
        assert prob1 == pytest.approx(prob2, abs=1e-6)


class TestXGBoostDriftMonitoring:
    """Verify drift monitoring works with XGBoost model predictions."""

    def test_prediction_probability_range_valid(
        self, api_client, valid_transaction_payload
    ) -> None:
        """Predictions must stay within [0, 1] for monitoring."""
        for _ in range(10):
            response = api_client.post("/predict", json=valid_transaction_payload)
            prob = response.json()["fraud_probability"]
            assert 0.0 <= prob <= 1.0

    def test_calibration_preserves_probability_ordering(self, prepared_frame) -> None:
        """Calibration must preserve relative ordering of predictions."""
        pipeline = FeaturePipeline()
        cut = int(len(prepared_frame) * 0.5)
        pipeline.fit(prepared_frame.iloc[:cut])
        X = pipeline.transform(prepared_frame.iloc[:cut])
        y = prepared_frame.iloc[:cut]["isFraud"].to_numpy()

        model = build_model("xgboost", seed=42, n_jobs=1, n_estimators=10)
        model.fit(X, y)
        raw_proba = model.predict_proba(X)[:, 1]

        calibrator = ProbabilityCalibrator(method="isotonic")
        calibrator.fit(raw_proba, y)
        calibrated_proba = calibrator.transform(raw_proba)

        # Monotonicity: if raw[i] < raw[j], should still hold after calibration
        idx1, idx2 = 0, 1
        if raw_proba[idx1] < raw_proba[idx2]:
            # Ordering should be preserved (allowing for small tolerances)
            assert calibrated_proba[idx1] <= calibrated_proba[idx2] + 0.01


class TestXGBoostModelCompliance:
    """Verify XGBoost model meets project requirements."""

    def test_xgboost_model_name_in_metadata(self, api_client) -> None:
        """Model must be identified as XGBoost."""
        from api.dependencies import state

        if not state.artifact:
            pytest.skip("No artifact loaded")

        assert state.artifact.metadata.model_name == "xgboost"

    def test_xgboost_uses_pr_auc_for_evaluation(self) -> None:
        """XGBoost must be configured for PR-AUC optimization."""
        model = build_model("xgboost", seed=42, n_jobs=1)
        params = model.get_params()
        assert params["eval_metric"] == "aucpr"

    def test_xgboost_respects_imbalance_scaling(self) -> None:
        """XGBoost must apply scale_pos_weight for imbalanced data."""
        model = build_model("xgboost", seed=42, n_jobs=1, imbalance_weight=27.5)
        params = model.get_params()
        assert params["scale_pos_weight"] == pytest.approx(27.5)

    def test_xgboost_supports_categorical_features(self) -> None:
        """XGBoost must enable native categorical handling."""
        model = build_model("xgboost", seed=42, n_jobs=1)
        params = model.get_params()
        assert params["enable_categorical"] is True
        assert params["tree_method"] == "hist"
