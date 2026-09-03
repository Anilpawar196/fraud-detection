"""Tests for the estimator factories and the shared early-stopping path.

Both boosted models are exercised, because the project compares them and a
comparison is only meaningful if each is configured the way it is claimed to be.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.estimators import (
    BOOSTED_MODELS,
    REQUIRES_DENSE_IMPUTED,
    build_model,
    fit_with_early_stopping,
    prepare_frame_for_model,
    scale_pos_weight,
)


class TestScalePosWeight:
    def test_is_the_negative_to_positive_ratio(self) -> None:
        y = np.array([0] * 90 + [1] * 10)
        assert scale_pos_weight(y) == pytest.approx(9.0)

    def test_rejects_a_single_class(self) -> None:
        with pytest.raises(ValueError, match="No positive samples"):
            scale_pos_weight(np.zeros(10))


class TestBuildModel:
    @pytest.mark.parametrize(
        "name", ["logistic_regression", "random_forest", "lightgbm", "xgboost"]
    )
    def test_every_declared_model_builds(self, name) -> None:
        assert build_model(name, seed=1, n_jobs=1) is not None

    def test_unknown_model_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("not_a_real_model", seed=1)  # type: ignore[arg-type]

    def test_boosters_take_native_categoricals_and_nan(self) -> None:
        """No booster may be handed an imputed dense matrix — routing NaN
        natively is why they are preferred on this dataset."""
        assert {"lightgbm", "xgboost", "catboost"} == BOOSTED_MODELS
        assert not (BOOSTED_MODELS & REQUIRES_DENSE_IMPUTED)

    def test_xgboost_is_configured_for_categoricals_and_pr_auc(self) -> None:
        params = build_model("xgboost", seed=1, n_jobs=1).get_params()
        assert params["enable_categorical"] is True
        assert params["tree_method"] == "hist"
        # PR-AUC is the project's selection metric; early stopping must optimise it.
        assert params["eval_metric"] == "aucpr"

    def test_imbalance_weight_reaches_both_boosters(self) -> None:
        lgbm = build_model("lightgbm", seed=1, n_jobs=1, imbalance_weight=27.6).get_params()
        xgboost = build_model("xgboost", seed=1, n_jobs=1, imbalance_weight=27.6).get_params()
        assert lgbm["scale_pos_weight"] == pytest.approx(27.6)
        assert xgboost["scale_pos_weight"] == pytest.approx(27.6)

    def test_overrides_win_over_defaults(self) -> None:
        assert build_model("xgboost", seed=1, max_depth=3).get_params()["max_depth"] == 3


class TestEarlyStopping:
    @pytest.mark.parametrize("name", ["lightgbm", "xgboost"])
    def test_both_boosters_fit_and_report_an_iteration(self, name, prepared_frame) -> None:
        """The two libraries express early stopping differently; the helper hides
        that, and both must return a usable best-iteration count."""
        from src.features.pipeline import FeaturePipeline

        cut = int(len(prepared_frame) * 0.7)
        pipeline = FeaturePipeline()
        pipeline.fit(prepared_frame.iloc[:cut])
        X = pipeline.transform(prepared_frame.iloc[:cut])
        y = prepared_frame.iloc[:cut]["isFraud"].to_numpy()
        split = int(len(X) * 0.8)

        model = build_model(name, seed=1, n_jobs=1, n_estimators=30)
        best = fit_with_early_stopping(
            model,
            name,
            X.iloc[:split],
            y[:split],
            X.iloc[split:],
            y[split:],
            pipeline.categorical_features,
        )
        assert isinstance(best, int) and best >= 0
        proba = model.predict_proba(X.iloc[split:])[:, 1]
        assert ((proba >= 0.0) & (proba <= 1.0)).all()

    def test_non_boosted_model_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not support early stopping"):
            fit_with_early_stopping(None, "random_forest", None, None, None, None, [])

    def test_catboost_is_configured_for_pr_auc_on_cpu(self) -> None:
        params = build_model("catboost", seed=1, n_jobs=1).get_params()
        assert params["eval_metric"] == "PRAUC"
        assert params["depth"] == 8
        # The reference solution used a GPU; this project has none, so the
        # tree budget matches the other boosters rather than their 5000.
        assert params["iterations"] == 2000

    def test_catboost_categorical_fill_is_applied_and_others_untouched(self) -> None:
        """CatBoost rejects NaN in categoricals; the fill must be identical at
        fit and predict, and must not alter the other models' input."""
        frame = pd.DataFrame({"cat": pd.Series(["a", None], dtype="object"), "num": [1.0, np.nan]})
        filled = prepare_frame_for_model(frame, "catboost", ["cat"])
        assert filled["cat"].tolist() == ["a", "__missing__"]
        assert np.isnan(filled["num"].iloc[1]), "numeric NaN must stay for native routing"
        for other in ("lightgbm", "xgboost"):
            assert prepare_frame_for_model(frame, other, ["cat"]) is frame


class TestOptionalDependencies:
    """CatBoost must stay optional.

    It is a ~329 MB benchmarking-only dependency that lost the model comparison,
    so it is in requirements-dev.txt rather than requirements.txt and the serving
    image never installs it. A module-scope import would therefore break every
    import of src.models.estimators in production and in CI -- which is exactly
    what happened once.
    """

    def test_estimators_imports_without_catboost(self) -> None:
        code = textwrap.dedent(
            """
            import builtins, sys
            _real = builtins.__import__

            def _blocked(name, *args, **kwargs):
                if name == "catboost" or name.startswith("catboost."):
                    raise ImportError("simulated: catboost not installed")
                return _real(name, *args, **kwargs)

            builtins.__import__ = _blocked
            from src.models.estimators import build_model
            assert build_model("lightgbm", seed=1, n_jobs=1) is not None
            try:
                build_model("catboost", seed=1)
            except ImportError as exc:
                assert "requirements-dev.txt" in str(exc)
            else:
                raise AssertionError("expected a helpful ImportError")
            print("OK")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout
