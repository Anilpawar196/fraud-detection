"""Model factories.

Three families, each with a distinct job in the comparison:

* **Logistic Regression** — the floor. A regularized linear model on imputed,
  scaled, one-hot features. If a complex model cannot clearly beat this, the
  complexity is not earning its keep. It also reveals how much of the signal is
  simply linear.
* **Random Forest** — the bagging contrast. Non-linear and interaction-capable
  but, unlike LightGBM, it cannot route NaN structurally, so it must be given
  imputed inputs. That difference is itself informative about how much of
  LightGBM's advantage comes from native missing-value handling.
* **LightGBM** — the candidate for production. Native categorical support,
  native NaN routing, and the only one of the three that is fast enough at
  590,540 × ~550 to tune properly.

Imbalance is handled by **reweighting, not resampling**. SMOTE would interpolate
between fraud rows across a ~550-column space that is largely categorical and
43% missing in the V block; the interpolants would not be plausible
transactions. ``scale_pos_weight`` / ``class_weight`` leaves the data honest and
only changes the loss. Because reweighting distorts predicted probabilities, and
this project's API returns a probability that drives a risk band, calibration is
applied afterwards rather than treated as optional.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

ModelName = Literal["logistic_regression", "random_forest", "lightgbm", "xgboost", "catboost"]

#: Models that cannot accept NaN and therefore require the imputed matrix.
REQUIRES_DENSE_IMPUTED: frozenset[str] = frozenset({"logistic_regression", "random_forest"})

#: Models fitted with early stopping against an inner temporal slice.
BOOSTED_MODELS: frozenset[str] = frozenset({"lightgbm", "xgboost", "catboost"})

#: Rounds without improvement before early stopping halts a boosted model.
EARLY_STOPPING_ROUNDS = 100


def scale_pos_weight(y: np.ndarray) -> float:
    """Ratio of negatives to positives, for LightGBM's ``scale_pos_weight``."""
    positives = float(np.sum(y))
    if positives == 0:
        raise ValueError("No positive samples — cannot compute scale_pos_weight")
    return float((len(y) - positives) / positives)


def lightgbm_params(
    seed: int = 42,
    n_jobs: int = -1,
    imbalance_weight: float | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Baseline LightGBM parameters.

    Defaults chosen for this dataset rather than copied from a tutorial:

    * ``num_leaves`` 64 with ``min_child_samples`` 100 — at 3.5% prevalence a
      leaf needs enough rows to contain a meaningful number of positives, or the
      model fits noise in the tail.
    * ``feature_fraction`` 0.6 — 339 of the ~550 features are the highly
      redundant V block; subsampling columns decorrelates trees cheaply, which
      is the intended substitute for the correlation-clustering we chose not to
      do before establishing a baseline.
    * ``max_bin`` 127 rather than the default 255 — halves the histogram memory
      on a wide frame with negligible accuracy cost.
    """
    params: dict[str, Any] = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": -1,
        "min_child_samples": 100,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 1.0,
        "max_bin": 127,
        "n_estimators": 2000,
        "random_state": seed,
        "n_jobs": n_jobs,
        "verbose": -1,
    }
    if imbalance_weight is not None:
        params["scale_pos_weight"] = imbalance_weight
    params.update(overrides)
    return params


def make_lightgbm(
    seed: int = 42,
    n_jobs: int = -1,
    imbalance_weight: float | None = None,
    **overrides: Any,
) -> lgb.LGBMClassifier:
    """Build a LightGBM classifier."""
    return lgb.LGBMClassifier(
        **lightgbm_params(seed=seed, n_jobs=n_jobs, imbalance_weight=imbalance_weight, **overrides)
    )


def make_random_forest(
    seed: int = 42, n_jobs: int = -1, **overrides: Any
) -> RandomForestClassifier:
    """Build a Random Forest classifier.

    ``max_depth`` and ``min_samples_leaf`` are capped deliberately: unbounded
    trees on 470k rows produce a model of several GB, and this machine has
    single-digit GB of headroom. The cap is a documented constraint, not a
    tuning result.
    """
    params: dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 16,
        "min_samples_leaf": 50,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "random_state": seed,
        "n_jobs": n_jobs,
        "verbose": 0,
    }
    params.update(overrides)
    return RandomForestClassifier(**params)


def make_logistic_regression(seed: int = 42, **overrides: Any) -> LogisticRegression:
    """Build a regularized logistic regression.

    ``saga`` is avoided (too slow at this scale) in favour of ``lbfgs``, which
    handles the dense standardised matrix well. ``class_weight='balanced'``
    matches the reweighting strategy used for the tree models so the comparison
    isolates model capacity rather than imbalance handling.
    """
    params: dict[str, Any] = {
        "penalty": "l2",
        "C": 0.1,
        "solver": "lbfgs",
        "max_iter": 300,
        "class_weight": "balanced",
        "random_state": seed,
        "n_jobs": None,
    }
    params.update(overrides)
    return LogisticRegression(**params)


def xgboost_params(
    seed: int = 42,
    n_jobs: int = -1,
    imbalance_weight: float | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Baseline XGBoost parameters, mirrored to LightGBM's where they correspond.

    The two libraries name the same ideas differently, so the defaults are chosen
    to make the comparison about the algorithms rather than about who got a
    luckier configuration:

        LightGBM                XGBoost
        feature_fraction 0.6 -> colsample_bytree 0.6
        bagging_fraction 0.8 -> subsample 0.8
        min_child_samples    -> min_child_weight
        max_bin 127          -> max_bin 127

    ``tree_method="hist"`` with ``enable_categorical=True`` lets XGBoost consume
    the same pandas ``category`` columns LightGBM uses, so neither model gets a
    different feature representation. ``eval_metric="aucpr"`` matches the
    project's selection metric.
    """
    params: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "enable_categorical": True,
        "max_cat_to_onehot": 4,
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 5.0,
        "colsample_bytree": 0.6,
        "subsample": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "max_bin": 127,
        "n_estimators": 2000,
        "early_stopping_rounds": 100,
        "random_state": seed,
        "n_jobs": n_jobs,
        "verbosity": 0,
    }
    if imbalance_weight is not None:
        params["scale_pos_weight"] = imbalance_weight
    params.update(overrides)
    return params


def make_xgboost(
    seed: int = 42,
    n_jobs: int = -1,
    imbalance_weight: float | None = None,
    **overrides: Any,
) -> xgb.XGBClassifier:
    """Build an XGBoost classifier."""
    return xgb.XGBClassifier(
        **xgboost_params(seed=seed, n_jobs=n_jobs, imbalance_weight=imbalance_weight, **overrides)
    )


def catboost_params(
    seed: int = 42,
    n_jobs: int = -1,
    imbalance_weight: float | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Baseline CatBoost parameters, mirrored to the other boosters.

    CatBoost earns a place in the comparison because its ordered target
    statistics handle high-cardinality categoricals differently from LightGBM's
    split-based approach — and this dataset is largely high-cardinality
    categorical, so that difference should show up rather than being assumed.

    Depth 8 and learning rate 0.05 mirror the XGBoost defaults so the comparison
    stays about the algorithm. ``task_type`` is left on CPU: the reference
    solution this was drawn from used a GPU, which is not available here, so the
    tree count is bounded to 2000 like the others rather than their 5000.
    """
    params: dict[str, Any] = {
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "iterations": 2000,
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 3.0,
        "border_count": 127,
        "random_seed": seed,
        "thread_count": n_jobs if n_jobs and n_jobs > 0 else -1,
        "verbose": False,
        "allow_writing_files": False,
    }
    if imbalance_weight is not None:
        params["scale_pos_weight"] = imbalance_weight
    params.update(overrides)
    return params


def make_catboost(
    seed: int = 42,
    n_jobs: int = -1,
    imbalance_weight: float | None = None,
    **overrides: Any,
) -> Any:
    """CatBoost classifier with this project's defaults.

    CatBoost is imported here rather than at module scope because it is a
    329 MB dependency used only to benchmark a model that lost the comparison.
    The serving path never constructs one, so importing it eagerly would put
    a third of a gigabyte into the Docker image to support a code path
    production never takes. It lives in requirements-dev.txt accordingly.
    """
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "CatBoost is not installed. It is a benchmarking-only dependency: "
            "install it with `pip install -r requirements-dev.txt`."
        ) from exc

    return CatBoostClassifier(
        **catboost_params(seed=seed, n_jobs=n_jobs, imbalance_weight=imbalance_weight, **overrides)
    )


def prepare_frame_for_model(X: Any, model_name: str, categorical_features: list[str]) -> Any:
    """Apply model-specific input requirements to a feature frame.

    Only CatBoost needs anything: it rejects NaN inside categorical columns, so
    those are filled with an explicit sentinel level. This must be applied
    identically at fit and at predict time — filling only at fit would raise on
    the first validation row with a missing category, and filling differently
    would be a training/serving skew. Numeric NaN is left alone; CatBoost routes
    it natively, as LightGBM and XGBoost do.
    """
    if model_name != "catboost" or not categorical_features:
        return X
    X = X.copy()
    for column in categorical_features:
        if column in X.columns:
            X[column] = X[column].astype("string").fillna("__missing__")
    return X


def fit_with_early_stopping(
    model: Any,
    model_name: str,
    X_train: Any,
    y_train: Any,
    X_eval: Any,
    y_eval: Any,
    categorical_features: list[str],
) -> int:
    """Fit a boosted model against a held-back eval slice; return best iteration.

    LightGBM and XGBoost express early stopping differently — callbacks versus a
    constructor argument — so the difference is absorbed here rather than
    branching inside the training loop.
    """
    if model_name == "lightgbm":
        model.fit(
            X_train,
            y_train,
            eval_X=X_eval,
            eval_y=y_eval,
            eval_metric="average_precision",
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(0),
            ],
            categorical_feature=categorical_features,
        )
        return int(getattr(model, "best_iteration_", 0) or 0)

    if model_name == "xgboost":
        # early_stopping_rounds and eval_metric are set on the constructor in
        # the XGBoost 2.x+ sklearn API; categoricals come from the dtype.
        model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)
        return int(getattr(model, "best_iteration", 0) or 0)

    if model_name == "catboost":
        X_train = prepare_frame_for_model(X_train, model_name, categorical_features)
        X_eval = prepare_frame_for_model(X_eval, model_name, categorical_features)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_eval, y_eval),
            cat_features=categorical_features,
            use_best_model=True,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False,
        )
        return int(model.get_best_iteration() or 0)

    raise ValueError(f"{model_name!r} does not support early stopping here")


def build_model(
    name: ModelName,
    seed: int = 42,
    n_jobs: int = -1,
    imbalance_weight: float | None = None,
    **overrides: Any,
):
    """Dispatch to the requested model factory."""
    if name == "lightgbm":
        return make_lightgbm(
            seed=seed, n_jobs=n_jobs, imbalance_weight=imbalance_weight, **overrides
        )
    if name == "xgboost":
        return make_xgboost(
            seed=seed, n_jobs=n_jobs, imbalance_weight=imbalance_weight, **overrides
        )
    if name == "catboost":
        return make_catboost(
            seed=seed, n_jobs=n_jobs, imbalance_weight=imbalance_weight, **overrides
        )
    if name == "random_forest":
        return make_random_forest(seed=seed, n_jobs=n_jobs, **overrides)
    if name == "logistic_regression":
        return make_logistic_regression(seed=seed, **overrides)
    raise ValueError(f"Unknown model {name!r}")
