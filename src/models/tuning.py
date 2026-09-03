"""Hyperparameter optimisation with Optuna.

Scope is deliberately bounded. Only the boosted model that won cross-validation
is tuned; spending compute on baselines whose job is to be a reference point
would not change any decision. LightGBM and XGBoost have mirrored search spaces,
so whichever wins is tuned on equal terms. The budget is capped by both trial count and
wall-clock timeout, and each trial is scored on a **subset of the later temporal
folds** rather than all of them: the early folds train on as little as 134k rows
and are the least representative of the deployed model's situation, so tuning
against them would optimise for the wrong regime.

The objective is mean PR-AUC across those folds. The holdout is never involved —
it is not read once during tuning.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import pandas as pd

from src.models.training import train_cv

logger = logging.getLogger(__name__)

# Optuna's own logging is verbose at INFO and duplicates ours.
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TrialRecord:
    """One completed trial, for logging to MLflow and reports/."""

    number: int
    params: dict[str, Any]
    pr_auc: float
    roc_auc: float
    train_seconds: float


@dataclass
class TuningResult:
    """Outcome of a tuning study."""

    best_params: dict[str, Any]
    best_pr_auc: float
    trials: list[TrialRecord] = field(default_factory=list)
    total_seconds: float = 0.0

    def to_frame(self) -> pd.DataFrame:
        """Trials as a DataFrame for ``reports/optuna_trials.csv``."""
        return pd.DataFrame(
            [
                {
                    "trial": t.number,
                    "pr_auc": t.pr_auc,
                    "roc_auc": t.roc_auc,
                    "train_seconds": round(t.train_seconds, 1),
                    **t.params,
                }
                for t in self.trials
            ]
        ).sort_values("pr_auc", ascending=False)


def suggest_lightgbm_params(trial: optuna.Trial) -> dict[str, Any]:
    """Search space for LightGBM on this dataset.

    Ranges are chosen around the measured characteristics rather than generic:
    ``min_child_samples`` stays high because a leaf at 3.5% prevalence needs
    enough rows to hold real positives, and ``feature_fraction`` stays low
    because 339 of ~528 features are the redundant V block.
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 300, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 0.9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
    }


def suggest_xgboost_params(trial: optuna.Trial) -> dict[str, Any]:
    """Search space for XGBoost, mirrored to the LightGBM one.

    Ranges correspond to their LightGBM counterparts so neither model is given a
    wider or better-centred space than the other — the comparison should reflect
    the algorithms, not the search.
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


SEARCH_SPACES = {
    "lightgbm": suggest_lightgbm_params,
    "xgboost": suggest_xgboost_params,
}


def tune_model(
    prepared: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    model_name: str = "lightgbm",
    n_trials: int = 25,
    timeout_seconds: int | None = 1800,
    seed: int = 42,
    n_jobs: int = -1,
    tuning_folds: int = 2,
) -> TuningResult:
    """Run a bounded Optuna study over one boosted model's hyperparameters.

    Args:
        prepared: Prepared modelling frame.
        model_name: Which booster to tune ("lightgbm" or "xgboost").
        folds: All temporal folds; only the last ``tuning_folds`` are used.
        n_trials: Maximum trials.
        timeout_seconds: Wall-clock cap; ``None`` for no cap.
        seed: Sampler seed, so the study is reproducible.
        n_jobs: Estimator parallelism.
        tuning_folds: How many of the latest folds to score each trial on.

    Returns:
        The best parameters and a record of every trial.
    """
    if model_name not in SEARCH_SPACES:
        raise ValueError(f"No search space defined for {model_name!r}")
    suggest = SEARCH_SPACES[model_name]

    selected_folds = folds[-tuning_folds:] if tuning_folds else folds
    logger.info(
        "Tuning %s: <=%d trials, timeout %ss, %d fold(s) per trial",
        model_name,
        n_trials,
        timeout_seconds,
        len(selected_folds),
    )
    records: list[TrialRecord] = []
    started = time.perf_counter()

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        trial_started = time.perf_counter()
        result = train_cv(
            prepared,
            selected_folds,
            model_name,
            seed=seed,
            n_jobs=n_jobs,
            params=params,
        )
        pr_auc = result.mean_metric("pr_auc")
        records.append(
            TrialRecord(
                number=trial.number,
                params=params,
                pr_auc=pr_auc,
                roc_auc=result.mean_metric("roc_auc"),
                train_seconds=time.perf_counter() - trial_started,
            )
        )
        logger.info("trial %d | PR-AUC %.4f | %s", trial.number, pr_auc, params)
        return pr_auc

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=f"{model_name}_pr_auc",
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, gc_after_trial=True)

    result = TuningResult(
        best_params=dict(study.best_params),
        best_pr_auc=float(study.best_value),
        trials=records,
        total_seconds=time.perf_counter() - started,
    )
    logger.info(
        "Tuning done: best PR-AUC %.4f in %.1fs over %d trials | %s",
        result.best_pr_auc,
        result.total_seconds,
        len(records),
        result.best_params,
    )
    return result


def tune_lightgbm(*args: Any, **kwargs: Any) -> TuningResult:
    """Backwards-compatible alias for :func:`tune_model` with LightGBM."""
    return tune_model(*args, model_name="lightgbm", **kwargs)
