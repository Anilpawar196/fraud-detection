"""Cross-validated training with fold-internal feature fitting.

The important detail: **the feature encoders are refitted inside every fold.**
Frequency counts and per-entity amount baselines are population statistics, so
fitting them once on the whole modelling period and then cross-validating would
leak each fold's validation rows into its own training features. Refitting per
fold is a few groupbys of extra cost and is the difference between
cross-validation that means something and cross-validation that flatters.

Early stopping uses an **inner temporal split** carved from the fold's own
training rows, never the fold's validation set. Using the validation fold to
choose the iteration count and then reporting that fold's score would make the
reported number optimistic by exactly the amount early stopping helps.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.data.preprocessing import LinearPreprocessor
from src.data.schema import TARGET, TIME_COL
from src.evaluation.metrics import ClassificationMetrics, compute_metrics
from src.features.pipeline import FeaturePipeline
from src.models.estimators import (
    REQUIRES_DENSE_IMPUTED,
    ModelName,
    build_model,
    fit_with_early_stopping,
    prepare_frame_for_model,
    scale_pos_weight,
)

logger = logging.getLogger(__name__)

#: Fraction of a fold's training rows (latest by time) held back for early
#: stopping. Temporal, not random, for the same reason the outer split is.
EARLY_STOPPING_FRACTION = 0.10


@dataclass
class FoldResult:
    """Outcome of one cross-validation fold."""

    fold: int
    metrics: ClassificationMetrics
    n_train: int
    n_validation: int
    n_features: int
    train_seconds: float
    best_iteration: int | None = None


@dataclass
class CVResult:
    """Aggregate cross-validation outcome for one model."""

    model_name: str
    fold_results: list[FoldResult] = field(default_factory=list)
    oof_predictions: np.ndarray | None = None
    oof_mask: np.ndarray | None = None
    total_train_seconds: float = 0.0
    n_train_rows_used: int = 0
    subsampled: bool = False

    def mean_metric(self, name: str) -> float:
        values = [getattr(f.metrics, name) for f in self.fold_results]
        return float(np.mean(values)) if values else float("nan")

    def std_metric(self, name: str) -> float:
        values = [getattr(f.metrics, name) for f in self.fold_results]
        return float(np.std(values)) if values else float("nan")

    def summary(self) -> str:
        return (
            f"{self.model_name}: PR-AUC {self.mean_metric('pr_auc'):.4f} "
            f"(+/-{self.std_metric('pr_auc'):.4f}) | "
            f"ROC-AUC {self.mean_metric('roc_auc'):.4f} "
            f"(+/-{self.std_metric('roc_auc'):.4f}) | "
            f"{self.total_train_seconds:.1f}s over {len(self.fold_results)} folds"
        )


def _temporal_tail_split(df: pd.DataFrame, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Split positional indices into (head, tail) by time."""
    cut = int(len(df) * (1.0 - fraction))
    cut = max(1, min(cut, len(df) - 1))
    return np.arange(cut), np.arange(cut, len(df))


def _subsample_training_rows(
    df: pd.DataFrame, max_rows: int | None, seed: int
) -> tuple[pd.DataFrame, bool]:
    """Cap training rows while preserving the positive class.

    Used only for the models that require a dense imputed matrix. All positives
    are kept and negatives are downsampled, because at 3.5% prevalence throwing
    away positives would change the problem rather than merely shrink it. The
    resulting class balance is corrected by the estimator's ``class_weight``.
    """
    if max_rows is None or len(df) <= max_rows:
        return df, False

    positives = df.index[df[TARGET] == 1]
    negatives = df.index[df[TARGET] == 0]
    n_negatives = max(1, max_rows - len(positives))
    rng = np.random.default_rng(seed)
    sampled_negatives = rng.choice(negatives, size=min(n_negatives, len(negatives)), replace=False)
    keep = np.concatenate([positives.to_numpy(), sampled_negatives])
    subset = df.loc[keep].sort_values(TIME_COL, kind="mergesort")
    logger.info(
        "Subsampled dense-model training rows: %d -> %d (all %d positives kept)",
        len(df),
        len(subset),
        len(positives),
    )
    return subset, True


def _downcast_floats(df: pd.DataFrame) -> pd.DataFrame:
    """Narrow float64 feature columns to float32, in place, one column at a time.

    Gradient-boosted trees bin their inputs before choosing a split, so float32
    already carries more precision than the histogram can use — the extra eight
    bytes per value buy nothing. Halving the matrix is what lets the later,
    larger folds fit: fold 4 otherwise asks for a single 1.29 GiB contiguous
    block. Converting column-by-column keeps the peak to one column rather than
    duplicating the whole frame.
    """
    for column in df.columns:
        if df[column].dtype == "float64":
            df[column] = df[column].astype("float32")
    return df


def _fit_one(
    model_name: ModelName,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    seed: int,
    n_jobs: int,
    params: dict[str, Any] | None,
    max_dense_rows: int | None,
    exclude_feature_suffixes: tuple[str, ...] = (),
    force_subsample: bool = False,
) -> tuple[np.ndarray, int, int | None, bool, Any, FeaturePipeline, LinearPreprocessor | None]:
    """Fit one model on one fold and predict on its validation rows."""
    pipeline = FeaturePipeline(exclude_feature_suffixes=exclude_feature_suffixes)
    pipeline.fit(train_df)

    needs_dense = model_name in REQUIRES_DENSE_IMPUTED
    subsampled = False
    fit_df = train_df
    if needs_dense or force_subsample:
        fit_df, subsampled = _subsample_training_rows(train_df, max_dense_rows, seed)

    X_train = pipeline.transform(fit_df)
    y_train = fit_df[TARGET].to_numpy()
    X_validation = pipeline.transform(validation_df)

    linear_prep: LinearPreprocessor | None = None
    best_iteration: int | None = None

    if needs_dense:
        linear_prep = LinearPreprocessor().fit(X_train)
        X_train_matrix = linear_prep.transform(X_train)
        X_validation_matrix = linear_prep.transform(X_validation)
        del X_train, X_validation
        gc.collect()

        model = build_model(model_name, seed=seed, n_jobs=n_jobs, **(params or {}))
        model.fit(X_train_matrix, y_train)
        probabilities = model.predict_proba(X_validation_matrix)[:, 1]
        n_features = X_train_matrix.shape[1]
        del X_train_matrix, X_validation_matrix
    else:
        X_train = _downcast_floats(X_train)
        X_validation = _downcast_floats(X_validation)
        gc.collect()
        # Inner temporal split for early stopping — never the outer validation fold.
        head, tail = _temporal_tail_split(fit_df, EARLY_STOPPING_FRACTION)
        model = build_model(
            model_name,
            seed=seed,
            n_jobs=n_jobs,
            imbalance_weight=scale_pos_weight(y_train),
            **(params or {}),
        )
        best_iteration = fit_with_early_stopping(
            model,
            model_name,
            X_train.iloc[head],
            y_train[head],
            X_train.iloc[tail],
            y_train[tail],
            pipeline.categorical_features,
        )
        probabilities = model.predict_proba(
            prepare_frame_for_model(X_validation, model_name, pipeline.categorical_features)
        )[:, 1]
        n_features = X_train.shape[1]
        del X_train, X_validation

    gc.collect()
    return probabilities, n_features, best_iteration, subsampled, model, pipeline, linear_prep


def train_cv(
    prepared: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    model_name: ModelName,
    seed: int = 42,
    n_jobs: int = -1,
    params: dict[str, Any] | None = None,
    max_dense_rows: int | None = None,
    exclude_feature_suffixes: tuple[str, ...] = (),
    force_subsample: bool = False,
) -> CVResult:
    """Cross-validate one model over pre-computed folds.

    Args:
        prepared: Output of the prepare stage — causal features plus the target,
            time column and internal entity keys the encoders need.
        folds: Positional ``(train_idx, validation_idx)`` pairs. Identical folds
            are used for every model so comparisons are meaningful.
        model_name: Which estimator to fit.
        seed: Random seed.
        n_jobs: Parallelism for the estimator.
        params: Hyperparameter overrides.
        max_dense_rows: Row cap for models requiring a dense imputed matrix.
        exclude_feature_suffixes: Feature suffixes to drop, for ablations. The
            encoders are still fitted per fold, so an ablation stays leakage-safe.
        force_subsample: Apply the dense-model row cap to a model that does not
            need it. Used only to run the boosters at the baselines' row count,
            which removes training-set size as a confound in the comparison.

    Returns:
        A :class:`CVResult` with per-fold metrics and out-of-fold predictions.
    """
    result = CVResult(model_name=model_name)
    oof = np.full(len(prepared), np.nan, dtype="float64")

    for i, (train_idx, validation_idx) in enumerate(folds):
        started = time.perf_counter()
        train_df = prepared.iloc[train_idx]
        validation_df = prepared.iloc[validation_idx]

        (
            probabilities,
            n_features,
            best_iteration,
            subsampled,
            _model,
            _pipeline,
            _prep,
        ) = _fit_one(
            model_name,
            train_df,
            validation_df,
            seed=seed,
            n_jobs=n_jobs,
            params=params,
            max_dense_rows=max_dense_rows,
            exclude_feature_suffixes=exclude_feature_suffixes,
            force_subsample=force_subsample,
        )
        elapsed = time.perf_counter() - started

        oof[validation_idx] = probabilities
        metrics = compute_metrics(validation_df[TARGET].to_numpy(), probabilities)
        result.fold_results.append(
            FoldResult(
                fold=i,
                metrics=metrics,
                n_train=int(train_idx.size),
                n_validation=int(validation_idx.size),
                n_features=n_features,
                train_seconds=elapsed,
                best_iteration=best_iteration,
            )
        )
        result.total_train_seconds += elapsed
        result.n_train_rows_used = int(train_idx.size)
        result.subsampled = result.subsampled or subsampled
        logger.info(
            "[%s] fold %d | %s | %.1fs%s",
            model_name,
            i,
            metrics.summary(),
            elapsed,
            f" | best_iter {best_iteration}" if best_iteration else "",
        )
        del train_df, validation_df, probabilities
        gc.collect()

    result.oof_predictions = oof
    result.oof_mask = ~np.isnan(oof)
    logger.info(result.summary())
    return result


def train_final(
    train_df: pd.DataFrame,
    model_name: ModelName,
    seed: int = 42,
    n_jobs: int = -1,
    params: dict[str, Any] | None = None,
    max_dense_rows: int | None = None,
    exclude_feature_suffixes: tuple[str, ...] = (),
) -> tuple[Any, FeaturePipeline, LinearPreprocessor | None, int | None]:
    """Fit the final model on the full modelling period.

    Early stopping still uses an inner temporal tail of the training data, so
    the holdout remains completely untouched until the single final evaluation.
    """
    pipeline = FeaturePipeline(exclude_feature_suffixes=exclude_feature_suffixes)
    pipeline.fit(train_df)

    needs_dense = model_name in REQUIRES_DENSE_IMPUTED
    fit_df = train_df
    if needs_dense:
        fit_df, _ = _subsample_training_rows(train_df, max_dense_rows, seed)

    X = pipeline.transform(fit_df)
    y = fit_df[TARGET].to_numpy()
    linear_prep: LinearPreprocessor | None = None
    best_iteration: int | None = None

    if needs_dense:
        linear_prep = LinearPreprocessor().fit(X)
        matrix = linear_prep.transform(X)
        del X
        gc.collect()
        model = build_model(model_name, seed=seed, n_jobs=n_jobs, **(params or {}))
        model.fit(matrix, y)
        del matrix
    else:
        head, tail = _temporal_tail_split(fit_df, EARLY_STOPPING_FRACTION)
        model = build_model(
            model_name,
            seed=seed,
            n_jobs=n_jobs,
            imbalance_weight=scale_pos_weight(y),
            **(params or {}),
        )
        best_iteration = fit_with_early_stopping(
            model,
            model_name,
            X.iloc[head],
            y[head],
            X.iloc[tail],
            y[tail],
            pipeline.categorical_features,
        )
        del X

    gc.collect()
    logger.info("Final %s trained on %d rows", model_name, len(fit_df))
    return model, pipeline, linear_prep, best_iteration


def predict(
    model: Any,
    pipeline: FeaturePipeline,
    linear_prep: LinearPreprocessor | None,
    df: pd.DataFrame,
) -> np.ndarray:
    """Score a prepared frame with a fitted model + pipeline."""
    X = pipeline.transform(df)
    if linear_prep is not None:
        return model.predict_proba(linear_prep.transform(X))[:, 1]
    return model.predict_proba(X)[:, 1]
