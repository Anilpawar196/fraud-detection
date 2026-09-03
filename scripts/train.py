"""Train, compare, tune, calibrate, explain, and register.

Separation of concerns, enforced by the order of operations:

* **Cross-validation** (purged forward-chaining, encoders refit per fold) decides
  which model family wins. Holdout is not read.
* **Tuning** optimises the winner against CV PR-AUC. Holdout is not read.
* **Calibration + threshold** are fitted on the latest CV fold's validation
  predictions. Holdout is not read.
* **Holdout** is scored exactly once, at the end, by one model, using the
  threshold chosen above. That single number is the honest performance estimate.

Everything is tracked in MLflow. Usage:

    python scripts/train.py                       # full run with tuning
    python scripts/train.py --no-tune             # baselines only
    python scripts/train.py --models lightgbm     # single model
    python scripts/train.py --skip-holdout        # iterate without burning it
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import TARGET  # noqa: E402
from src.data.splitting import load_folds, load_split  # noqa: E402
from src.evaluation.calibration import ProbabilityCalibrator  # noqa: E402
from src.evaluation.compare import (  # noqa: E402
    build_comparison,
    cv_result_row,
    metrics_row,
    save_comparison,
)
from src.evaluation.metrics import (  # noqa: E402
    calibration_table,
    compute_metrics,
    find_best_threshold,
)
from src.explainability.shap_explainer import (  # noqa: E402
    ShapExplainer,
    sample_for_explanation,
)
from src.models.artifact import (  # noqa: E402
    ArtifactMetadata,
    ModelArtifact,
    library_versions,
    utc_now_iso,
)
from src.models.training import CVResult, train_cv, train_final  # noqa: E402
from src.models.tuning import SEARCH_SPACES, tune_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402
from src.utils.paths import (  # noqa: E402
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    ensure_dir,
)
from src.utils.seed import set_seed  # noqa: E402

logger = logging.getLogger("train")

DEFAULT_MODELS = ("logistic_regression", "random_forest", "lightgbm", "xgboost")

#: Row cap for models needing a dense imputed matrix. Logistic Regression and
#: Random Forest cannot take NaN, so they are fitted on a dense one-hot matrix
#: roughly 970 columns wide (after identical missingness indicators are
#: collapsed). At 133,979 rows that is ~520 MB, and a contiguous block that large
#: failed to allocate on this machine even with ~4.9 GB free, because the address
#: space fragments across folds. 100,000 rows keeps it near 390 MB.
#:
#: The subsample preserves every positive and downsamples negatives, and both the
#: row count and the `subsampled` flag are recorded in model_comparison.csv, so
#: the comparison is visibly unequal rather than silently so. LightGBM and
#: XGBoost use every row, since they consume NaN and categoricals natively.
DENSE_MODEL_MAX_ROWS = 100_000


def _raw_input_columns() -> list[str]:
    """Raw dataset columns the feature pipeline reads, from the interim schema.

    Read from the Parquet footer rather than hardcoded, so the API's accepted
    input can never drift away from what the pipeline actually consumes.
    """
    import pyarrow.parquet as pq

    from src.data.loading import joined_path

    schema = pq.read_schema(joined_path("train"))
    return [name for name in schema.names if name != TARGET]


def _load_prepared() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load the prepared modelling and holdout frames plus split metadata."""
    modelling_path = PROCESSED_DIR / "modelling_prepared.parquet"
    holdout_path = PROCESSED_DIR / "holdout_prepared.parquet"
    for path in (modelling_path, holdout_path):
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing — run scripts/build_dataset.py first")

    modelling = pd.read_parquet(modelling_path)
    holdout = pd.read_parquet(holdout_path)
    _, metadata = load_split()
    logger.info(
        "Loaded modelling %s (fraud %.4f%%) | holdout %s (fraud %.4f%%)",
        modelling.shape,
        modelling[TARGET].mean() * 100,
        holdout.shape,
        holdout[TARGET].mean() * 100,
    )
    return modelling, holdout, metadata


def _mlflow_setup(experiment: str):
    """Configure MLflow, defaulting to a local file store."""
    import mlflow

    import os

    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        uri = (Path.cwd() / "mlruns").as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    logger.info("MLflow tracking URI: %s | experiment: %s", uri, experiment)
    return mlflow


def _log_cv_result(mlflow, result: CVResult) -> None:
    """Log one model's CV metrics under a nested run."""
    with mlflow.start_run(run_name=f"cv_{result.model_name}", nested=True):
        mlflow.log_param("model", result.model_name)
        mlflow.log_param("cv_scheme", "purged_forward_chaining")
        mlflow.log_param("n_folds", len(result.fold_results))
        mlflow.log_param("subsampled_training_rows", result.subsampled)
        for metric in ("pr_auc", "roc_auc", "precision", "recall", "f1", "brier", "pr_auc_lift"):
            mlflow.log_metric(f"cv_{metric}", result.mean_metric(metric))
            mlflow.log_metric(f"cv_{metric}_std", result.std_metric(metric))
        mlflow.log_metric("cv_train_seconds", result.total_train_seconds)
        for fold in result.fold_results:
            mlflow.log_metric("fold_pr_auc", fold.metrics.pr_auc, step=fold.fold)
            mlflow.log_metric("fold_roc_auc", fold.metrics.roc_auc, step=fold.fold)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate fraud models.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--no-tune", action="store_true", help="Skip Optuna tuning.")
    parser.add_argument("--trials", type=int, default=None, help="Override Optuna trial count.")
    parser.add_argument(
        "--skip-holdout",
        action="store_true",
        help="Do not score the holdout (use while iterating).",
    )
    parser.add_argument(
        "--random-cv-control",
        action="store_true",
        help="Also run random stratified CV to quantify how optimistic it is.",
    )
    parser.add_argument(
        "--oof-npz",
        type=Path,
        default=None,
        help=(
            "Load out-of-fold probabilities from a .npy written by "
            "scripts/run_ablation.py --save-oof, instead of recomputing CV. Only the "
            "last fold's validation slice is used, for calibration and threshold."
        ),
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help=(
            "Skip the model-comparison CV loop and go straight to tuning, final fit "
            "and holdout. Use when fold metrics were already produced per-process by "
            "scripts/run_ablation.py; the five-fold loop holds more memory at once "
            "than some machines can give it."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Alternate config YAML (used for feature ablations).",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    set_seed(config.train.seed)
    mlflow = _mlflow_setup("fraud-detection")

    modelling, holdout, split_metadata = _load_prepared()
    temporal_folds = load_folds("temporal")
    logger.info("Loaded %d temporal folds", len(temporal_folds))

    rows: list[dict[str, object]] = []
    cv_results: dict[str, CVResult] = {}

    with mlflow.start_run(run_name=f"training_{utc_now_iso()}") as parent_run:
        mlflow.set_tag("dataset", "ieee-cis-fraud-detection")
        mlflow.set_tag("cv_scheme", "purged_forward_chaining")
        mlflow.log_params(
            {
                "holdout_fraction": config.split.holdout_fraction,
                "n_cv_folds": config.split.n_cv_folds,
                "purge_days": config.split.purge_days,
                "seed": config.train.seed,
                "keep_all_v_columns": config.features.keep_all_v_columns,
                "velocity_windows_hours": str(config.features.velocity_windows_hours),
                "modelling_rows": len(modelling),
                "holdout_rows": len(holdout),
                "prepared_columns": modelling.shape[1],
            }
        )
        mlflow.log_dict(split_metadata, "split_metadata.json")

        # --- baselines: identical folds for every model ---------------------
        for model_name in [] if args.skip_cv else args.models:
            logger.info("=" * 70)
            logger.info("Cross-validating %s", model_name)
            result = train_cv(
                modelling,
                temporal_folds,
                model_name,  # type: ignore[arg-type]
                seed=config.train.seed,
                n_jobs=config.train.n_jobs,
                max_dense_rows=DENSE_MODEL_MAX_ROWS,
                exclude_feature_suffixes=config.features.exclude_feature_suffixes,
            )
            cv_results[model_name] = result
            rows.append(cv_result_row(result))
            _log_cv_result(mlflow, result)

        # --- optional control: how optimistic is random CV? -----------------
        if args.random_cv_control and "lightgbm" in cv_results:
            logger.info("=" * 70)
            logger.info("Control experiment: random stratified CV (never used for selection)")
            random_result = train_cv(
                modelling,
                load_folds("random"),
                "lightgbm",
                seed=config.train.seed,
                n_jobs=config.train.n_jobs,
                exclude_feature_suffixes=config.features.exclude_feature_suffixes,
            )
            random_row = cv_result_row(random_result)
            random_row["model"] = "lightgbm"
            random_row["evaluation"] = "cv_random_control"
            rows.append(random_row)
            optimism = random_result.mean_metric("pr_auc") - cv_results["lightgbm"].mean_metric(
                "pr_auc"
            )
            mlflow.log_metric("random_cv_optimism_pr_auc", optimism)
            logger.info(
                "Random CV PR-AUC %.4f vs temporal %.4f -> optimism %+.4f",
                random_result.mean_metric("pr_auc"),
                cv_results["lightgbm"].mean_metric("pr_auc"),
                optimism,
            )

        # --- select winner on CV PR-AUC -------------------------------------
        if args.skip_cv:
            best_name = args.models[0]
            logger.info("Skipped CV; taking %s as the model to tune and fit", best_name)
        else:
            best_name = max(cv_results, key=lambda name: cv_results[name].mean_metric("pr_auc"))
            logger.info(
                "Selected %s on CV PR-AUC %.4f",
                best_name,
                cv_results[best_name].mean_metric("pr_auc"),
            )
        mlflow.log_param("selected_model", best_name)

        # --- tuning ---------------------------------------------------------
        best_params: dict[str, object] = {}
        if not args.no_tune and best_name in SEARCH_SPACES:
            tuning = tune_model(
                modelling,
                temporal_folds,
                model_name=best_name,
                n_trials=args.trials or config.train.optuna_trials,
                timeout_seconds=config.train.optuna_timeout_seconds,
                seed=config.train.seed,
                n_jobs=config.train.n_jobs,
            )
            best_params = tuning.best_params
            trials_path = ensure_dir(REPORTS_DIR) / "optuna_trials.csv"
            tuning.to_frame().to_csv(trials_path, index=False)
            mlflow.log_artifact(str(trials_path))
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("tuning_best_cv_pr_auc", tuning.best_pr_auc)
            mlflow.log_metric("tuning_seconds", tuning.total_seconds)

            tuned = train_cv(
                modelling,
                temporal_folds,
                best_name,
                seed=config.train.seed,
                n_jobs=config.train.n_jobs,
                params=best_params,
                exclude_feature_suffixes=config.features.exclude_feature_suffixes,
            )
            tuned_row = cv_result_row(tuned)
            tuned_row["model"] = f"{best_name}_tuned"
            rows.append(tuned_row)
            _log_cv_result(mlflow, tuned)
            baseline_pr_auc = (
                cv_results[best_name].mean_metric("pr_auc") if best_name in cv_results else None
            )
            if baseline_pr_auc is None or tuned.mean_metric("pr_auc") > baseline_pr_auc:
                cv_results[f"{best_name}_tuned"] = tuned
                if baseline_pr_auc is None:
                    logger.info(
                        "Tuned CV PR-AUC %.4f (no in-run baseline; CV was skipped)",
                        tuned.mean_metric("pr_auc"),
                    )
                else:
                    logger.info(
                        "Tuning improved CV PR-AUC %.4f -> %.4f",
                        baseline_pr_auc,
                        tuned.mean_metric("pr_auc"),
                    )
            else:
                logger.info(
                    "Tuning did not improve CV PR-AUC (%.4f vs %.4f) — keeping baseline params",
                    tuned.mean_metric("pr_auc"),
                    baseline_pr_auc,
                )
                best_params = {}

        # --- calibration + threshold from the last fold ---------------------
        last_train_idx, last_val_idx = temporal_folds[-1]
        if args.oof_npz is not None:
            oof = np.load(args.oof_npz)
            if len(oof) != len(modelling):
                raise SystemExit(f"{args.oof_npz} has {len(oof)} rows, expected {len(modelling)}")
            if np.isnan(oof[last_val_idx]).any():
                raise SystemExit(
                    f"{args.oof_npz} has no predictions for the last fold's validation "
                    "rows — it must come from `run_ablation.py --fold 4 --save-oof`."
                )
            logger.info("Loaded out-of-fold probabilities from %s", args.oof_npz)
        else:
            selected_cv = cv_results.get(f"{best_name}_tuned") or cv_results.get(best_name)
            if selected_cv is None:
                raise SystemExit(
                    "No CV result to calibrate on. Either drop --skip-cv, or pass "
                    "--oof-npz with predictions from run_ablation.py --save-oof."
                )
            oof = selected_cv.oof_predictions
            if oof is None:
                raise RuntimeError("Selected CV result has no out-of-fold predictions")
        validation_probabilities = oof[last_val_idx]
        validation_targets = modelling[TARGET].to_numpy()[last_val_idx]

        calibrator: ProbabilityCalibrator | None = None
        if config.train.calibrate:
            candidate = ProbabilityCalibrator().fit(validation_targets, validation_probabilities)
            if candidate.improved:
                calibrator = candidate
                validation_probabilities = candidate.transform(validation_probabilities)
                mlflow.log_metric("calibration_ece_before", candidate.ece_before or 0.0)
                mlflow.log_metric("calibration_ece_after", candidate.ece_after or 0.0)
            else:
                logger.warning(
                    "Calibration did not reduce ECE (%.5f -> %.5f) — serving uncalibrated scores",
                    candidate.ece_before or float("nan"),
                    candidate.ece_after or float("nan"),
                )

        threshold = find_best_threshold(validation_targets, validation_probabilities)
        validation_metrics = compute_metrics(
            validation_targets, validation_probabilities, threshold=threshold
        )
        logger.info("Validation (last fold): %s", validation_metrics.summary())
        mlflow.log_metrics(
            {
                f"val_{k}": v
                for k, v in validation_metrics.to_flat_dict().items()
                if isinstance(v, int | float)
            }
        )
        rows.append(
            metrics_row(
                best_name,
                "validation_last_fold",
                validation_metrics,
                n_train_rows=int(last_train_idx.size),
            )
        )

        # --- final fit on the whole modelling period ------------------------
        logger.info("=" * 70)
        final_name = best_name
        started = time.perf_counter()
        model, pipeline, linear_prep, best_iteration = train_final(
            modelling,
            final_name,  # type: ignore[arg-type]
            seed=config.train.seed,
            n_jobs=config.train.n_jobs,
            params=best_params or None,
            max_dense_rows=DENSE_MODEL_MAX_ROWS,
            exclude_feature_suffixes=config.features.exclude_feature_suffixes,
        )
        final_train_seconds = time.perf_counter() - started
        mlflow.log_metric("final_train_seconds", final_train_seconds)
        if best_iteration:
            mlflow.log_metric("final_best_iteration", best_iteration)

        artifact = ModelArtifact(
            model=model,
            feature_pipeline=pipeline,
            calibrator=calibrator,
            linear_preprocessor=linear_prep,
            decision_threshold=threshold,
            metadata=ArtifactMetadata(
                model_name=final_name,
                trained_at=utc_now_iso(),
                seed=config.train.seed,
                n_features=len(pipeline.feature_names),
                n_train_rows=len(modelling),
                dataset_rows_total=len(modelling) + len(holdout),
                holdout_cut_dt=int(split_metadata["holdout_cut_dt"]),
                feature_config={
                    "keep_all_v_columns": config.features.keep_all_v_columns,
                    "velocity_windows_hours": list(config.features.velocity_windows_hours),
                    "anchor_d_columns": config.features.anchor_d_columns,
                },
                hyperparameters=dict(best_params or {}),
                validation_metrics=validation_metrics.to_flat_dict(),
                calibrated=calibrator is not None,
                library_versions=library_versions(),
                raw_input_columns=_raw_input_columns(),
            ),
        )

        # --- SHAP -----------------------------------------------------------
        shap_importance_path = None
        if final_name in SEARCH_SPACES:
            logger.info("Computing SHAP explanations")
            # Sample rows *before* transforming: transforming all 472k rows to
            # then keep 5,000 wastes several minutes and ~1 GB for no gain.
            X_sample = pipeline.transform(
                sample_for_explanation(modelling, n=5000, seed=config.train.seed)
            )
            explainer = ShapExplainer(model, pipeline.feature_names)
            importance = explainer.global_importance(X_sample)
            shap_importance_path = ensure_dir(REPORTS_DIR) / "shap_global_importance.csv"
            importance.to_csv(shap_importance_path, index=False)
            summary_png = explainer.summary_plot(X_sample, FIGURES_DIR / "shap_summary.png")
            bar_png = explainer.bar_plot(importance, FIGURES_DIR / "shap_importance.png")
            for path in (shap_importance_path, summary_png, bar_png):
                mlflow.log_artifact(str(path))
            logger.info(
                "Top 10 features by mean |SHAP|: %s",
                ", ".join(importance.head(10)["feature"].tolist()),
            )
            example = explainer.explain_row(X_sample, row=0, top_n=8)
            mlflow.log_dict(
                {"example_explanation": [c.to_dict() for c in example]}, "example_explanation.json"
            )

        # --- holdout: scored exactly once -----------------------------------
        if not args.skip_holdout:
            logger.info("=" * 70)
            logger.info("Scoring the untouched holdout ONCE")
            holdout_probabilities = artifact.predict_proba(holdout)
            holdout_metrics = compute_metrics(
                holdout[TARGET].to_numpy(), holdout_probabilities, threshold=threshold
            )
            logger.info("HOLDOUT: %s", holdout_metrics.summary())
            mlflow.log_metrics(
                {
                    f"holdout_{k}": v
                    for k, v in holdout_metrics.to_flat_dict().items()
                    if isinstance(v, int | float)
                }
            )
            artifact.metadata.holdout_metrics = holdout_metrics.to_flat_dict()
            rows.append(
                metrics_row(
                    final_name,
                    "holdout_final",
                    holdout_metrics,
                    n_features=len(pipeline.feature_names),
                    n_train_rows=len(modelling),
                    training_time_seconds=final_train_seconds,
                )
            )
            calibration_path = ensure_dir(REPORTS_DIR) / "holdout_calibration.json"
            calibration_path.write_text(
                json.dumps(
                    calibration_table(holdout[TARGET].to_numpy(), holdout_probabilities), indent=2
                ),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(calibration_path))

        # --- persist + register ---------------------------------------------
        artifact_path = artifact.save(ensure_dir(MODELS_DIR))
        # Only the metadata sidecar is logged, not the bundle itself. The pickle is
        # ~52 MB and MLflow keeps one copy per run, which exhausted the disk mid-run
        # on this machine. The metadata carries the provenance that matters
        # (versions, hyperparameters, metrics, feature config); the bundle lives at
        # models/model_artifact.pkl and is reproducible from a seeded run.
        mlflow.log_artifact(str(MODELS_DIR / "model_metadata.json"), artifact_path="model_artifact")
        mlflow.log_param("model_artifact_path", str(artifact_path.relative_to(MODELS_DIR.parent)))

        comparison = build_comparison(rows)
        comparison_path = save_comparison(comparison)
        mlflow.log_artifact(str(comparison_path))
        logger.info("\n%s", comparison.to_string(index=False))

        (ensure_dir(REPORTS_DIR) / "last_run.json").write_text(
            json.dumps(
                {
                    "run_id": parent_run.info.run_id,
                    "selected_model": final_name,
                    "threshold": threshold,
                    "calibrated": calibrator is not None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    logger.info("Training complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
