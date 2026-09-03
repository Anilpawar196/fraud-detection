# Reports

Everything in this directory is **generated output**, not source, and is
gitignored. The headline numbers are quoted inline in the root `README.md`, so
nothing here needs to be committed to read the results — but each artifact is
reproducible with one command.

| artifact | produced by |
|---|---|
| `dataset_audit.json`, `dataset_audit.md` | `python scripts/inspect_dataset.py` |
| `model_comparison.csv` | `python scripts/train.py` |
| `optuna_trials.csv` | `python scripts/train.py` (tuning enabled) |
| `shap_global_importance.csv` | `python scripts/train.py` |
| `holdout_calibration.json`, `last_run.json` | `python scripts/train.py` |
| `evaluation_holdout.json` + `figures/*.png` | `python scripts/evaluate.py --partition holdout` |
| `monitoring/feature_drift.csv` | `python scripts/monitor.py --current test` |
| `monitoring/monitoring_summary.json` | `python scripts/monitor.py --current test` |
| `*.log` | stdout redirects from the above |

## Why these are not committed

They are large (the dataset audit alone is ~484 KB), they change on every run, and
they are reproducible from the raw data plus a fixed seed. A repository that
tracks them accumulates diff noise carrying no information.

`model_comparison.csv` is the one worth noting: it is small and is the canonical
record of the model comparison, so its contents are reproduced verbatim in the
root `README.md` under *Model Comparison*. The result survives even though the
file does not.

Experiment tracking is separate — `scripts/train.py` writes params, metrics and
artifacts to a local `./mlruns` file store (also gitignored). Run `mlflow ui` from
the repository root to browse it; no MLflow server is required.
