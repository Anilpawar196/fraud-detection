# Kaggle submission

Optional add-on. Nothing in `src/`, `api/` or the CI pipeline depends on this
folder, and the leaderboard is not how this project judges itself — the
[temporal validation](../README.md#temporal-validation) is.

## Generate

```bash
python kaggle/make_submission.py
```

Writes `kaggle/submission.csv` (506,691 rows, `TransactionID,isFraud`), which
matches `data/raw/sample_submission.csv` exactly in shape, columns and ID set.

Requires `data/processed/test_prepared.parquet` (from `scripts/build_dataset.py`)
and `models/model_artifact.pkl` (from `scripts/train.py`).

## Two choices that cost leaderboard rank on purpose

**Scores are uncalibrated.** IEEE-CIS is judged on ROC-AUC, which reads only the
*ordering* of predictions. The isotonic calibrator is a monotone step function:
it cannot improve an ordering, and its flat segments create ties that can only
hurt. Calibration stays in the served model, where a probability has to mean
something. Pass `--calibrated` to compare — expect it to score the same or
slightly worse.

**Encoders are fitted on training data only.** Most competitive solutions fit
frequency and target encodings over train and test together. That is worth real
leaderboard rank, and it is a form of leakage: it lets the model see the test
distribution. This pipeline refuses, because a fraud model has to work on
transactions that have not happened yet — there is no test set to peek at in
production.

Expect a mid-tier public score. That is the intended trade.

## Known issue affecting this submission specifically

The 15 `D*_anchored` features are anchored to an **absolute** day index, and the
Kaggle test period sits at days 213–396 against training's 1–182. They drift
completely against it — KS up to **1.000**, meaning the distributions do not
overlap at all.

The model still leans on them (`D1_anchored` ranks 3rd by SHAP), so this
submission is scored partly on features that do not transfer to the test period.
An [ablation](../README.md#drift-monitoring) prices them at **0.0115 PR-AUC**
in-period — the size of the trap, not a gain worth keeping.

Reformulating the anchor relative to the transaction rather than to absolute
time should improve the leaderboard score. It is listed under
[Future Improvements](../README.md#future-improvements) and has not been done.

## Submitting

Upload `submission.csv` at the
[competition page](https://www.kaggle.com/c/ieee-fraud-detection/submit), or:

```bash
kaggle competitions submit -c ieee-fraud-detection -f kaggle/submission.csv -m "leakage-safe temporal LightGBM"
```
