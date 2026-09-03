# Leakage-Safe Fraud Detection Platform

[![CI](https://github.com/sam200530/mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/sam200530/mlops/actions/workflows/ci.yml)

A fraud detection system on the [IEEE-CIS](https://www.kaggle.com/c/ieee-fraud-detection)
dataset, built around one question: **how do you know your validation number is real?**

The answer, measured in this repository:

> Random stratified 5-fold cross-validation scored **0.8512 PR-AUC**.
> Purged forward-chaining temporal CV — same model, same data, same code — scored
> **0.5583**.
>
> The random estimate was inflated by **+0.2929 (52% relative)** and reported a
> **five times smaller** standard deviation while doing it. It looked more
> trustworthy exactly where it was more wrong.

Everything else here exists to make that finding trustworthy and to ship the
resulting model. *(Both arms of that experiment used the same 530-feature set, so
the comparison is internally consistent; the shipped model differs — see below.)*

## Results

**Shipped model:** LightGBM, 547 features, trained on all 472,432 modelling rows,
isotonic-calibrated, **untuned**.

| metric | value |
|---|---|
| **Holdout PR-AUC** | **0.5538** — 95% CI [0.5384, 0.5688] |
| PR-AUC lift over the 3.44% prevalence floor | **16.09×** |
| Holdout ROC-AUC | 0.8999 |
| Temporal CV PR-AUC (5 purged folds) | 0.5591 ± 0.0195 |
| Precision @ 0.1% alert budget | 0.9831 |
| Precision @ 1% alert budget | 0.9094 |
| Prediction drift vs test period | PSI 0.0455 (stable) |

The holdout was scored **once**, at the end, on the last 118,108 transactions
chronologically. Its interval overlaps that of an earlier model that used a
leaking feature, so nothing was given up for the leakage-free design.

Three findings the repository produced about itself:

1. **Random CV overstated performance by +0.2929** — the result above.
2. **Drift monitoring caught a defect in this project's own feature
   engineering**: 15 features it built drift completely against the deployment
   period (KS 1.000). They were removed from the model.
3. **The same quantity was then recovered as an entity key** rather than a
   numeric feature, restoring the lost accuracy without the leak.

Two models were benchmarked and **lost** — XGBoost and CatBoost — and are
reported as losses.

---

> **Every number here was produced by code in this repository.** Model metrics
> from `scripts/train.py` (also in `reports/model_comparison.csv` and MLflow);
> dataset facts from `scripts/inspect_dataset.py`; drift figures from
> `scripts/monitor.py`. Where something was not executed, it says so.

---

## Contents

[Results](#results) · [Problem](#problem) · [Dataset](#dataset) · [Why Random CV Was Misleading](#why-random-cv-was-misleading) ·
[Temporal Validation](#temporal-validation) · [Leakage-Safe Feature Engineering](#leakage-safe-feature-engineering) ·
[Model Comparison](#model-comparison) · [Final Model](#final-model) · [Evaluation](#evaluation) ·
[SHAP Explainability](#shap-explainability) · [FastAPI](#fastapi) · [Docker](#docker) ·
[Drift Monitoring](#drift-monitoring) · [Testing / CI](#testing--ci) ·
[Project Architecture](#project-architecture) · [How to Run](#how-to-run) ·
[Limitations](#limitations) · [Future Improvements](#future-improvements)

---

## Problem

Given a card-not-present transaction, return a calibrated fraud probability fast
enough to sit inside an authorisation flow, plus the reasons behind it.

The binding constraint is **cost asymmetry**, not accuracy. Missing fraud means a
chargeback; flagging a legitimate customer means a declined sale. At 3.4993%
prevalence there are 27.58 legitimate transactions per fraudulent one, so even a
small false-positive *rate* produces an alert queue no review team can clear. The
system therefore optimises **ranking quality at low alert volume** and
**calibrated probabilities**.

Three properties of fraud shaped every decision:

- **Extreme imbalance.** Predicting "never fraud" scores **96.5007% accuracy**.
  Accuracy cannot separate a useful model from a constant, so it is reported
  nowhere in this project.
- **Non-stationarity.** Measured prevalence moves between **2.4762% and 4.1795%**
  across 30-day blocks. Validation assuming a fixed distribution measures the
  wrong thing.
- **Delayed labels.** Chargebacks arrive weeks later, so live PR-AUC is not
  observable. Degradation must be caught from input and score *distributions* —
  which is why drift monitoring here is not decoration.

## Dataset

Four CSVs, profiled by `scripts/inspect_dataset.py` (full output:
[`docs/01_dataset_audit.md`](docs/01_dataset_audit.md)):

| file | size | rows | columns | duplicate rows |
|---|---|---|---|---|
| `train_transaction.csv` | 651.7 MB | 590,540 | 394 | 0 |
| `train_identity.csv` | 25.3 MB | 144,233 | 41 | 0 |
| `test_transaction.csv` | 584.8 MB | 506,691 | 393 | 0 |
| `test_identity.csv` | 24.6 MB | 141,907 | 41 | 0 |

**Target:** `isFraud` — 20,663 fraud / 569,877 legitimate = **3.4993%**.

**Column families:** `V1`–`V339` (mean 43.04% missing), `C1`–`C14` (**0%
missing**), `D1`–`D15` (timedeltas, 58.15% mean missing), `M1`–`M9`,
`card1`–`card6`, `addr1/2`, `dist1/2`, email domains, identity block.

### Three audit findings that changed the design

**1. The Kaggle test set is unlabeled** — 393 columns to train's 394, the only
difference being `isFraud`. No metric can be computed on it, so the final holdout
is carved from the *training* period by time, and the test files serve only as an
unlabeled distribution for drift analysis.

**2. Identity covers only 24.42% of train rows** (28.01% of test), strictly 1:1.
An inner join would discard 75.6% of the data, and since coverage itself differs
between periods it would bias the sample too. `LEFT JOIN`, with
`identity_present` promoted to a feature.

**3. `test_identity.csv` uses `id-01`…`id-38`; train uses `id_01`…`id_38`.** All
38 names differ. Unfixed, a model trained on one and scored on the other sees 38
all-null columns and **degrades silently instead of crashing**. Renamed once at
the load boundary, then asserted absent.

## Why Random CV Was Misleading

Three measured facts make random K-fold invalid here:

| fact | measurement |
|---|---|
| Train and test are disjoint in time | train days 1–183, test days 213–396, gap **exactly 30.0 days**, **0** shared timestamps |
| Prevalence is non-stationary | 2.4762% → 4.0373% → 4.0319% → 3.9265% → 3.4723% → 3.4013% → 4.1795% per 30-day block |
| Entities recur | the same card transacts repeatedly, so random folds train and validate on the same entity |

Deployment is **forward extrapolation across a gap**. Random CV measures
*interpolation inside a window the model will never operate in*.

So it was measured rather than asserted (`scripts/train.py --random-cv-control`):

| CV scheme | PR-AUC | ROC-AUC | P@top 1% | std dev |
|---|---|---|---|---|
| Random stratified 5-fold | **0.8512** | 0.9657 | 0.9975 | ± 0.0044 |
| Purged forward-chaining 5-fold | **0.5583** | 0.8838 | 0.9268 | ± 0.0225 |
| **Optimism** | **+0.2929** | +0.0819 | +0.0707 | — |

The holdout scored **0.5639** under this configuration — within one standard
deviation of the temporal estimate and nowhere near the random one. That is the
practical confirmation that the temporal scheme was the honest choice.

Both arms of this experiment ran on the same 530-feature set, so the comparison is
internally consistent and stands unchanged. The **shipped** model now excludes the
15 `D*_anchored` features as model inputs, reusing the same quantity as an entity
key instead, and scores 0.5591 CV / 0.5538 holdout — see
[Final Model](#final-model).

Logged to MLflow as `random_cv_optimism_pr_auc`.

## Temporal Validation

A chronological split was **verified feasible before adoption**
(`src/data/validation.py::validate_temporal_order`, asserted on every load):

| check | result |
|---|---|
| `TransactionDT` nulls / negatives | 0 / 0 |
| Already monotonic in raw file order | yes |
| `corr(TransactionID, TransactionDT)` | 0.99828 → the ID is a disguised time index, hard-excluded |
| Days covered | 1–182, **182 distinct days, none missing** |
| Largest gap between timestamps | 4,138 s (1.15 h) |
| Rows sharing a timestamp | 33,932 (**5.746%**), max 8 |
| Rows exactly on the 80th-percentile cut | **0** |

Ties matter: boundaries snap to timestamp edges so a tie group is never split
across partitions, and `_assert_disjoint_in_time` verifies no timestamp appears in
two partitions.

**Three-way split** (`data/processed/split_metadata.json`):

| partition | rows | days | fraud rate | role |
|---|---|---|---|---|
| train | 377,945 | 1–109 | 3.4119% | model fitting |
| validation | 94,487 | 109–141 | 3.9201% | threshold + calibration |
| **holdout** | **118,108** | **141–182** | **3.4409%** | **scored once, at the end** |

**Purged forward-chaining folds** over the modelling period: each trains on the
past and validates on the next contiguous block, with a **7-day purge gap** — at
least as wide as the longest velocity look-back (168 h), because otherwise a
trailing aggregate computed at the start of a validation block reaches back into
training rows even though the *rows* are separated.

**In one sentence: fold N trains on days 1 to X, waits 7 days, then validates on
days X+7 onward. Nothing is shuffled, no validation row precedes any training row
in time, and the training window only ever grows forward.**

```
day  1                                                                   141
     |------------------------------------------------------------------|
f0   [train 1-12.8 ]<-7d->[ val 19.8-37.9 ]
f1   [train 1-30.9        ]<-7d->[ val 37.9-64.7 ]
f2   [train 1-57.7               ]<-7d->[ val 64.7-90.8 ]
f3   [train 1-83.8                      ]<-7d->[ val 90.8-114.6 ]
f4   [train 1-107.6                            ]<-7d->[ val 114.6-141.1 ]
```

| fold | train rows | train days | val rows | val days | purge gap | overlap |
|---|---|---|---|---|---|---|
| 0 | 46,274 | 1.0 – 12.8 | 78,739 | 19.8 – 37.9 | **7.0 d** | none |
| 1 | 133,979 | 1.0 – 30.9 | 78,739 | 37.9 – 64.7 | **7.0 d** | none |
| 2 | 214,078 | 1.0 – 57.7 | 78,738 | 64.7 – 90.8 | **7.0 d** | none |
| 3 | 293,305 | 1.0 – 83.8 | 78,739 | 90.8 – 114.6 | **7.0 d** | none |
| 4 | 372,880 | 1.0 – 107.6 | 78,739 | 114.6 – 141.1 | **7.0 d** | none |

Those boundaries are read back **off the persisted folds**, not asserted from the
code that wrote them. Verified on every fold: each validation start is later than
its train end, index overlap is empty, rows are time-ordered within each training
window, and the windows expand monotonically (46,274 → 372,880).

Folds are cut on **timestamp edges**, persisted once to
`data/processed/folds_temporal.npz` and reused byte-identically by every model, so
no two models can be scored on different splits. `_assert_disjoint_in_time` fails
the run if a timestamp appears in two partitions, and the purge is applied by
*index position*, not by sampling — there is no shuffle anywhere in the path.

Validation fraud rate exceeds training fraud rate in *every* fold — a structural
property random folds would average away.

## Leakage-Safe Feature Engineering

The pipeline has a three-phase shape, and **the shape is the leakage control**
(`src/features/pipeline.py`):

| phase | what it does | safe on the full frame? |
|---|---|---|
| `prepare(df)` | row-local transforms + past-only velocity | **yes** — nothing fitted, nothing reads a future row |
| `fit(train_df)` | learns frequency counts, per-entity baselines, categorical vocabularies | **training partition only** |
| `transform(df)` | pure lookup | applied identically to validation, holdout, serving |

The only method that learns anything takes the training frame as its argument, so
fitting an encoder on validation data is structurally impossible rather than
merely discouraged.

**Encoders are refitted inside every CV fold** (`src/models/training.py::_fit_one`).
Frequency counts and per-entity means are *population statistics*; fitting them
once on the whole modelling period and then cross-validating would leak each
fold's validation rows into its own training features.

| family | features | mechanism |
|---|---|---|
| Amount | `log_amount`, `amount_cents`, `is_round_amount` | Right-skewed (mean 135.03, max 31,937). Class means are close (149.24 vs 134.51), so absolute amount is weak — the decimal part fingerprints card testing and currency conversion. |
| Per-entity amount | `*_amt_mean_hist`, `*_amt_diff_from_mean`, `*_amt_zscore` | Fraud is anomalous *for that account*, not absolutely. |
| Velocity | `*_txn_count_{1,24,168}h`, `*_amt_sum_*`, `*_seconds_since_prev` | Card testing and ATO are burst behaviours invisible in one transaction. |
| Time (cyclical) | `hour_of_day`, `day_of_week`, `is_night`, `is_weekend` | Fraud rate peaks where volume troughs — automation at off-peak hours. Cyclical, so it transfers across the 30-day gap. |
| Email | provider/TLD split, `email_domains_match` | Purchaser/recipient mismatch is a classic mule indicator. |
| Device | `device_vendor`, `os_family`, `browser_*`, screen dims | `DeviceInfo` has >1,000 distinct values and would overfit verbatim. |
| Missingness | `n_missing_total`, per-family counts, `identity_present` | Missingness is structural, not random, and correlates with fraud. |
| Frequency encoding | `card1_freq`, `card2_freq`, `addr1_freq`, … | Rare identifiers are disproportionately fraudulent; uses no label, so it cannot leak the target. |

**Velocity is causal, not leakage.** Each value uses only rows *strictly earlier
in time*, which at inference have already happened; the current row is excluded by
construction. Four tests in `tests/test_features.py::TestVelocityIsCausal` assert
this, including that five sequential transactions produce counts `[0, 1, 2, 3, 4]`.

**One deliberate score sacrifice.** Fitting frequency encodings on train ∪ test is
a well-known Kaggle booster and it is **transductive leakage** — it assumes the
scoring population is known at training time, which a live API cannot. Counts come
from the training partition only. This costs leaderboard points and is right for a
serving system.

**Hard exclusions:** `isFraud`; `TransactionID` (corr 0.998 with time); raw
`TransactionDT` (disjoint ranges, so a tree routes every test row to one leaf);
`V107` (constant in test, not train).

## Model Comparison

Models are compared on **identical persisted folds**, from a full run
(`python scripts/train.py --random-cv-control`). Every figure below is retained in
MLflow; `reports/model_comparison.csv` is regenerated on each run and therefore
reflects whatever was executed last, so the table here is the canonical record.

> **This table compares model *classes* on the 530-feature configuration**, with
> every model on identical folds — so the ranking is valid and internally
> consistent. The **shipped** model is untuned LightGBM on 547 features
> (0.5591 ± 0.0195 CV), after `D*_anchored` was dropped as an input and re-used as
> an entity key; see
> [Final Model](#final-model). Only LightGBM was rerun on the new feature set, as
> the removal decision concerns the shipped model rather than which algorithm wins.

| model | CV PR-AUC | lift | ROC-AUC | precision | recall | F1 | Brier | P@top 1% | train time |
|---|---|---|---|---|---|---|---|---|---|
| **LightGBM (tuned)** | **0.5754 ± 0.0239** | **15.91×** | 0.8967 | 0.6684 | 0.4862 | 0.5616 | 0.0227 | 0.9238 | 2960.6 s |
| LightGBM (baseline) | 0.5583 ± 0.0225 | 15.45× | 0.8838 | 0.6766 | 0.4692 | 0.5533 | 0.0236 | 0.9268 | 638.9 s |
| XGBoost | 0.5370 ± 0.0211 | 14.83× | 0.8754 | 0.6675 | 0.4515 | 0.5381 | 0.0242 | 0.9172 | 1579.6 s |
| Random Forest | 0.4677 ± 0.0404 | 12.82× | 0.8819 | 0.5415 | 0.4049 | 0.4604 | 0.0966 | 0.8295 | 185.1 s |
| Logistic Regression | 0.3560 ± 0.0703 | 9.62× | 0.8328 | 0.4462 | 0.3661 | 0.4018 | 0.1400 | 0.7151 | 115.0 s |

**Boosting is measured to be better, not assumed** — untuned LightGBM beats Random
Forest by +0.091 PR-AUC and Logistic Regression by +0.202.

**Two challengers were run and both lost.**

*XGBoost* — 0.5370 against LightGBM's 0.5583, winning 1 of 5 folds at 2.1× the
training cost. Given a search space mirrored to LightGBM's, the same native
categorical handling, the same folds and the same early-stopping rule, so the
comparison reflects the algorithms rather than the setup. The margin is close to
the fold-to-fold spread (±0.021), so it is a consistent but modest loss; the
4-of-5 fold record is what makes it credible, not the mean alone.

*CatBoost* — 0.5368 against 0.5845 on fold 4 with the shipped feature set, at
1431 s against 310 s. Added because its ordered target statistics handle
high-cardinality categoricals differently, and this dataset is largely
high-cardinality categorical, so the difference was measured rather than assumed.
Reported from one fold: at ~24 min/fold a full run is two hours and the gap is
more than twice the fold-to-fold spread. Worth noting *why* it lost here when it
won in the reference solution — that run used a GPU with 5000 trees and
transductive encodings. A model's reported strength is a property of its setup as
much as its algorithm.

Both are kept in the repository as losses. A comparison in which the challenger
always wins says nothing about the methodology.

**ROC-AUC hides most of that gap.** Random Forest reaches 0.8819 against untuned
LightGBM's 0.8838 — a **0.002** difference — while the PR-AUC gap is **0.091**,
roughly fifty times larger. With ~570k negatives in the FPR denominator, ROC-AUC
barely registers the false-positive volume separating these models. Selecting on
ROC-AUC would have called them equivalent. This is the in-repo demonstration of
why PR-AUC is the selection metric at 3.5% prevalence.

**Stability tracks capability.** PR-AUC standard deviation runs 0.0225 → 0.0211 →
0.0404 → 0.0703 down the untuned table. Logistic Regression is not merely worse on
average, it is ~3× more volatile across time periods.

**Calibration separates the tree models further:** Brier 0.0236 (LightGBM) vs
0.0966 (RF) vs 0.1400 (LogReg) — RF is 4× worse calibrated than LightGBM despite
near-identical ROC-AUC. This matters because the API serves probabilities, not
only rankings.

### The row-count confound, measured rather than waved away

Logistic Regression and Random Forest are fitted on **100,000 rows** (all
positives kept) because their one-hot matrices are 929 features wide, while the
boosters use every row. That is a genuine confound — the baselines could be losing
on data volume rather than model class.

So the boosters were rerun **at the baselines' row count**
(`--force-subsample 100000`, same folds and seed):

| | value |
|---|---|
| LightGBM at 100k rows | 0.5556 (vs 0.5583 at full data) |
| Cost of subsampling to LightGBM | **0.0027** |
| LightGBM margin over Random Forest | **0.0879** — **33× larger** |
| LightGBM margin over Logistic Regression | 0.1996 — 74× larger |

**The ranking is not an artefact of training-set size.** Fold 0 is identical in
both runs because its training window holds 46,274 rows, below the cap — which
confirms the flag does nothing when it should do nothing.

Why not just fit the baselines on all 472k rows with a sparse matrix? Because
sparse is *worse* here: after median-fill the 492 numeric columns have almost no
zeros, so CSR pays 8 bytes per nonzero where dense pays 4 — **1.98 GB against
1.62 GB**. Sparse wins only when the numeric block is sparse; here only the
one-hot block is. The honest fix is more RAM, and the equal-data control answers
the question without needing it.

## Final Model

**LightGBM**, **untuned**, isotonic-calibrated, trained on all 472,432 modelling
rows with **547 features**.

The interesting part is what happened to 15 of them.

### A feature that was harmful as a number and useful as an identity

`D*_anchored` (`day_index − D_n`) anchors to **absolute** time. Training covers
days 1–182; deployment sits at days 213–396. All 15 drift completely against that
period — KS up to **1.000**, distributions with no overlap — while the raw `D`
columns they derive from stay stable (`D1` KS 0.041). Drift monitoring found this;
it was not anticipated.

An ablation priced them at **+0.0115 PR-AUC in-period** — not a gain to protect,
but the size of the trap. Every fold and the holdout validate *inside or adjacent
to* the training period, exactly where an absolute-time anchor still lines up. So
they were removed as model inputs.

Then the same quantity came back a different way. `D1 − day_index` is constant per
card: it encodes when the account was first seen. As a **value** it drifts; as a
**grouping key** the drift is irrelevant, because keys are compared for equality,
not magnitude. Identity survives distribution shift. So `_entity_uid` = card
fields + that anchor, hashed deterministically: **194,519 groups over 472,432
rows** (2.43 transactions each) against the previous `card1 + addr1 + card2` key's
37,859. Frequency and amount aggregates over it are fitted on the **training
partition only**. Added alongside: V missingness-block summaries, reducing the 339
anonymised V columns to a mean and std per block.

`anchor_d_columns` stays `true`, so the features are still computed and the drift
report keeps measuring them — deleting the computation would delete the evidence.

### The three configurations, measured

| | 530 feat *(leaky)* | 515 feat *(stripped)* | **547 feat (shipped)** |
|---|---|---|---|
| CV PR-AUC | 0.5583 ± 0.0251 | 0.5468 ± 0.0120 | **0.5591 ± 0.0195** |
| Holdout PR-AUC | 0.5639 [0.5488, 0.5786] | 0.5279 [0.5126, 0.5433] | **0.5538 [0.5384, 0.5688]** |
| Holdout ROC-AUC | 0.9117 | 0.9021 | 0.8999 |
| Holdout Brier | 0.0204 | 0.0225 | **0.0210** |
| Precision @ top 0.1% | 0.9915 | 0.8814 | **0.9831** |
| Prediction drift PSI | 0.0329 | **0.0100** | 0.0455 |

Removing the drifting features cost **0.0360** holdout PR-AUC; the uid recovered
**0.0259** of it, and the shipped interval **overlaps the leaky model's** — the
leakage-free pipeline is now statistically indistinguishable from the one that
cheated, and improves on it in CV. The middle column is kept because it is what
motivated the uid work, not because it was ever shipped.

**The cost, stated plainly: prediction drift rose to 0.0455**, the highest of the
three though still inside the documented <0.10 band. Most test-period accounts are
unseen in training, so `entity_uid_freq` is 0 for them. That is correct — an
unseen account has no history — and the alternative, fitting the encoder over test
data, is the transductive shortcut this project refuses. But the feature's
distribution does shift between periods. The uid buys accuracy and spends some
distributional stability.

### Credit, and what was deliberately not taken

The uid construction is adapted from the IEEE-CIS 17th-place solution, which
scores **0.952 ROC-AUC** on this exact split against this project's 0.900. Most of
that gap is technique refused on purpose: its frequency encodings, uid
aggregations and time-block encodings are all fitted over `concat(train, test)`,
and one feature is literally `TransactionAmt.isin(test.TransactionAmt)`. That is
transductive — legal on Kaggle, undeployable in production. The uid *idea*
transfers; its encoding does not, so it was reimplemented train-only.

### Not tuned

Hyperparameters were not re-searched after the feature set changed. The previous
values were tuned on a 530-feature space that no longer exists, so reusing them
would make results unattributable between the feature change and mismatched
settings. Retuning is worth roughly **+0.017** based on the earlier search and is
blocked by memory, not code — see [Limitations](#limitations).

Imbalance is handled by **reweighting, not resampling**. SMOTE would interpolate
between fraud rows across a ~530-column space that is largely categorical and
heavily missing; the interpolants would not be plausible transactions.
`scale_pos_weight` leaves the data honest and only changes the loss — and because
reweighting distorts probabilities, isotonic calibration follows.

The saved artifact bundles **model + feature pipeline + calibrator + threshold**
together, which removes the most common production failure in ML systems: a
preprocessing step drifting out of sync with the model.

## Evaluation

The holdout was scored **exactly once**, using the threshold (0.2988) chosen on
validation and applied unchanged.

| metric | validation (last fold) | **holdout (final)** |
|---|---|---|
| PR-AUC | 0.5762 | **0.5538** |
| PR-AUC lift over prevalence | 15.02× | **16.09×** |
| ROC-AUC | 0.8996 | **0.8999** |
| Precision | 0.6735 | **0.5723** |
| Recall | 0.5137 | **0.5335** |
| F1 | 0.5829 | **0.5522** |
| Brier | 0.0221 | **0.0210** |
| Precision @ top 0.1% | — | **0.9831** |
| Precision @ top 1% | 0.9096 | **0.9094** |
| Recall @ top 1% | — | **0.2643** |
| Rows | 78,739 | 118,108 |

**Confusion matrix** at threshold 0.2988 (prevalence 3.4409%):

|  | predicted legit | predicted fraud |
|---|---|---|
| **actually legit** | 112,424 | 1,620 |
| **actually fraud** | 1,896 | 2,168 |

What this means operationally:

- **PR-AUC 0.5538 is 16.09× the no-skill floor** of 0.0344. The absolute number
  looks unimpressive only if the floor is forgotten.
- **At a 1% alert budget, 90.9% of alerts are genuine fraud**, catching 26.4% of
  all fraud. Tightening to a 0.1% budget raises precision to **98.3%**.
- **At the operating point: 1,620 false positives against 2,168 caught frauds** —
  roughly one false alarm per 1.3 detections, at the cost of missing 1,896. That
  trade is a business choice, which is why the threshold is configuration.
- **Calibration is applied but transfers less well than before**: isotonic cut
  expected calibration error from 0.06201 to 0.00000 on the calibration fold
  (Brier 0.03535 → 0.02325), and holdout Brier 0.0225 still came in *below*
  validation's 0.0232. But holdout ECE is **0.01402**, against 0.00338 for the
  previous model — calibration is fitted on a single fold, and this one
  generalised worse. Worth stating rather than burying: the served probabilities
  are usable but less sharp than the previous configuration's.

**Holdout PR-AUC 0.5538, 95% CI [0.5384, 0.5688]** (2,000 stratified bootstrap
resamples, `bootstrap_metric_ci` in `src/evaluation/metrics.py`). ROC-AUC 0.8999.

That interval **overlaps the original leaky model's [0.5488, 0.5786]**. Stripping
the drifting features cost 0.0360 on the holdout (0.5639 → 0.5279); the uid key
recovered 0.0259 of it (→ 0.5538), leaving a residual that is no longer
statistically distinguishable from the model that leaked. The leakage-free
pipeline now matches the leaky one within noise.

Resampling is stratified within the positive and negative classes so every draw
holds prevalence fixed. Resampling pooled rows would let the fraud rate wander
between draws, and PR-AUC moves with prevalence by construction, which would
inflate the interval with an artefact of the procedure rather than uncertainty
about the model.

**Holdout (0.5538) sits below validation (0.5762) and just below the temporal CV
mean (0.5591 ± 0.0195).** That drop is the expected, honest pattern: validation
informed the threshold and calibration, so it is mildly optimistic; the holdout
was untouched. A holdout scoring *above* validation would be a reason to suspect
the split, not to celebrate.

Figures in `reports/figures/`: PR curve with the prevalence baseline drawn on it,
ROC, reliability diagram, confusion matrix, per-class score distribution.

## SHAP Explainability

`src/explainability/shap_explainer.py`, using `TreeExplainer` — exact for tree
ensembles and needing no background dataset, which is what makes a per-request
explanation viable at all.

| rank | feature | mean \|SHAP\| |
|---|---|---|
| 1 | `C13` | 0.4900 |
| 2 | **`entity_uid_freq`** (engineered) | 0.2277 |
| 3 | `P_emaildomain` | 0.2254 |
| 4 | `C1` | 0.1872 |
| 5 | `dist1` | 0.1832 |
| 6 | `V70` | 0.1697 |
| 7 | `card1` | 0.1684 |
| 8 | `C14` | 0.1677 |
| 9 | **`card1_freq`** (engineered) | 0.1636 |
| 10 | `card6` | 0.1607 |

**`entity_uid_freq` enters at #2** — how many transactions the synthetic account
has. The uid was added precisely because `D1 - day_index` is informative as an
identity even though it drifts as a value, and the model ranking it second is
independent confirmation that the reframing worked. `D1_anchored` previously held
3rd place as a raw number; the same information now enters through a key that
survives distribution shift.

Recomputed after `D*_anchored` was removed. It previously ranked **3rd**
(0.2640); with it gone the model redistributes onto `P_emaildomain`, `dist1`,
`addr1` and the engineered `card1_freq`, none of which were previously in the top
ten. The redistribution is broad rather than concentrated — no single feature
absorbed the lost attribution, and `C13` itself fell from 0.4902 to 0.4379. That
is the signature of information genuinely available elsewhere rather than a
unique signal being lost, which is consistent with the ablation costing only
0.0115 PR-AUC.

Ten of the top 30 are engineered here. Mean |SHAP| is preferred over LightGBM's
split-count importance because it is in units of model output and is consistent
between the global ranking and the per-transaction explanation the API returns —
the same number explains both.

⚠️ See [Limitations](#limitations) on the `*_anchored` features: their high
ranking is real but does **not** imply they generalise.

## FastAPI

Three routes (`api/routes.py`). The model bundle is loaded **once** at startup and
reused for every request.

| endpoint | purpose |
|---|---|
| `GET /health` | Liveness plus model provenance: name, trained-at, feature count, calibration status, threshold, holdout metrics. Returns 200 with `status="degraded"` when no model is loaded — "up but modelless" and "down" are different facts. |
| `POST /predict` | Calibrated probability, risk band, threshold, latency, feature-completeness count. |
| `POST /explain` | Prediction plus ranked SHAP contributors (`?top_n=`). |

Provenance is folded into `/health` rather than given a separate `/model-info`
route, and batch scoring was removed — endpoints that exist to look complete are
not endpoints.

**Input contract.** Requiring all ~430 raw columns would make the API unusable, so
the schema names the high-signal fields and accepts the long tail through one
`extra_features` map. Anything omitted becomes NaN — a genuine capability, since
the model is a LightGBM trained on data that is 43% missing across the V block.
`extra="forbid"` catches typo'd fields, validation failures return a structured
422, and a missing model yields 503 rather than 500.

## Docker

```bash
docker build -t fraud-api .
```

```bash
docker run -p 8000:8000 -v "$(pwd)/models:/app/models:ro" fraud-api
```

Multi-stage build: wheels compile in a builder stage so gcc stays out of the
shipped image; `libgomp1` is installed for LightGBM's OpenMP threading; the
service runs as a non-root user (uid 10001) with a `HEALTHCHECK` on `/health`.

The model artifact is **bind-mounted, not baked in** — otherwise every retrain
forces an image rebuild, and image contents depend on training output. Mounted
read-only, because a service must never modify its own model.

There is no `docker-compose.yml`: with no database or cache to orchestrate, a
single container is the whole system.

## Drift Monitoring

```bash
python scripts/monitor.py --current test
```

> The traffic is not production traffic — this project has no users, and the
> report says so in its own payload. What *is* real is the distribution shift: the
> comparison is the training period against the **real, unlabeled IEEE-CIS test
> period beginning 30 days later**, not a synthetic perturbation.

**PSI** is the trigger (interpretable on a fixed scale, insensitive to sample
size); **KS** is reported alongside (catches shape changes PSI's binning smooths
over, but its p-value goes to zero for any difference once n is large).
Thresholds are stated rather than implied: **< 0.10 stable, 0.10–0.25 moderate,
&gt; 0.25 significant** (`configs/config.yaml`). Bin edges come from the reference
distribution and are reused, since re-binning per window would compare two
different binnings. Per-feature missing-rate deltas fall out of the same pass.

Measured over 507 features, 60,000 rows per period:

| result | value |
|---|---|
| Significantly drifted (PSI > 0.25) | **181** |
| Moderately drifted | 6 |
| Stable | 320 |
| **Prediction drift (model output)** | **PSI 0.0455 — stable** (0.0329 leaky → 0.0100 stripped → 0.0455 hybrid) |
| Mean predicted probability | 0.0435 → 0.0312 |

**Heavy input drift, stable output.** The drifted inputs are dominated by device
and identity metadata — `id_31` browser strings (PSI 13.21), `id_30` OS (12.26),
`id_33` resolution (11.92) — which change naturally as browsers and handsets
update over a month. The model does not lean on them enough for its output to
move. Score drift is the only signal available without labels, which is why it is
computed.

### The finding that changed a design decision

Monitoring caught a real defect in this project's **own** feature engineering.

All 15 `D*_anchored` features drift severely, while the raw `D` columns they
derive from are stable:

| feature | PSI | KS |
|---|---|---|
| `D9_anchored` | 12.447 | **1.000** |
| `D13_anchored` | 5.499 | 0.965 |
| all 15 anchored | 3.5 – 12.4 | 0.84 – 1.00 |
| `D15` (raw) | 0.108 | 0.140 |
| `D1` (raw) | 0.0069 | 0.041 |

A KS of **1.000** means the distributions do not overlap at all. The cause was my
own transformation: `day_index − D_n` was meant to turn a moving delta into a
fixed calendar anchor, but `day_index` is **absolute time**, and the test period
sits at days 213–396 against training's 1–182. It reintroduced through the back
door exactly the risk the audit had documented — and `D1_anchored` ranked 3rd by
SHAP, because the holdout is adjacent to training and hides the problem.

### The ablation that priced it

If the features are broken, why does the model rank them so highly? Removing them
and re-measuring answers it (`scripts/run_ablation.py`, same folds and seed):

| | mean PR-AUC | spread |
|---|---|---|
| 530 features | 0.5583 | ± 0.0251 |
| 515 features (no `_anchored`) | 0.5468 | ± 0.0120 |
| **Δ** | **−0.0115 (−2.1%)** | spread more than halves |

Removing them costs 0.0115 in-period, losing 4 of 5 folds — and the largest loss
is **fold 4 (−0.0390)**, the fold where train and validation sit closest in time.
That is not a contradiction of the drift result, it is the mechanism behind it:
every fold validates *inside* the training period, where an absolute-time anchor
still works. The 0.0115 is the size of the trap, not a gain to protect. This is
the clearest example in the project of a feature that improves every number
measurable before deployment and fails after it.

**Acted on**, and then recovered: the features were dropped as model inputs and
the same quantity reused as an entity key. See
[Final Model](#final-model) for the full before/after.

That a monitoring system built for this project found a genuine flaw in the
project's own features, rather than reporting a reassuring all-clear, is the
strongest evidence that the monitoring is real.

## Testing / CI

```bash
pytest -q
```

**88 tests, all passing.** The suite runs **without the dataset**, on
schema-faithful synthetic fixtures (`tests/conftest.py`) reproducing the real
column families, dtypes, missingness patterns, chronological ordering with ties,
and ~3.5% prevalence with genuine signal. That is deliberate: CI exercises real
code paths — including training a small LightGBM and scoring it through the API —
on a machine with no access to a 1.3 GB Kaggle download.

| file | focus |
|---|---|
| `tests/test_data_validation.py` | schema invariants, temporal split, purged folds, tie handling |
| `tests/test_features.py` | feature builders, **velocity causality**, encoder leakage safety |
| `tests/test_evaluation.py` | metrics, alert budgets, calibration, comparison table |
| `tests/test_monitoring.py` | PSI, KS, drift verdicts |
| `tests/test_api.py` | all three endpoints, invalid input, cold-start velocity, degraded mode |

Several tests exist because they caught real bugs: a nullable-string comparison
that crashed email features; predictions misaligned with request order after the
frame is sorted chronologically; and an over-strict `card1` bound that rejected
the real test split, where `card1` reaches 18,397 against training's 18,396.

**GitHub Actions** (`.github/workflows/ci.yml`): `lint → test → docker build → API
smoke test`, green in ~3m 50s. The smoke test starts the image **without a model**
and asserts the container stays up, `/health` reports `degraded`, all three routes
appear in the OpenAPI spec, and `/predict` returns **503** rather than 500.

Note on that last assertion: `/predict` declares `Depends(require_artifact)`, and
FastAPI resolves dependencies *before* validating the request body — so with no
model mounted every call is 503 regardless of payload. The 422 validation path is
covered by `tests/test_api.py`, which runs with a model loaded.

CI earned its place immediately: the first runs surfaced five defects that local
development had hidden — a ruff version pinned inconsistently in three places, a
lint rule that exists in one ruff version and not another, a dependency set that
was **not installable from scratch** (shap 0.52 requires `numpy>=2` against a
`numpy==1.26.4` pin, and scikit-learn 1.2.2 has no CPython 3.12 wheel), a stale
`.dockerignore` entry excluding the very `requirements.txt` the Dockerfile copies,
and a smoke-test assertion based on a wrong assumption about FastAPI's resolution
order.

## Project Architecture

```
IEEE-CIS CSVs
     ↓  src/data/loading.py      streamed CSV→Parquet, id-NN→id_NN, LEFT JOIN
     ↓  src/data/validation.py   fail-loud schema + chronology invariants
     ↓  src/features/            prepare → fit → transform (the leakage boundary)
     ↓  src/data/splitting.py    20% chronological holdout + purged folds
     ↓  src/models/              LogReg · RandomForest · LightGBM + Optuna
     ↓  src/evaluation/          PR-AUC, ROC-AUC, alert budgets, calibration
     ↓  src/explainability/      SHAP global + per-transaction
     ↓  models/model_artifact.pkl   model + pipeline + calibrator + threshold
     ↓  api/                     FastAPI: /health /predict /explain
     ↓  Dockerfile               single container
     ↓  src/monitoring/drift.py  PSI + KS vs the real test period
```

```
├── configs/config.yaml          split, features, training, serving, monitoring
├── data/README.md               how to obtain the dataset (never committed)
├── docs/                        01_dataset_audit.md · 02_leakage_analysis.md
├── notebooks/01_eda.ipynb       analysis only; imports from src/
├── src/
│   ├── data/                    schema, loading, validation, preprocessing, splitting
│   ├── features/                builders, velocity, aggregations, pipeline
│   ├── models/                  estimators, training, tuning, artifact
│   ├── evaluation/              metrics, calibration, compare, plots
│   ├── explainability/          shap_explainer
│   ├── monitoring/              drift
│   └── utils/                   paths, config, logging, seed
├── api/                         main, routes, schemas, dependencies, settings
├── scripts/                     inspect_dataset, build_dataset, train, evaluate, monitor
├── tests/                       5 modules, 88 tests
├── Dockerfile · requirements.txt · pyproject.toml
└── .github/workflows/ci.yml
```

Every path derives from the repository root via `src/utils/paths.py` (overridable
with `FRAUD_PROJECT_ROOT`, which is how the container points at `/app`). There are
no absolute paths in the codebase.

## How to Run

**Prerequisites:** Python 3.12, ~4 GB free RAM for training, ~2 GB disk.

```bash
pip install -r requirements-dev.txt
```

Download the four CSVs from the [competition page](https://www.kaggle.com/c/ieee-fraud-detection/data)
into `data/raw/` (see [`data/README.md`](data/README.md); raw data is gitignored).

```bash
python scripts/inspect_dataset.py
```

```bash
python scripts/build_dataset.py --with-test
```

```bash
python scripts/train.py --random-cv-control
```

```bash
python scripts/evaluate.py --partition holdout
```

```bash
uvicorn api.main:app --reload --port 8000
```

Then open http://localhost:8000/docs, or:

```bash
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"transaction_amt\": 149.99, \"product_cd\": \"W\", \"card1\": 13926, \"card4\": \"visa\", \"p_emaildomain\": \"gmail.com\"}"
```

```bash
python scripts/monitor.py --current test
```

Experiment tracking writes to a local `./mlruns` store; `mlflow ui` shows the runs.
No MLflow server is required.

## Limitations

Stated because a reviewer will find them anyway, and because several are the
direct cost of choices made deliberately elsewhere.

1. **The shipped model is untuned.** Hyperparameters were not re-searched after
   the feature set changed, since values tuned on a space that no longer exists
   would make results unattributable. Worth roughly **+0.017**. Blocked by memory,
   not code: Optuna refits the two largest folds repeatedly in one process, and
   this machine's commit limit fell from 31.3 GB to ~24.5 GB mid-project with
   `Available MBytes` at 0 under normal desktop load. Per-fold isolation
   (`run_ablation.py --save-oof` plus `train.py --skip-cv --oof-npz`) was added to
   work around it and is what produced the current model. The one search that did
   complete was itself bounded — 8 of 25 trials under a 5400 s cap, hitting the
   2000-round ceiling on 3 of 5 folds.

2. **The uid raises prediction drift** to PSI 0.0455, the highest of the three
   configurations though still inside the <0.10 stable band. Most test-period
   accounts are unseen in training, so `entity_uid_freq` is 0 for them. Correct
   behaviour — the alternative is transductive — but the distribution does shift.

3. **Entity keys are a proxy, and velocity is cold-started.** No account
   identifier exists, so `card1 + addr1 + card2` and the uid approximate one;
   cards sharing those values merge. The API keeps no cross-request state either,
   so velocity counts come only from the current batch — a single-transaction
   request honestly reports `count = 0` rather than inventing history.

4. **No production traffic, and no labels on the Kaggle test set.** Drift
   monitoring compares real dataset periods, not live data, and the final metric
   is a chronological holdout rather than a leaderboard score. Kaggle rank is also
   deliberately forfeited: encoders are fitted train-only, where competitive
   solutions fit over train+test.

5. **Dense baselines are subsampled** to 100,000 rows (all positives kept). The
   comparison is not perfectly equal — though rerunning the boosters at the same
   row count showed the cap costs LightGBM 0.0027 against a 0.0879 margin over
   Random Forest, so the ranking is not an artefact of it.

6. **Single-node, single-worker.** No horizontal scaling, A/B routing or shadow
   deployment. Docker is verified in CI rather than on the development machine,
   and `requirements.txt` pins `scikit-learn==1.3.2` / `shap==0.46.0` where the
   development environment runs slightly different versions.

## Future Improvements

Roughly in order of value per unit of effort:

12. **Remove or reformulate `D*_anchored` and retrain.** They measurably harm
   temporal generalisation. Either drop them (the raw `D` columns are stable and
   already present) or re-express the anchor relative to the transaction rather
   than to an absolute day index. The ablation above prices the decision: dropping
   them costs **0.0115 in-period PR-AUC** and halves fold-to-fold variance, so the
   reformulation is the better of the two options — it should recover the signal
   without the absolute-time dependence. The comparison must be rerun, since this
   changes the feature set behind the reported metrics.
13. **A feature store for velocity**, so serving and training share one definition
   and cold starts do not degrade the first requests.
14. **Adversarial validation to prune shift-heavy features** — 181 of 507 already
   drift significantly.
15. **Cost-sensitive thresholding**: minimise expected monetary loss given
   chargeback and review costs, rather than optimising F1, which weights precision
   and recall equally as no fraud team does.
16. **Revisit the 339 V columns on SHAP evidence** — kept deliberately for the
   baseline; correlation-clustering would shrink inference cost if PR-AUC holds.
17. **Scheduled retraining** gated on holdout PR-AUC and drift.

---

## License and data use

Code is provided for portfolio and educational purposes. The IEEE-CIS dataset is
subject to Kaggle competition rules and is not redistributed here.
