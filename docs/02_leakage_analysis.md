# Leakage Analysis

Phase 3 output. Every claim here is either a measured fact from
`scripts/inspect_dataset.py` or a decision implemented in code, with the file
and mechanism named. Nothing is asserted on intuition alone.

The short version: on this dataset the *validation design* leaks far more easily
than the features do. Three of the four most damaging risks below are properties
of how the data is split, not of any individual column.

---

## 1. Why the split design is the main leakage surface

Measured facts that force the design:

| fact | measurement |
|---|---|
| Train time range | `TransactionDT` 86,400 → 15,811,131 (days 1–183) |
| Test time range | 18,403,224 → 34,214,345 (days 213–396) |
| Gap between them | 2,592,093 s = **exactly 30.0 days** |
| Shared timestamps | **0** |
| Train/test `TransactionID` overlap | none (2,987,000–3,577,539 vs 3,663,549–4,170,239) |
| Fraud rate by 30-day block | 2.4762%, 4.0373%, 4.0319%, 3.9265%, 3.4723%, 3.4013%, 4.1795% |

The deployment task is **forward extrapolation across a 30-day gap**, and
prevalence is **not stationary**. A random split measures neither.

### Verified: a chronological split is actually possible

Before adopting it, the ordering assumptions were tested rather than assumed
(`src/data/validation.py::validate_temporal_order`, asserted on every load):

| check | result |
|---|---|
| `TransactionDT` nulls | 0 |
| Negative values | 0 |
| Already monotonic in raw file order | **yes** |
| `corr(TransactionID, TransactionDT)` | 0.99828022 |
| Days covered | 1–182, **182 distinct days, none missing** |
| Largest gap between consecutive timestamps | 4,138 s (1.15 h); only 1 gap > 1 h |
| Rows per day | 2,048 min / 6,852 max |
| Rows sharing a timestamp with another row | 33,932 (**5.746%**), max 8 on one timestamp |
| Rows exactly on the 80th-percentile cut | **0** |

So a chronological split is sound. Two consequences were handled explicitly
rather than ignored:

- **Ties.** 5.746% of rows share a timestamp. `find_time_cut` snaps every
  boundary to a timestamp edge, so a tie group is never split across
  partitions — near-simultaneous transactions are plausibly the same
  card-testing burst, and separating them would put the same event on both sides
  of the boundary. `_assert_disjoint_in_time` then verifies no timestamp value
  appears in two partitions.
- **`TransactionID` is a time proxy.** At correlation 0.998 with
  `TransactionDT`, a model given the ID can locate the time window. It is on the
  hard-exclusion list.

**No timestamp issues were found, so no random split was used.** Had ordering
been broken, the failure would have surfaced as a raised
`DataValidationError` — the pipeline cannot silently fall back to a random split.

---

## 2. The leakage register

Each risk, its mechanism, and where it is controlled.

### R1 — Random K-fold across a temporal dataset · **controlled**

Random folds let a model interpolate inside a window it will never have at
inference. Control: `PurgedForwardChainingCV` in `src/data/splitting.py`. Folds
are contiguous time blocks; fold *i* trains on the past and validates on the
next block.

Measured folds (equal-count blocks over the modelling period):

| fold | train rows | train fraud | val rows | val fraud |
|---|---|---|---|---|
| 0 | 54,466 | — | 78,739 | — |
| 1 | 133,979 | 2.5288% | 78,739 | 4.1974% |
| 2 | 214,078 | 3.0517% | 78,738 | 3.9396% |
| 3 | 293,305 | 3.3481% | 78,739 | 3.7085% |
| 4 | 372,880 | 3.4030% | 78,739 | 3.8342% |

Note how validation fraud rate exceeds training fraud rate in every fold — a
structural property of this dataset that random folds would average away.

### R2 — Entity bleed across folds · **partially controlled**

The same card transacts repeatedly, so a random split trains and validates on the
same entity. Time-ordered folds largely prevent this: an entity's later
transactions fall in later blocks. It is *partially* controlled and stated as
such — a card active across a fold boundary still appears on both sides. Full
control would need grouped-and-time-ordered folds, which fragments the time axis.
The purge gap (R3) is what limits the residual effect.

### R3 — Feature values straddling a fold boundary · **controlled**

Subtler than row leakage: a 168-hour trailing count computed for a row at the
start of a validation block reaches back into training rows. The *rows* are
separated but the *feature values* are not. Control: a **7-day purge gap**
(`purge_days: 7` in `configs/config.yaml`), at least as wide as the longest
look-back window (168 h). Asserted in
`tests/test_data_validation.py::test_train_always_precedes_validation_with_a_gap`.

### R4 — Fitting encoders on the full modelling set before CV · **controlled**

Frequency counts and per-entity amount baselines are *population statistics*.
Fitting them once on the whole modelling period and then cross-validating leaks
each fold's validation rows into its own training features.

Control: **encoders are refitted inside every fold** (`src/models/training.py::_fit_one`
calls `FeaturePipeline.fit(train_df)` per fold). This is why the pipeline is split
into three phases — `prepare` / `fit` / `transform` — so the only method that
learns anything takes the training frame as its argument.

### R5 — Frequency encoding fit on train + test combined · **rejected by choice**

Fitting counts over train ∪ test is a well-known Kaggle score-booster and it is
**transductive leakage**: it assumes the scoring population is known at training
time. A deployed API scores one transaction at a time and cannot see the future
population.

Control: counts come from the training partition only
(`src/features/aggregations.py::FrequencyEncoder`); unseen values map to 0,
literally "never observed in the training window". This costs leaderboard points
and is the correct engineering decision for a serving system. Asserted in
`tests/test_features.py::test_frequency_encoder_uses_only_fitted_counts`.

### R6 — Target encoding · **not implemented, deliberately**

Any encoding touching `isFraud` must be computed strictly inside each training
fold. Given R1–R3 already constrain the fold structure, a correct fold-internal
target encoding adds real complexity for a gain that frequency encoding largely
captures. It is omitted rather than implemented carelessly. Nothing in
`src/features/` reads the target.

### R7 — Absolute time as a model input · **controlled**

Every test `TransactionDT` value lies outside the training range, so a tree
routes all test rows into the rightmost leaf — the feature cannot generalise and
actively misleads. Control: raw `TransactionDT` is hard-excluded; only *cyclical*
(`hour_of_day`, `day_of_week`) and *relative* (deltas, trailing windows) time
features are produced. The internal `_day_index` used to build folds and anchors
is dropped before the model sees the frame (`FeaturePipeline._select_feature_columns`
drops everything prefixed `_`).

### R8 — Velocity features · **not leakage, and here is why**

Trailing-window counts look like leakage but are not: each value uses only rows
*strictly earlier in time*, which at inference have already happened. The current
row is excluded by construction — `searchsorted` returns the count of positions
*before* `i` within the window. So velocity can legitimately be computed over the
whole timeline before splitting, and a holdout row may legitimately see earlier
holdout rows.

Verified by four tests in `tests/test_features.py::TestVelocityIsCausal`,
including that the first transaction of an entity has zero history and that
counts exclude the current row (`[0, 1, 2, 3, 4]` for five sequential
transactions).

The contrast with R4/R5 is the whole point: velocity is a *causal* function of
the past; frequency counts are a *population statistic*. The first is safe on the
full frame, the second is not.

### R9 — `D`-column time anchoring · **implemented, then measured to be harmful**

`D1`–`D15` behave like day-deltas from a moving reference (`D1` ∈ [0, 640];
`D4`, `D6`, `D11`, `D12`, `D14`, `D15` go negative — measured minima −122, −83,
−53, −83, −193, −83). If the reference moves, raw values shift with absolute time
and will not match the test period.

`day_index − D_n` converts a moving delta into a fixed calendar anchor.
**Both forms are kept** and the model decides.

The audit flagged this as an unverified hypothesis. It has now been verified, and
**the verdict is negative.**

SHAP ranks `D1_anchored` third overall, which initially looked like confirmation.
Drift monitoring against the real test period says otherwise: **all 15 anchored
features drift significantly** (PSI 3.5–12.4, KS 0.84–1.00, with `D9_anchored` at
KS = 1.000 — completely disjoint distributions), while the raw `D` columns they
derive from remain **stable** (PSI 0.0002–0.108).

The cause is self-inflicted: `day_index` is absolute time, so anchoring reintroduces
exactly the risk documented in R7. The transformation converts a *stable relative*
quantity into an *unstable absolute* one. The holdout does not reveal this because
it sits immediately after the training period, where the ranges still overlap;
the genuine 30-day gap exposes it.

The features remain in the reported model — removing them post hoc would
invalidate every metric in this repository — and the defect is documented in the
README limitations instead, with removal scoped as the top future improvement.
This is what the monitoring layer is for: it found a real flaw in the project's own
feature engineering rather than reporting an all-clear.

### R10 — Measured train↔test covariate shift · **quantified**

Not leakage, but the reason honest validation matters. Missing rates already move
materially between train and the test period:

| feature | train missing | test missing | delta |
|---|---|---|---|
| `D13` | 89.509% | 75.649% | −13.860% |
| `V39`–`V52` (entire block) | 28.613% | 15.168% | −13.445% |

A model leaning on these meets a different world 30 days later.
`scripts/monitor.py` measures this directly against the real test period.

### R11 — Zero-variance-at-scoring columns · **controlled**

`V107` is **constant in `test_transaction.csv` but not in train**. Hard-excluded
(`src/data/schema.py::EXCLUDED_FROM_FEATURES`).

### R12 — The `id-NN` / `id_NN` rename · **not leakage, but silently destructive**

`train_identity.csv` uses `id_01`…`id_38`; `test_identity.csv` uses
`id-01`…`id-38`. **All 38 differ.** Unfixed, a model trained on train and scored
on test sees 38 all-null columns and degrades quietly instead of raising.
Renamed once at the load boundary and then asserted absent
(`validate_identity_rename`).

---

## 3. Hard exclusions

| feature | measured reason |
|---|---|
| `isFraud` | the target |
| `TransactionID` | monotonic, corr 0.998 with time ⇒ absolute-time proxy (R7) |
| `TransactionDT` (raw) | train/test ranges disjoint (R7) |
| `V107` | constant in test (R11) |
| `_day_index`, `_entity_card*` | internal build helpers, dropped by prefix |

Enforced in one place — `EXCLUDED_FROM_FEATURES` — and asserted by
`tests/test_features.py::test_excluded_columns_never_reach_the_model`.

---

## 4. Kept deliberately, against the usual reflex

- **All 339 V columns.** No correlation-clustering before a baseline exists.
  Redundancy is handled by `feature_fraction: 0.6` (column subsampling
  decorrelates trees cheaply) and will be revisited on SHAP evidence, not on a
  prior guess.
- **`dist2` (93.6% missing), `D7` (93.4%), `id_07`/`id_08`/`id_21`–`id_27` (≥96%).**
  A rare signal can still be a strong one for a minority class; dropping on
  missing rate alone is a guess. They stay unless they fail to earn validation
  lift.

---

## 5. The control experiment

Because "random CV is optimistic here" should be measured rather than asserted,
`scripts/train.py --random-cv-control` runs the identical model on random
stratified folds and logs the PR-AUC difference to MLflow as
`random_cv_optimism_pr_auc`. Random folds are generated and persisted
(`folds_random.npz`) purely for this comparison and are **never** used for model
selection.

---

## 6. What the controls cost

Being honest about the trade-off, because every item above has a price:

| control | cost |
|---|---|
| Train-only frequency encoding (R5) | lower leaderboard score than the standard Kaggle approach |
| 7-day purge gap (R3) | ~7 days of training rows discarded per fold |
| Time-ordered folds (R1) | early folds train on as little as 54k rows; higher fold variance |
| No target encoding (R6) | forgoes a feature family that often helps |
| Holdout scored once (§ evaluation) | no opportunity to tune against the final number |

The resulting numbers are lower than a leaderboard-optimised pipeline would
report, and they are the ones that would survive deployment.
