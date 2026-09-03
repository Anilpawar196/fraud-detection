"""Shared test fixtures.

Tests must run in CI, where the 1.3 GB IEEE-CIS dataset is not present. So the
fixtures synthesise a frame that is *schema-faithful*: the same column families,
dtypes, categorical values, missingness patterns and chronological ordering as
the real data, at a size that trains in seconds.

The synthetic target is deliberately given real signal (driven by amount, a
velocity-like burst pattern and one categorical level) so that model tests can
assert a model learns *something* rather than merely that it runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.schema import SECONDS_PER_DAY

N_ROWS = 3_000
SEED = 7

# Small but representative subsets of each family. Using all 339 V columns would
# slow every test for no additional coverage.
V_COLUMNS = [f"V{i}" for i in range(1, 21)]
C_COLUMNS = [f"C{i}" for i in range(1, 15)]
D_COLUMNS = [f"D{i}" for i in range(1, 16)]
M_COLUMNS = [f"M{i}" for i in range(1, 10)]
ID_NUMERIC = [f"id_{i:02d}" for i in (1, 2, 3, 5, 11, 13, 17, 19, 20, 32)]
ID_CATEGORICAL = [
    "id_12",
    "id_15",
    "id_16",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_33",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
]


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    """Synthetic joined transaction+identity frame, sorted chronologically."""
    generator = np.random.default_rng(SEED)
    n = N_ROWS

    # Timestamps spanning 60 days, sorted, with deliberate ties (the real data
    # has 5.746% of rows sharing a timestamp).
    timestamps = np.sort(generator.integers(SECONDS_PER_DAY, SECONDS_PER_DAY * 60, size=n))

    amount = np.round(np.exp(generator.normal(4.2, 1.0, size=n)), 2)
    card1 = generator.integers(1000, 18396, size=n)
    # Force repeated entities so velocity features have history to find.
    card1[: n // 3] = generator.choice([1234, 5678, 9012], size=n // 3)

    frame = pd.DataFrame(
        {
            "TransactionID": np.arange(2_987_000, 2_987_000 + n, dtype="int64"),
            "TransactionDT": timestamps.astype("int32"),
            "TransactionAmt": amount.astype("float32"),
            "ProductCD": generator.choice(["W", "C", "R", "H", "S"], size=n),
            "card1": card1.astype("float32"),
            "card2": generator.choice([*range(100, 601, 25), np.nan], size=n).astype("float32"),
            "card3": generator.choice([150.0, 185.0, np.nan], size=n).astype("float32"),
            "card4": generator.choice(["visa", "mastercard", "discover", None], size=n),
            "card5": generator.choice([102.0, 142.0, 166.0, np.nan], size=n).astype("float32"),
            "card6": generator.choice(["debit", "credit", None], size=n),
            "addr1": generator.choice([*range(100, 541, 10), np.nan], size=n).astype("float32"),
            "addr2": generator.choice([87.0, 60.0, np.nan], size=n).astype("float32"),
            "dist1": generator.choice([*range(0, 200), np.nan], size=n).astype("float32"),
            "dist2": np.where(
                generator.random(n) < 0.94, np.nan, generator.integers(0, 500, size=n)
            ).astype("float32"),
            "P_emaildomain": generator.choice(
                ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", None], size=n
            ),
            "R_emaildomain": np.where(
                generator.random(n) < 0.77,
                None,
                generator.choice(["gmail.com", "yahoo.com", "anonymous.com"], size=n),
            ),
            "DeviceType": generator.choice(["mobile", "desktop", None], size=n),
            "DeviceInfo": generator.choice(
                ["Windows", "iOS Device", "MacOS", "SAMSUNG SM-G892A Build/NRD90M", None], size=n
            ),
        }
    )

    # C columns: never missing in the real data.
    for column in C_COLUMNS:
        frame[column] = generator.integers(0, 20, size=n).astype("float32")

    # D columns: day deltas, ~30-90% missing, some negative.
    for i, column in enumerate(D_COLUMNS):
        missing_rate = 0.2 + 0.05 * i
        values = generator.integers(-80, 640, size=n).astype("float32")
        values[generator.random(n) < missing_rate] = np.nan
        frame[column] = values

    # M columns: T/F categoricals, ~30-60% missing.
    for i, column in enumerate(M_COLUMNS):
        choices = ["M0", "M1", "M2"] if column == "M4" else ["T", "F"]
        values = generator.choice(choices, size=n).astype(object)
        values[generator.random(n) < (0.3 + 0.03 * i)] = None
        frame[column] = values

    # V columns: correlated blocks with a shared missing pattern, as in the real data.
    block_missing = generator.random(n) < 0.28
    for column in V_COLUMNS:
        values = generator.integers(0, 10, size=n).astype("float32")
        values[block_missing] = np.nan
        frame[column] = values

    # Identity block: present for ~25% of rows, matching measured coverage.
    identity_present = generator.random(n) < 0.25
    for column in ID_NUMERIC:
        values = generator.normal(100, 30, size=n).astype("float32")
        values[~identity_present] = np.nan
        frame[column] = values
    for column in ID_CATEGORICAL:
        if column == "id_30":
            pool = ["Android 7.0", "iOS 11.1.2", "Windows 10"]
        elif column == "id_31":
            pool = ["chrome 62.0", "mobile safari 11.0", "samsung browser 6.2"]
        elif column == "id_33":
            pool = ["1920x1080", "1334x750", "2220x1080"]
        elif column == "id_34":
            pool = ["match_status:1", "match_status:2"]
        else:
            pool = ["T", "F", "Found", "NotFound", "New"]
        values = generator.choice(pool, size=n).astype(object)
        values[~identity_present] = None
        frame[column] = values

    frame["identity_present"] = identity_present.astype("int8")

    # Target with genuine signal: large amounts, thin records and one product code
    # raise risk. Keeps prevalence near the real 3.5%.
    logit = (
        -3.9
        + 0.55 * (np.log1p(amount) - np.log1p(amount).mean())
        + 0.9 * (frame["ProductCD"] == "C").to_numpy()
        - 0.6 * identity_present
    )
    probability = 1 / (1 + np.exp(-logit))
    frame["isFraud"] = (generator.random(n) < probability).astype("int8")

    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].astype("category")

    return frame.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)


@pytest.fixture
def prepared_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """Frame with causal features applied, ready for pipeline fit/transform."""
    from src.features.pipeline import FeaturePipeline

    return FeaturePipeline().prepare(raw_frame.copy())


@pytest.fixture
def fitted_pipeline(prepared_frame: pd.DataFrame):
    """Feature pipeline fitted on the first 70% (by time) of the prepared frame."""
    from src.features.pipeline import FeaturePipeline

    cut = int(len(prepared_frame) * 0.7)
    pipeline = FeaturePipeline()
    pipeline.fit(prepared_frame.iloc[:cut])
    return pipeline


@pytest.fixture
def trained_artifact(prepared_frame: pd.DataFrame):
    """A real, small ModelArtifact — trained here so serving tests are end-to-end."""
    import lightgbm as lgb

    from src.evaluation.calibration import ProbabilityCalibrator
    from src.features.pipeline import FeaturePipeline
    from src.models.artifact import ArtifactMetadata, ModelArtifact, utc_now_iso

    cut = int(len(prepared_frame) * 0.7)
    train_df = prepared_frame.iloc[:cut]
    validation_df = prepared_frame.iloc[cut:]

    pipeline = FeaturePipeline()
    pipeline.fit(train_df)
    X_train = pipeline.transform(train_df)
    y_train = train_df["isFraud"].to_numpy()

    model = lgb.LGBMClassifier(
        n_estimators=40,
        num_leaves=15,
        learning_rate=0.1,
        min_child_samples=20,
        random_state=SEED,
        n_jobs=1,
        verbose=-1,
    )
    model.fit(X_train, y_train, categorical_feature=pipeline.categorical_features)

    validation_probabilities = model.predict_proba(pipeline.transform(validation_df))[:, 1]
    calibrator = None
    if validation_df["isFraud"].nunique() > 1:
        candidate = ProbabilityCalibrator().fit(
            validation_df["isFraud"].to_numpy(), validation_probabilities
        )
        calibrator = candidate if candidate.improved else None

    raw_columns = [
        c for c in prepared_frame.columns if c in set(_raw_column_names()) and c != "isFraud"
    ]

    return ModelArtifact(
        model=model,
        feature_pipeline=pipeline,
        calibrator=calibrator,
        decision_threshold=0.5,
        metadata=ArtifactMetadata(
            model_name="lightgbm",
            trained_at=utc_now_iso(),
            seed=SEED,
            n_features=len(pipeline.feature_names),
            n_train_rows=len(train_df),
            dataset_rows_total=len(prepared_frame),
            holdout_cut_dt=int(prepared_frame["TransactionDT"].max()),
            raw_input_columns=raw_columns,
        ),
    )


def _raw_column_names() -> list[str]:
    """Raw column names present in the synthetic frame."""
    base = [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "addr2",
        "dist1",
        "dist2",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
        "DeviceInfo",
        "identity_present",
    ]
    return base + V_COLUMNS + C_COLUMNS + D_COLUMNS + M_COLUMNS + ID_NUMERIC + ID_CATEGORICAL


@pytest.fixture
def api_client(trained_artifact):
    """TestClient with a real (small) trained model injected.

    The model is injected rather than loaded from disk because CI has no trained
    artifact. Everything downstream of that — validation, frame construction,
    feature pipeline, model, SHAP — is the real code path.
    """
    from fastapi.testclient import TestClient

    from api import dependencies, routes

    dependencies.state.artifact = trained_artifact
    dependencies.state.request_count = 0
    routes.reset_explainer()

    from api.main import app

    # The lifespan handler runs and finds no artifact on disk; state is
    # re-injected afterwards so the tests exercise a loaded model.
    with TestClient(app) as client:
        dependencies.state.artifact = trained_artifact
        yield client


@pytest.fixture
def valid_transaction_payload() -> dict:
    """A minimal, valid /predict request body."""
    return {
        "transaction_amt": 149.99,
        "product_cd": "W",
        "card1": 13926,
        "card2": 404.0,
        "card3": 150.0,
        "card4": "visa",
        "card5": 142.0,
        "card6": "debit",
        "addr1": 315.0,
        "addr2": 87.0,
        "p_emaildomain": "gmail.com",
        "device_type": "mobile",
        "device_info": "iOS Device",
        "extra_features": {"C1": 1.0, "C2": 1.0, "D1": 14.0, "M1": "T", "V1": 1.0},
    }
