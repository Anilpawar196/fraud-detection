"""Pydantic request/response models.

Input contract: the trained model reads ~430 raw columns, and requiring all of
them would make the API unusable. So the schema names the fields carrying most of
the signal and accepts the long tail (``C*``, ``D*``, ``M*``, ``V*``, ``id_*``)
through one ``extra_features`` map. Anything omitted becomes NaN, which the model
routes natively — the response reports ``features_supplied`` so a caller can see
how complete the input behind a score was.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

RiskLevel = Literal["low", "medium", "high"]


class TransactionRequest(BaseModel):
    """A single transaction to score."""

    model_config = ConfigDict(extra="forbid")

    transaction_amt: Annotated[float, Field(gt=0, le=1_000_000, description="Transaction amount.")]
    product_cd: Annotated[str, Field(min_length=1, max_length=8, description="Product code.")]

    card1: Annotated[int | None, Field(default=None, ge=0, le=100_000)]
    card2: Annotated[float | None, Field(default=None, ge=0, le=10_000)]
    card3: Annotated[float | None, Field(default=None, ge=0, le=10_000)]
    card4: Annotated[str | None, Field(default=None, max_length=32)]
    card5: Annotated[float | None, Field(default=None, ge=0, le=10_000)]
    card6: Annotated[str | None, Field(default=None, max_length=32)]

    addr1: Annotated[float | None, Field(default=None, ge=0, le=10_000)]
    addr2: Annotated[float | None, Field(default=None, ge=0, le=10_000)]
    dist1: Annotated[float | None, Field(default=None, ge=0)]
    dist2: Annotated[float | None, Field(default=None, ge=0)]

    p_emaildomain: Annotated[str | None, Field(default=None, max_length=64)]
    r_emaildomain: Annotated[str | None, Field(default=None, max_length=64)]

    device_type: Annotated[str | None, Field(default=None, max_length=32)]
    device_info: Annotated[str | None, Field(default=None, max_length=128)]

    transaction_dt: Annotated[
        int | None,
        Field(
            default=None,
            ge=0,
            description="Seconds offset in the dataset time base; drives cyclical time features.",
        ),
    ]

    extra_features: Annotated[
        dict[str, float | str | None],
        Field(
            default_factory=dict,
            description="Remaining dataset columns (C*, D*, M*, V*, id_*). Omitted -> NaN.",
        ),
    ]

    @field_validator("extra_features")
    @classmethod
    def _limit_extra_features(
        cls, value: dict[str, float | str | None]
    ) -> dict[str, float | str | None]:
        """Bound the map so one request cannot carry unbounded keys."""
        if len(value) > 600:
            raise ValueError("extra_features may not contain more than 600 keys")
        return value

    def to_raw_record(self) -> dict[str, Any]:
        """Flatten into dataset column names."""
        record: dict[str, Any] = {
            "TransactionAmt": self.transaction_amt,
            "ProductCD": self.product_cd,
            "card1": self.card1,
            "card2": self.card2,
            "card3": self.card3,
            "card4": self.card4,
            "card5": self.card5,
            "card6": self.card6,
            "addr1": self.addr1,
            "addr2": self.addr2,
            "dist1": self.dist1,
            "dist2": self.dist2,
            "P_emaildomain": self.p_emaildomain,
            "R_emaildomain": self.r_emaildomain,
            "DeviceType": self.device_type,
            "DeviceInfo": self.device_info,
        }
        if self.transaction_dt is not None:
            record["TransactionDT"] = self.transaction_dt
        record.update(self.extra_features)
        return record


class PredictionResponse(BaseModel):
    """Scoring result for one transaction."""

    fraud_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    risk_level: RiskLevel
    flagged: bool = Field(description="Whether the probability meets the decision threshold.")
    decision_threshold: float
    model_version: str
    latency_ms: float
    features_supplied: int


class FeatureContributionResponse(BaseModel):
    """One SHAP contribution."""

    feature: str
    value: float | str | None
    shap_value: float
    direction: Literal["increases_risk", "decreases_risk"]


class ExplanationResponse(BaseModel):
    """Prediction plus the factors that drove it."""

    fraud_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    risk_level: RiskLevel
    model_version: str
    base_value: float = Field(description="Model output for an average input, in log-odds.")
    top_factors: list[FeatureContributionResponse]
    latency_ms: float


class HealthResponse(BaseModel):
    """Liveness plus model provenance.

    Provenance is folded in here rather than given its own endpoint: it is the
    same question ("what is running?") and one fewer route to document.
    """

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str
    model_name: str | None = None
    trained_at: str | None = None
    n_features: int | None = None
    calibrated: bool | None = None
    decision_threshold: float | None = None
    holdout_metrics: dict[str, Any] = Field(default_factory=dict)
    uptime_seconds: float
    requests_served: int
