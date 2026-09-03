"""The three API routes: /health, /predict, /explain.

Previously these were split across four router modules alongside batch scoring,
a model-info endpoint and monitoring endpoints backed by Postgres. Three routes
do not need four files, and endpoints that only existed to look complete were
removed.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import (
    AppState,
    build_prepared_frame,
    count_supplied_features,
    get_state,
    require_artifact,
)
from api.schemas import (
    ExplanationResponse,
    FeatureContributionResponse,
    HealthResponse,
    PredictionResponse,
    TransactionRequest,
)
from src.explainability.shap_explainer import ShapExplainer
from src.models.artifact import ModelArtifact

logger = logging.getLogger(__name__)

router = APIRouter()

#: Built once on first /explain call. Constructing a TreeExplainer walks the
#: whole ensemble, which is far too slow to repeat per request.
_explainer: ShapExplainer | None = None


def get_explainer(artifact: ModelArtifact) -> ShapExplainer:
    """Lazily construct and cache the SHAP explainer."""
    global _explainer
    if _explainer is None:
        _explainer = ShapExplainer(artifact.model, artifact.feature_pipeline.feature_names)
        logger.info("Initialised SHAP explainer")
    return _explainer


def reset_explainer() -> None:
    """Drop the cached explainer (used when the model reloads, and by tests)."""
    global _explainer
    _explainer = None


@router.get("/health", response_model=HealthResponse)
def health(state: AppState = Depends(get_state)) -> HealthResponse:
    """Liveness plus what model is loaded and how it performed.

    Returns 200 with ``status="degraded"`` when no model is present, rather than
    failing: the distinction between "process is up but has no model" and
    "process is down" is exactly what a health check should make visible.
    """
    artifact = state.artifact
    if artifact is None:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            model_version=state.model_version,
            uptime_seconds=round(state.uptime_seconds, 1),
            requests_served=state.request_count,
        )

    metadata = artifact.metadata
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=state.model_version,
        model_name=metadata.model_name,
        trained_at=metadata.trained_at,
        n_features=metadata.n_features,
        calibrated=metadata.calibrated,
        decision_threshold=artifact.decision_threshold,
        holdout_metrics={
            k: v
            for k, v in (metadata.holdout_metrics or {}).items()
            if k in {"pr_auc", "roc_auc", "precision", "recall", "f1", "pr_auc_lift"}
        },
        uptime_seconds=round(state.uptime_seconds, 1),
        requests_served=state.request_count,
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(
    request: TransactionRequest,
    artifact: ModelArtifact = Depends(require_artifact),
    state: AppState = Depends(get_state),
) -> PredictionResponse:
    """Score a single transaction and return a calibrated fraud probability."""
    started = time.perf_counter()
    record = request.to_raw_record()
    state.request_count += 1

    try:
        prepared = build_prepared_frame(
            [record], artifact, default_timestamp=artifact.metadata.holdout_cut_dt
        )
        probability = float(artifact.predict_proba(prepared)[0])
    except HTTPException:
        raise
    except Exception as error:  # noqa: BLE001
        logger.exception("Scoring failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring failed: {type(error).__name__}",
        ) from error

    risk = artifact.risk_level(
        probability, state.settings.risk_threshold_medium, state.settings.risk_threshold_high
    )
    return PredictionResponse(
        fraud_probability=round(probability, 6),
        risk_level=risk,  # type: ignore[arg-type]
        flagged=probability >= artifact.decision_threshold,
        decision_threshold=artifact.decision_threshold,
        model_version=state.model_version,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        features_supplied=count_supplied_features(record),
    )


@router.post("/explain", response_model=ExplanationResponse)
def explain(
    request: TransactionRequest,
    top_n: int = 10,
    artifact: ModelArtifact = Depends(require_artifact),
    state: AppState = Depends(get_state),
) -> ExplanationResponse:
    """Score a transaction and return the factors that drove the score.

    An analyst cannot action "0.87"; they can action "0.87, driven by a card with
    no prior history and an unusual amount for this account". TreeSHAP is exact
    for tree ensembles and needs no background sample, which is what makes this
    viable inside a request rather than as a batch job.

    The probability is calibrated while ``base_value`` and the SHAP values are in
    the model's log-odds space, so contributions sum to the raw margin rather
    than to the returned probability.
    """
    if not 1 <= top_n <= 50:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="top_n must be between 1 and 50",
        )
    if artifact.linear_preprocessor is not None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="SHAP explanations are implemented for the tree model only.",
        )

    started = time.perf_counter()
    record = request.to_raw_record()
    state.request_count += 1

    try:
        prepared = build_prepared_frame(
            [record], artifact, default_timestamp=artifact.metadata.holdout_cut_dt
        )
        features = artifact.transform_features(prepared)
        probability = float(artifact.predict_proba(prepared)[0])
        explainer = get_explainer(artifact)
        contributions = explainer.explain_row(features, row=0, top_n=top_n)
        base_value = explainer.base_value
    except HTTPException:
        raise
    except Exception as error:  # noqa: BLE001
        logger.exception("Explanation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {type(error).__name__}",
        ) from error

    risk = artifact.risk_level(
        probability, state.settings.risk_threshold_medium, state.settings.risk_threshold_high
    )
    return ExplanationResponse(
        fraud_probability=round(probability, 6),
        risk_level=risk,  # type: ignore[arg-type]
        model_version=state.model_version,
        base_value=round(base_value, 6),
        top_factors=[
            FeatureContributionResponse(**contribution.to_dict()) for contribution in contributions
        ],
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )
