"""FastAPI application entry point.

The model is loaded once in the lifespan handler and reused for every request.
Startup deliberately does not fail when the artifact is missing: the service
comes up, `/health` reports `degraded`, and the prediction routes return 503.
That is far easier to diagnose in a container than a crash-loop with no HTTP
surface.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.dependencies import load_artifact, state
from api.routes import router
from api.settings import get_settings
from src.utils.logging_config import setup_logging

logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Load the model once, at startup."""
    settings = get_settings()
    setup_logging(settings.log_level)
    state.settings = settings

    try:
        state.artifact = load_artifact(settings)
        logger.info(
            "Model loaded: %s trained %s, %d features, threshold %.4f",
            state.artifact.metadata.model_name,
            state.artifact.metadata.trained_at,
            state.artifact.metadata.n_features,
            state.artifact.decision_threshold,
        )
    except FileNotFoundError:
        logger.error(
            "No model artifact at %s — /health will report degraded and prediction "
            "routes will return 503. Run: python scripts/train.py",
            settings.model_artifact_path,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Model load failed — starting in degraded mode")

    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Leakage-Safe Fraud Detection API",
    description=(
        "Serves a LightGBM fraud model trained on the IEEE-CIS dataset with "
        "leakage-safe temporal validation. Returns calibrated probabilities and "
        "SHAP explanations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return a structured 422 rather than FastAPI's default shape."""
    logger.warning("Validation failure on %s: %s", request.url.path, exc.errors()[:3])
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Request validation failed",
            "error_type": "RequestValidationError",
            "errors": [
                {"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]}
                for e in exc.errors()[:10]
            ],
        },
    )


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    """Point callers at the docs."""
    return {"service": "fraud-detection-api", "docs": "/docs", "health": "/health"}
