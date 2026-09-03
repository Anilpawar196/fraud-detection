# Serving image for the fraud detection API.
#
# Multi-stage: wheels are built in a stage carrying a compiler, and the runtime
# stage installs only prebuilt wheels, keeping gcc and build headers out of the
# shipped image.

# ---------- build stage ----------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /wheels
COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt

# ---------- runtime stage ----------
FROM python:3.12-slim AS runtime

# FRAUD_PROJECT_ROOT points the path resolver at the image layout, so no module
# needs an absolute path.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    FRAUD_PROJECT_ROOT=/app \
    PYTHONPATH=/app

# libgomp1 is a runtime requirement of LightGBM's OpenMP threading.
# curl is used by the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Application code only; data and notebooks are excluded via .dockerignore.
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/

# The model artifact is mounted at runtime rather than baked in: a 52 MB binary
# in the image would force a rebuild on every retrain.
RUN mkdir -p /app/models

RUN useradd --create-home --shell /bin/bash --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# /health returns 200 with status="degraded" when no model is mounted, so the
# container is reported healthy as a process even before a model exists.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
