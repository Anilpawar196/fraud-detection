"""Project path resolution.

All paths in this project are derived from the repository root, which is
located relative to *this file*. Nothing anywhere should contain an absolute
path, so the project runs identically on a laptop and inside a container.

The root can be overridden with the ``FRAUD_PROJECT_ROOT`` environment
variable, which is how the Docker image points at ``/app``.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/utils/paths.py -> src/utils -> src -> <repo root>
_DEFAULT_ROOT = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    """Return the repository root directory."""
    env_root = os.getenv("FRAUD_PROJECT_ROOT")
    return Path(env_root).resolve() if env_root else _DEFAULT_ROOT


ROOT: Path = project_root()

DATA_DIR: Path = ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"

CONFIG_DIR: Path = ROOT / "configs"
MODELS_DIR: Path = ROOT / "models"
REPORTS_DIR: Path = ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if missing, then return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
