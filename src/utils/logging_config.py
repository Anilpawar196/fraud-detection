"""Centralised logging setup.

Scripts call :func:`setup_logging` once at entry; library modules only ever do
``logger = logging.getLogger(__name__)`` so that the application controls
handlers and formatting.
"""

from __future__ import annotations

import logging
import os
import sys

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str | int | None = None) -> None:
    """Configure root logging for a script or service entry point.

    Args:
        level: Explicit level. Defaults to the ``LOG_LEVEL`` environment
            variable, falling back to ``INFO``.
    """
    resolved = level or os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=resolved,
        format=_FORMAT,
        datefmt=_DATEFMT,
        stream=sys.stdout,
        force=True,
    )
    # These libraries are extremely chatty at INFO.
    for noisy in ("matplotlib", "numba", "botocore", "urllib3", "git"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
