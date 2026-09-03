"""Reproducible seeding.

Called once per entry point. Anything that draws randomness afterwards
(numpy, python's ``random``, LightGBM, scikit-learn) is deterministic given the
same seed and the same data.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed every RNG this project touches and return the seed used."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.debug("Seeded RNGs with %d", seed)
    return seed
