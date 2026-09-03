"""Probability calibration.

Calibration is not cosmetic here. Two things actively distort the raw scores:

1. ``scale_pos_weight`` reweights the loss to counter 27.58:1 imbalance, which
   inflates predicted probabilities away from the true rate.
2. The API returns a ``fraud_probability`` and bands it into low/medium/high at
   fixed thresholds (0.30 / 0.70). Those thresholds are only meaningful if the
   number they compare against is an actual probability.

A model can rank near-perfectly (high PR-AUC) while being badly calibrated, so
ranking metrics alone cannot detect this — hence the Brier score and ECE
reported alongside.

Isotonic regression is used rather than Platt scaling: it is non-parametric, and
the distortion introduced by reweighting is monotone but not sigmoid-shaped, so
a sigmoid fit would leave systematic error behind. Isotonic needs a reasonable
number of positives to be stable, which the validation fold has (~2,700).

The calibrator is deliberately a *thin post-processor on probabilities* rather
than a ``CalibratedClassifierCV`` wrapper. That keeps the serving path explicit
(model -> probability -> calibrator -> probability), makes the artifact trivially
picklable, and avoids re-running the feature pipeline inside a sklearn wrapper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression

from src.evaluation.metrics import expected_calibration_error

logger = logging.getLogger(__name__)


@dataclass
class ProbabilityCalibrator:
    """Monotone recalibration of predicted probabilities.

    Fit on a validation set the model did not train on. Fitting on training
    predictions would learn the model's in-sample overconfidence, which is a
    different distortion from the one present at inference.
    """

    method: str = "isotonic"
    model: IsotonicRegression | None = field(default=None, repr=False)
    ece_before: float | None = None
    ece_after: float | None = None
    brier_before: float | None = None
    brier_after: float | None = None

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> ProbabilityCalibrator:
        """Fit the calibration map and record the improvement it produced."""
        y_true = np.asarray(y_true, dtype="float64")
        y_prob = np.asarray(y_prob, dtype="float64")
        if self.method != "isotonic":
            raise ValueError(f"Unsupported calibration method {self.method!r}")

        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.model.fit(y_prob, y_true)

        calibrated = self.transform(y_prob)
        self.ece_before = expected_calibration_error(y_true, y_prob)
        self.ece_after = expected_calibration_error(y_true, calibrated)
        self.brier_before = float(np.mean((y_prob - y_true) ** 2))
        self.brier_after = float(np.mean((calibrated - y_true) ** 2))
        logger.info(
            "Calibration (%s): ECE %.5f -> %.5f | Brier %.5f -> %.5f",
            self.method,
            self.ece_before,
            self.ece_after,
            self.brier_before,
            self.brier_after,
        )
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply the fitted calibration map."""
        if self.model is None:
            raise RuntimeError("ProbabilityCalibrator.transform called before fit")
        return np.clip(self.model.predict(np.asarray(y_prob, dtype="float64")), 0.0, 1.0)

    @property
    def improved(self) -> bool:
        """Whether calibration actually reduced expected calibration error.

        Checked rather than assumed — if it does not help, the honest action is
        to serve the uncalibrated score and say so.
        """
        if self.ece_before is None or self.ece_after is None:
            return False
        return self.ece_after < self.ece_before
