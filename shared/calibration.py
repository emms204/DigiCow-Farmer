"""
Probability calibration utilities.

Since the competition metric is 75 % Log Loss, well-calibrated probabilities
are critical.  This module provides isotonic regression and Platt scaling
wrappers, plus a hierarchical enforcement step that guarantees:

    P(7 days) ≤ P(90 days) ≤ P(120 days)
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from shared.constants import PROB_CLIP_MAX, PROB_CLIP_MIN, RANDOM_SEED

logger = logging.getLogger(__name__)


class CalibrationMethod(Enum):
    """Supported calibration strategies."""

    ISOTONIC = auto()
    PLATT = auto()
    NONE = auto()


class Calibrator:
    """Fit a calibration mapping on validation data and apply it to test.

    Parameters
    ----------
    method : CalibrationMethod
        ``ISOTONIC`` for non-parametric monotonic regression (flexible but
        can overfit on small datasets).
        ``PLATT`` for parametric sigmoid scaling (more stable).
        ``NONE`` to skip calibration (only clip and enforce hierarchy).

    Example
    -------
    >>> cal = Calibrator(CalibrationMethod.ISOTONIC)
    >>> cal.fit(y_val_true, y_val_prob)
    >>> calibrated = cal.transform(y_test_prob)
    """

    def __init__(self, method: CalibrationMethod = CalibrationMethod.ISOTONIC) -> None:
        self.method = method
        self._model: Optional[IsotonicRegression | LogisticRegression] = None

    # ── Public API ─────────────────────────────────────────────────────

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> "Calibrator":
        """Learn the calibration mapping from (predicted_prob, true_label).

        Parameters
        ----------
        y_true : array of {0, 1}, shape (n,)
        y_prob : array of floats, shape (n,)

        Returns
        -------
        self
        """
        if self.method == CalibrationMethod.ISOTONIC:
            self._model = IsotonicRegression(
                y_min=PROB_CLIP_MIN, y_max=PROB_CLIP_MAX, out_of_bounds="clip"
            )
            self._model.fit(y_prob, y_true)

        elif self.method == CalibrationMethod.PLATT:
            self._model = LogisticRegression(random_state=RANDOM_SEED)
            self._model.fit(y_prob.reshape(-1, 1), y_true)

        logger.info("Calibrator fitted with method=%s", self.method.name)
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply the learned calibration mapping.

        Complexity: O(n) for isotonic (binary-search interpolation),
        O(n) for Platt (sigmoid evaluation).
        """
        if self.method == CalibrationMethod.NONE or self._model is None:
            return np.clip(y_prob, PROB_CLIP_MIN, PROB_CLIP_MAX)

        if self.method == CalibrationMethod.ISOTONIC:
            result = self._model.transform(y_prob)
        else:  # PLATT
            result = self._model.predict_proba(y_prob.reshape(-1, 1))[:, 1]

        return np.clip(result, PROB_CLIP_MIN, PROB_CLIP_MAX)

    # ── Hierarchical enforcement ───────────────────────────────────────

    @staticmethod
    def enforce_hierarchy(
        p07: np.ndarray, p90: np.ndarray, p120: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Guarantee that P(7d) ≤ P(90d) ≤ P(120d) element-wise.

        Strategy: anchor on the 120-day prediction (widest window, most
        positive samples) and pull shorter windows down where violated.

        Complexity: O(n) — three vectorised np.minimum calls.

        Parameters
        ----------
        p07, p90, p120 : np.ndarray
            Raw or calibrated probabilities for each target.

        Returns
        -------
        p07, p90, p120 : tuple of np.ndarray
            Adjusted probabilities satisfying the ordering constraint.
        """
        # Ensure p90 ≤ p120
        p90 = np.minimum(p90, p120)
        # Ensure p07 ≤ p90
        p07 = np.minimum(p07, p90)

        return p07, p90, p120

