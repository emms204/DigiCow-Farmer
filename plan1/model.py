"""
Plan 1 — LightGBM model with rich feature engineering and calibration.

Uses the shared ``BaseModel`` template with LightGBM as the core learner.
Each target gets its own independently trained LightGBM model with early
stopping on a validation fold, then final predictions are calibrated via
isotonic regression and enforced to satisfy the hierarchical constraint.
"""

from __future__ import annotations

import logging
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from plan1.config import EARLY_STOPPING_ROUNDS, LGBM_PARAMS, NUM_BOOST_ROUND
from shared.base_model import BaseModel
from shared.calibration import CalibrationMethod

logger = logging.getLogger(__name__)


class LGBMModel(BaseModel):
    """LightGBM-based model for Plan 1.

    Inherits the full pipeline from ``BaseModel`` and implements the
    LightGBM-specific training and prediction logic.

    Parameters
    ----------
    params : dict, optional
        LightGBM parameters.  Defaults to ``LGBM_PARAMS``.
    num_boost_round : int
        Maximum number of boosting rounds.
    early_stopping_rounds : int
        Patience for early stopping when a validation set is provided.

    Example
    -------
    >>> model = LGBMModel()
    >>> submission_path = model.run()
    """

    def __init__(
        self,
        params: dict | None = None,
        num_boost_round: int = NUM_BOOST_ROUND,
        early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    ) -> None:
        super().__init__(
            plan_name="plan1_lgbm",
            calibration_method=CalibrationMethod.ISOTONIC,
            n_cv_folds=5,
        )
        self.params = params or LGBM_PARAMS.copy()
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

    # ── Core interface ─────────────────────────────────────────────────

    def _train_single_target(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        target_name: str = "",
    ) -> lgb.Booster:
        """Train a LightGBM booster for one target.

        When a validation set is supplied (during CV), early stopping
        prevents overfitting.  For the final model (no val set), the full
        ``num_boost_round`` iterations are used.

        Complexity: O(n * d * T) where n = rows, d = features, T = trees.
        """
        train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)

        callbacks = [lgb.log_evaluation(period=100)]  # log every 100 iterations
        valid_sets = [train_set]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_set = lgb.Dataset(X_val, label=y_val, free_raw_data=False)
            valid_sets.append(val_set)
            valid_names.append("valid")
            callbacks.append(
                lgb.early_stopping(self.early_stopping_rounds, verbose=True)
            )

        logger.debug("Training LightGBM: %d samples, %d features", len(X_train), X_train.shape[1])
        booster = lgb.train(
            params=self.params,
            train_set=train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        logger.debug(
            "%s | %s — best iteration: %d",
            self.plan_name,
            target_name,
            booster.best_iteration,
        )
        return booster

    def _predict_single_target(
        self, model: object, X: pd.DataFrame
    ) -> np.ndarray:
        """Return predicted probabilities from a trained LightGBM booster.

        Complexity: O(n * T) — each sample traverses T trees.
        """
        booster: lgb.Booster = model  # type: ignore[assignment]
        return booster.predict(X, num_iteration=booster.best_iteration)

