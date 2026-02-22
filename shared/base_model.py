"""
Abstract base model for all competition plans.

Defines the contract that every plan must implement: ``train()``,
``predict()``, and ``run_cv()``.  Handles the boilerplate of loading data,
engineering features, running cross-validation, generating submissions, and
enforcing probability hierarchy.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from shared.calibration import Calibrator, CalibrationMethod
from shared.constants import (
    CV_STRATEGY_DEFAULT,
    CV_TIME_CUTOFF_DEFAULT,
    ID_COL,
    RANDOM_SEED,
    TARGET_COLS,
)
from shared.data_loader import DataLoader
from shared.feature_engineering import FeatureEngineer
from shared.submission import SubmissionGenerator
from shared.validation import CVResult, Validator

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Template for all competition models.

    Subclasses must implement:
        - ``_train_single_target``
        - ``_predict_single_target``

    The base class orchestrates the full pipeline:
        data → features → per-target training → calibration → hierarchy
        → submission.

    Parameters
    ----------
    plan_name : str
        Human-readable name for logging and file naming.
    calibration_method : CalibrationMethod
        How to calibrate predicted probabilities.
    n_cv_folds : int
        Number of folds for stratified CV.
    """

    def __init__(
        self,
        plan_name: str,
        calibration_method: CalibrationMethod = CalibrationMethod.ISOTONIC,
        n_cv_folds: int = 5,
        cv_strategy: str = CV_STRATEGY_DEFAULT,
        cv_time_cutoff: str = CV_TIME_CUTOFF_DEFAULT,
    ) -> None:
        self.plan_name = plan_name
        self.calibration_method = calibration_method
        self.n_cv_folds = n_cv_folds
        self.cv_strategy = cv_strategy
        self.cv_time_cutoff = cv_time_cutoff

        # State populated during run()
        self.loader = DataLoader()
        self.validator = Validator(n_splits=n_cv_folds)
        self._models: dict[str, object] = {}
        self._calibrators: dict[str, Calibrator] = {}

    # ── Abstract interface ─────────────────────────────────────────────

    @abstractmethod
    def _train_single_target(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        target_name: str = "",
    ) -> object:
        """Train a model for one target column.

        Must return a fitted model object that can be passed to
        ``_predict_single_target``.
        """

    @abstractmethod
    def _predict_single_target(
        self, model: object, X: pd.DataFrame
    ) -> np.ndarray:
        """Return predicted probabilities from a fitted model."""

    # ── Template pipeline ──────────────────────────────────────────────

    def run(self, submission_filename: str | None = None) -> Path:
        """Execute the full pipeline: load → features → CV → train → submit.

        Returns the path to the generated submission file.
        """
        logger.info("=" * 60)
        logger.info("Running %s", self.plan_name)
        logger.info("=" * 60)

        # 1. Load data
        train_df, test_df, prior_df, sample_sub = self.loader.load_all()

        # 2. Feature engineering
        fe = FeatureEngineer(prior_df)
        fe.fit(train_df)
        X_train = fe.transform(train_df)
        X_test = fe.transform(test_df)

        # 3. Cross-validation + training per target
        cv_result = CVResult()
        predictions_ll: dict[str, np.ndarray] = {}

        for target in TARGET_COLS:
            y = train_df[target]
            logger.info("── Target: %s (pos_rate=%.4f) ──", target, y.mean())

            # CV evaluation
            oof_preds = np.zeros(len(y))
            oof_mask = np.zeros(len(y), dtype=bool)
            splits = list(
                self.validator.cv_splits(
                    train_df,
                    y,
                    strategy=self.cv_strategy,
                    cutoff=self.cv_time_cutoff,
                )
            )
            for fold_idx, (tr_idx, va_idx) in enumerate(splits):
                logger.info("  Fold %d: training on %d samples, validating on %d", fold_idx, len(tr_idx), len(va_idx))
                X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
                y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

                fold_model = self._train_single_target(
                    X_tr, y_tr, X_va, y_va, target_name=target
                )
                oof_preds[va_idx] = self._predict_single_target(fold_model, X_va)
                oof_mask[va_idx] = True

                fold_result = self.validator.evaluate(
                    y_va.values, oof_preds[va_idx], target, fold_idx
                )
                cv_result.add(fold_result)

            # 4. Calibrate on OOF predictions
            calibrator = Calibrator(self.calibration_method)
            calibrator.fit(y.values[oof_mask], oof_preds[oof_mask])
            self._calibrators[target] = calibrator

            # 5. Train final model on all training data
            logger.info("  Training final model on all %d training samples...", len(X_train))
            final_model = self._train_single_target(
                X_train, y, target_name=target
            )
            self._models[target] = final_model

            # 6. Predict on test
            raw_preds = self._predict_single_target(final_model, X_test)
            predictions_ll[target] = calibrator.transform(raw_preds)

        # 7. Enforce hierarchy P(7d) ≤ P(90d) ≤ P(120d)
        p07, p90, p120 = Calibrator.enforce_hierarchy(
            predictions_ll[TARGET_COLS[0]],
            predictions_ll[TARGET_COLS[1]],
            predictions_ll[TARGET_COLS[2]],
        )
        predictions_ll[TARGET_COLS[0]] = p07
        predictions_ll[TARGET_COLS[1]] = p90
        predictions_ll[TARGET_COLS[2]] = p120

        # 8. Log CV summary
        summary = cv_result.summary()
        logger.info("\n%s CV Summary:\n%s", self.plan_name, summary.to_string())

        # 9. Generate submission
        filename = submission_filename or f"{self.plan_name}_submission.csv"
        gen = SubmissionGenerator(sample_sub)
        path = gen.generate(
            test_ids=test_df[ID_COL],
            predictions=predictions_ll,
        )

        logger.info("✓ %s complete → %s", self.plan_name, path)
        return path
