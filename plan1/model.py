"""
Plan 1 — LightGBM model with rich feature engineering and calibration.

Trains on LogLoss and AUC as separate columns (dual optimisation like Plan 4):
  - Model A (LogLoss): conservative, calibrated → LogLoss columns.
  - Model B (AUC):      aggressive, ranking-focused → AUC columns.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from plan1.config import EARLY_STOPPING_ROUNDS, LGBM_PARAMS, NUM_BOOST_ROUND
from plan4.config import AUC_PARAMS, LOGLOSS_PARAMS
from shared.base_model import BaseModel
from shared.calibration import Calibrator, CalibrationMethod, get_default_calibration_method
from shared.constants import ID_COL, TARGET_COLS
from shared.data_loader import DataLoader
from shared.feature_engineering import FeatureEngineer
from shared.submission import SubmissionGenerator
from shared.validation import CVResult, Validator

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

    # ── Dual optimisation (LogLoss + AUC columns) ────────────────────────

    def run(self, submission_filename: str | None = None) -> Path:
        """Execute dual-optimisation pipeline: train LogLoss and AUC models per target."""
        from plan4.config import EARLY_STOPPING_ROUNDS as P4_ESR
        from plan4.config import N_CV_FOLDS, NUM_BOOST_ROUND as P4_NBR

        logger.info("=" * 60)
        logger.info("Running %s (dual: LogLoss + AUC)", self.plan_name)
        logger.info("=" * 60)

        train_df, test_df, prior_df, sample_sub = self.loader.load_all()
        fe = FeatureEngineer(prior_df)
        fe.fit(train_df)
        X_train = fe.transform(train_df)
        X_test = fe.transform(test_df)

        validator = Validator(n_splits=N_CV_FOLDS)
        cv_result = CVResult()
        preds_logloss: dict[str, np.ndarray] = {}
        preds_auc: dict[str, np.ndarray] = {}

        for target in TARGET_COLS:
            y = train_df[target]
            logger.info("── Target: %s (pos_rate=%.4f) ──", target, y.mean())
            splits = list(
                validator.cv_splits(
                    train_df,
                    y,
                    strategy=self.cv_strategy,
                    cutoff=self.cv_time_cutoff,
                )
            )
            n_folds_eff = len(splits)

            # Model A: LogLoss-optimised
            logger.info("  Training Model A (LogLoss-optimised) …")
            oof_ll = np.zeros(len(y))
            oof_ll_mask = np.zeros(len(y), dtype=bool)
            test_ll = np.zeros(len(X_test))
            for fold_idx, (tr_idx, va_idx) in enumerate(splits):
                booster = self._train_lgbm(
                    LOGLOSS_PARAMS,
                    X_train.iloc[tr_idx],
                    y.iloc[tr_idx],
                    X_train.iloc[va_idx],
                    y.iloc[va_idx],
                    num_boost_round=P4_NBR,
                    early_stopping_rounds=P4_ESR,
                )
                oof_ll[va_idx] = booster.predict(
                    X_train.iloc[va_idx], num_iteration=booster.best_iteration
                )
                oof_ll_mask[va_idx] = True
                test_ll += (
                    booster.predict(X_test, num_iteration=booster.best_iteration)
                    / n_folds_eff
                )
            ll_result = validator.evaluate(
                y.values[oof_ll_mask], oof_ll[oof_ll_mask], f"{target}_LL", fold=0
            )
            cv_result.add(ll_result)
            cal_ll = Calibrator(get_default_calibration_method())
            cal_ll.fit(y.values[oof_ll_mask], oof_ll[oof_ll_mask])
            preds_logloss[target] = cal_ll.transform(test_ll)

            # Model B: AUC-optimised
            logger.info("  Training Model B (AUC-optimised) …")
            oof_auc_arr = np.zeros(len(y))
            oof_auc_mask = np.zeros(len(y), dtype=bool)
            test_auc = np.zeros(len(X_test))
            for fold_idx, (tr_idx, va_idx) in enumerate(splits):
                booster = self._train_lgbm(
                    AUC_PARAMS,
                    X_train.iloc[tr_idx],
                    y.iloc[tr_idx],
                    X_train.iloc[va_idx],
                    y.iloc[va_idx],
                    num_boost_round=P4_NBR,
                    early_stopping_rounds=P4_ESR,
                )
                oof_auc_arr[va_idx] = booster.predict(
                    X_train.iloc[va_idx], num_iteration=booster.best_iteration
                )
                oof_auc_mask[va_idx] = True
                test_auc += (
                    booster.predict(X_test, num_iteration=booster.best_iteration)
                    / n_folds_eff
                )
            auc_result = validator.evaluate(
                y.values[oof_auc_mask], oof_auc_arr[oof_auc_mask], f"{target}_AUC", fold=0
            )
            cv_result.add(auc_result)
            preds_auc[target] = np.clip(test_auc, 0.001, 0.999)

        for pred_dict in (preds_logloss, preds_auc):
            p07, p90, p120 = Calibrator.enforce_hierarchy(
                pred_dict[TARGET_COLS[0]],
                pred_dict[TARGET_COLS[1]],
                pred_dict[TARGET_COLS[2]],
            )
            pred_dict[TARGET_COLS[0]] = p07
            pred_dict[TARGET_COLS[1]] = p90
            pred_dict[TARGET_COLS[2]] = p120

        summary = cv_result.summary()
        logger.info("\n%s CV Summary:\n%s", self.plan_name, summary.to_string())

        filename = submission_filename or f"{self.plan_name}_submission.csv"
        gen = SubmissionGenerator(sample_sub)
        path = gen.generate(
            test_ids=test_df[ID_COL],
            predictions=preds_logloss,
            predictions_auc=preds_auc,
            filename=filename,
        )
        logger.info("✓ %s complete → %s", self.plan_name, path)
        return path

    @staticmethod
    def _train_lgbm(
        params: dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        num_boost_round: int = NUM_BOOST_ROUND,
        early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    ) -> lgb.Booster:
        """Train a LightGBM booster with the given parameter set."""
        train_ds = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        callbacks = [lgb.log_evaluation(period=0)]
        valid_sets = [train_ds]
        valid_names = ["train"]
        if X_val is not None and y_val is not None:
            val_ds = lgb.Dataset(X_val, label=y_val, free_raw_data=False)
            valid_sets.append(val_ds)
            valid_names.append("valid")
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
        return lgb.train(
            params,
            train_ds,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

