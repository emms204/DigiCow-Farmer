"""
Plan 4 — Dual-optimisation strategy.

Exploits the fact that the submission file has *separate* columns for
AUC evaluation and LogLoss evaluation.  Two differently-tuned LightGBM
models are trained per target:

    Model A (LogLoss):  conservative, heavily calibrated → LogLoss columns.
    Model B (AUC):      aggressive, ranking-focused     → AUC columns.

This allows each metric to receive probabilities optimised specifically
for it, rather than a compromise that's sub-optimal for both.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from plan4.config import (
    AUC_PARAMS,
    EARLY_STOPPING_ROUNDS,
    LOGLOSS_PARAMS,
    N_CV_FOLDS,
    NUM_BOOST_ROUND,
)
from shared.calibration import Calibrator, CalibrationMethod
from shared.constants import (
    CV_STRATEGY_DEFAULT,
    CV_TIME_CUTOFF_DEFAULT,
    ID_COL,
    SUBMISSION_DIR,
    SUBMISSION_MAP,
    TARGET_COLS,
)
from shared.data_loader import DataLoader
from shared.feature_engineering import FeatureEngineer
from shared.submission import SubmissionGenerator
from shared.validation import CVResult, Validator

logger = logging.getLogger(__name__)


class DualOptimiser:
    """Train two LightGBM models per target — one for each metric.

    Parameters
    ----------
    n_folds : int
        Number of folds for OOF calibration evaluation.

    Example
    -------
    >>> opt = DualOptimiser()
    >>> path = opt.run()
    """

    def __init__(
        self,
        n_folds: int = N_CV_FOLDS,
        cv_strategy: str = CV_STRATEGY_DEFAULT,
        cv_time_cutoff: str = CV_TIME_CUTOFF_DEFAULT,
    ) -> None:
        self.n_folds = n_folds
        self.cv_strategy = cv_strategy
        self.cv_time_cutoff = cv_time_cutoff
        self.plan_name = "plan4_dual"

    # ── Main pipeline ──────────────────────────────────────────────────

    def run(
        self,
        submission_filename: str | None = None,
        write_train_submission: bool = False,
    ) -> Path:
        """Execute the dual-optimisation pipeline.

        Returns the path to the generated submission file.
        """
        logger.info("=" * 60)
        logger.info("Running %s", self.plan_name)
        logger.info("=" * 60)

        # Load & features
        loader = DataLoader()
        train_df, test_df, prior_df, sample_sub = loader.load_all()

        fe = FeatureEngineer(prior_df)
        fe.fit(train_df)
        X_train = fe.transform(train_df)
        X_test = fe.transform(test_df)

        validator = Validator(n_splits=self.n_folds)
        cv_result = CVResult()

        preds_logloss: dict[str, np.ndarray] = {}
        preds_auc: dict[str, np.ndarray] = {}
        oof_logloss: dict[str, np.ndarray] = {}
        oof_auc: dict[str, np.ndarray] = {}
        oof_mask_per_target: dict[str, np.ndarray] = {}

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

            # ── Model A: LogLoss-optimised ─────────────────────────────
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
                )
                oof_ll[va_idx] = booster.predict(
                    X_train.iloc[va_idx], num_iteration=booster.best_iteration
                )
                oof_ll_mask[va_idx] = True
                test_ll += (
                    booster.predict(X_test, num_iteration=booster.best_iteration)
                    / n_folds_eff
                )

            # Evaluate LogLoss model
            ll_result = validator.evaluate(
                y.values[oof_ll_mask], oof_ll[oof_ll_mask], f"{target}_LL", fold=0
            )
            cv_result.add(ll_result)

            # Calibrate LogLoss predictions (critical for 75 % weight)
            cal_ll = Calibrator(CalibrationMethod.ISOTONIC)
            cal_ll.fit(y.values[oof_ll_mask], oof_ll[oof_ll_mask])
            preds_logloss[target] = cal_ll.transform(test_ll)
            if write_train_submission:
                oof_logloss[target] = cal_ll.transform(oof_ll)
                oof_mask_per_target[target] = oof_ll_mask.copy()

            # ── Model B: AUC-optimised ─────────────────────────────────
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
                )
                oof_auc_arr[va_idx] = booster.predict(
                    X_train.iloc[va_idx], num_iteration=booster.best_iteration
                )
                oof_auc_mask[va_idx] = True
                test_auc += (
                    booster.predict(X_test, num_iteration=booster.best_iteration)
                    / n_folds_eff
                )

            # Evaluate AUC model
            auc_result = validator.evaluate(
                y.values[oof_auc_mask], oof_auc_arr[oof_auc_mask], f"{target}_AUC", fold=0
            )
            cv_result.add(auc_result)

            # No heavy calibration for AUC (ranking is invariant to monotone transforms)
            preds_auc[target] = np.clip(test_auc, 0.001, 0.999)
            if write_train_submission:
                oof_auc[target] = np.clip(oof_auc_arr, 0.001, 0.999)

        # Enforce hierarchy on both prediction sets independently
        for pred_dict in (preds_logloss, preds_auc):
            p07, p90, p120 = Calibrator.enforce_hierarchy(
                pred_dict[TARGET_COLS[0]],
                pred_dict[TARGET_COLS[1]],
                pred_dict[TARGET_COLS[2]],
            )
            pred_dict[TARGET_COLS[0]] = p07
            pred_dict[TARGET_COLS[1]] = p90
            pred_dict[TARGET_COLS[2]] = p120

        # Log summary
        summary = cv_result.summary()
        logger.info("\n%s CV Summary:\n%s", self.plan_name, summary.to_string())

        if write_train_submission and oof_logloss and oof_auc:
            mask_final = (
                oof_mask_per_target[TARGET_COLS[0]]
                & oof_mask_per_target[TARGET_COLS[1]]
                & oof_mask_per_target[TARGET_COLS[2]]
            )
            oof_ll_07, oof_ll_90, oof_ll_120 = Calibrator.enforce_hierarchy(
                oof_logloss[TARGET_COLS[0]],
                oof_logloss[TARGET_COLS[1]],
                oof_logloss[TARGET_COLS[2]],
            )
            oof_auc_07, oof_auc_90, oof_auc_120 = Calibrator.enforce_hierarchy(
                oof_auc[TARGET_COLS[0]],
                oof_auc[TARGET_COLS[1]],
                oof_auc[TARGET_COLS[2]],
            )
            train_sub_dir = SUBMISSION_DIR / "train"
            train_sub_dir.mkdir(parents=True, exist_ok=True)
            train_ids = train_df.loc[mask_final, ID_COL]
            sub = pd.DataFrame({ID_COL: train_ids.values})
            for i, tc in enumerate(TARGET_COLS):
                auc_col, ll_col = SUBMISSION_MAP[tc]
                oof_ll = (oof_ll_07, oof_ll_90, oof_ll_120)[i]
                oof_a = (oof_auc_07, oof_auc_90, oof_auc_120)[i]
                sub[ll_col] = oof_ll[mask_final]
                sub[auc_col] = oof_a[mask_final]
            out = train_sub_dir / f"{self.plan_name}_submission.csv"
            sub.to_csv(out, index=False)
            logger.info("Train (OOF) submission written to %s  (%d rows)", out, len(sub))

        # Write submission with SEPARATE predictions per column type
        filename = submission_filename or f"{self.plan_name}_submission.csv"
        gen = SubmissionGenerator(sample_sub)
        path = gen.generate(
            test_ids=test_df[ID_COL],
            predictions=preds_logloss,       # → LogLoss columns
            predictions_auc=preds_auc,       # → AUC columns
            filename=filename,
        )

        logger.info("✓ %s complete → %s", self.plan_name, path)
        return path

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _train_lgbm(
        params: dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> lgb.Booster:
        """Train a LightGBM booster with the given parameter set.

        Complexity: O(n * d * T).
        """
        train_ds = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        callbacks = [lgb.log_evaluation(period=0)]
        valid_sets = [train_ds]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_ds = lgb.Dataset(X_val, label=y_val, free_raw_data=False)
            valid_sets.append(val_ds)
            valid_names.append("valid")
            callbacks.append(
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)
            )

        return lgb.train(
            params,
            train_ds,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
