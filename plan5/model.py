"""
Plan 5 — Prior-history feature-dominant + simple calibrated model.

Dual columns: Logistic Regression (well-calibrated) → LogLoss columns;
LightGBM optimised for AUC → AUC columns.

Additional enrichment:
    - Per-farmer interaction features (history × current context).
    - Polynomial features for key interactions (degree 2).
    - Standardised inputs for stable optimisation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from plan4.config import AUC_PARAMS, EARLY_STOPPING_ROUNDS, NUM_BOOST_ROUND
from plan5.config import ELASTIC_NET_PARAMS, N_CV_FOLDS
from shared.calibration import Calibrator, CalibrationMethod, get_default_calibration_method
from shared.constants import (
    CV_STRATEGY_DEFAULT,
    CV_TIME_CUTOFF_DEFAULT,
    ID_COL,
    TARGET_COLS,
)
from shared.data_loader import DataLoader
from shared.feature_engineering import FeatureEngineer
from shared.submission import SubmissionGenerator
from shared.validation import CVResult, Validator

logger = logging.getLogger(__name__)


class SimpleModel:
    """Regularised Logistic Regression with rich Prior-derived features.

    The model is deliberately simple to maximise calibration quality.
    All complexity is pushed into the feature engineering stage.

    Parameters
    ----------
    params : dict, optional
        ``LogisticRegression`` keyword arguments.
    n_folds : int
        Cross-validation folds.

    Example
    -------
    >>> model = SimpleModel()
    >>> path = model.run()
    """

    def __init__(
        self,
        params: dict | None = None,
        n_folds: int = N_CV_FOLDS,
        cv_strategy: str = CV_STRATEGY_DEFAULT,
        cv_time_cutoff: str = CV_TIME_CUTOFF_DEFAULT,
    ) -> None:
        self.params = params or ELASTIC_NET_PARAMS.copy()
        self.n_folds = n_folds
        self.cv_strategy = cv_strategy
        self.cv_time_cutoff = cv_time_cutoff
        self.plan_name = "plan5_simple"

    # ── Main pipeline ──────────────────────────────────────────────────

    def run(self, submission_filename: str | None = None) -> Path:
        """Execute the simple-model pipeline."""
        logger.info("=" * 60)
        logger.info("Running %s", self.plan_name)
        logger.info("=" * 60)

        # Load
        loader = DataLoader()
        train_df, test_df, prior_df, sample_sub = loader.load_all()

        # Feature engineering (shared)
        fe = FeatureEngineer(prior_df)
        fe.fit(train_df)
        X_train_base = fe.transform(train_df)
        X_test_base = fe.transform(test_df)

        # Add Plan-5-specific interaction features
        X_train = self._add_interactions(X_train_base)
        X_test = self._add_interactions(X_test_base)

        # Standardise (important for LR convergence and regularisation)
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )

        validator = Validator(n_splits=self.n_folds)
        cv_result = CVResult()
        predictions: dict[str, np.ndarray] = {}
        predictions_auc: dict[str, np.ndarray] = {}

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

            # Model A: Logistic Regression → LogLoss columns
            logger.info("  Training Model A (LogLoss / LR) …")
            oof_preds = np.zeros(len(y))
            oof_mask_ll = np.zeros(len(y), dtype=bool)
            for fold_idx, (tr_idx, va_idx) in enumerate(splits):
                model = LogisticRegression(**self.params)
                model.fit(X_train_scaled.iloc[tr_idx], y.iloc[tr_idx])
                oof_preds[va_idx] = model.predict_proba(
                    X_train_scaled.iloc[va_idx]
                )[:, 1]
                oof_mask_ll[va_idx] = True
                fold_result = validator.evaluate(
                    y.iloc[va_idx].values, oof_preds[va_idx], f"{target}_LL", fold_idx
                )
                cv_result.add(fold_result)
            final_lr = LogisticRegression(**self.params)
            final_lr.fit(X_train_scaled, y)
            coef = pd.Series(
                final_lr.coef_[0], index=X_train_scaled.columns
            ).abs().sort_values(ascending=False)
            logger.info("  Top 10 features:\n%s", coef.head(10).to_string())
            raw_ll = final_lr.predict_proba(X_test_scaled)[:, 1]
            cal_method = get_default_calibration_method()
            if cal_method != CalibrationMethod.NONE:
                cal = Calibrator(cal_method)
                cal.fit(y.values[oof_mask_ll], oof_preds[oof_mask_ll])
                predictions[target] = np.clip(cal.transform(raw_ll), 0.001, 0.999)
            else:
                predictions[target] = np.clip(raw_ll, 0.001, 0.999)

            # Model B: LightGBM AUC → AUC columns
            logger.info("  Training Model B (AUC / LightGBM) …")
            oof_auc = np.zeros(len(y))
            oof_auc_mask = np.zeros(len(y), dtype=bool)
            test_auc_sum = np.zeros(len(X_test_scaled))
            n_folds_eff = len(splits)
            for fold_idx, (tr_idx, va_idx) in enumerate(splits):
                train_ds = lgb.Dataset(
                    X_train_scaled.iloc[tr_idx], label=y.iloc[tr_idx], free_raw_data=False
                )
                val_ds = lgb.Dataset(
                    X_train_scaled.iloc[va_idx], label=y.iloc[va_idx], free_raw_data=False
                )
                booster = lgb.train(
                    AUC_PARAMS,
                    train_ds,
                    num_boost_round=NUM_BOOST_ROUND,
                    valid_sets=[train_ds, val_ds],
                    valid_names=["train", "valid"],
                    callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
                )
                oof_auc[va_idx] = booster.predict(
                    X_train_scaled.iloc[va_idx], num_iteration=booster.best_iteration
                )
                oof_auc_mask[va_idx] = True
                test_auc_sum += (
                    booster.predict(X_test_scaled, num_iteration=booster.best_iteration)
                    / n_folds_eff
                )
            cv_result.add(
                validator.evaluate(
                    y.values[oof_auc_mask], oof_auc[oof_auc_mask], f"{target}_AUC", fold=99
                )
            )
            predictions_auc[target] = np.clip(test_auc_sum, 0.001, 0.999)

        # Enforce hierarchy on both
        for pred_dict in (predictions, predictions_auc):
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

        # Write submission
        filename = submission_filename or f"{self.plan_name}_submission.csv"
        gen = SubmissionGenerator(sample_sub)
        path = gen.generate(
            test_ids=test_df[ID_COL],
            predictions=predictions,
            predictions_auc=predictions_auc,
            filename=filename,
        )

        logger.info("✓ %s complete → %s", self.plan_name, path)
        return path

    # ── Plan-5-specific interaction features ───────────────────────────

    @staticmethod
    def _add_interactions(X: pd.DataFrame) -> pd.DataFrame:
        """Add hand-crafted interaction features.

        These capture non-linear relationships that Logistic Regression
        cannot learn on its own.  Only high-signal interactions are added
        to avoid dimensionality bloat.

        Complexity: O(n) per feature — all vectorised operations.
        """
        X = X.copy()

        # Interaction: has_topic × prior adoption rate
        for suffix in [
            "prior_adopted_within_07_days_rate",
            "prior_adopted_within_90_days_rate",
            "prior_adopted_within_120_days_rate",
        ]:
            if suffix in X.columns:
                X[f"topic_x_{suffix}"] = (
                    X.get("has_topic_trained_on", 0) * X[suffix]
                )

        # Interaction: prior training count × prior adoption rate
        if "prior_training_count" in X.columns:
            for suffix in [
                "prior_adopted_within_07_days_rate",
                "prior_adopted_within_90_days_rate",
                "prior_adopted_within_120_days_rate",
            ]:
                if suffix in X.columns:
                    X[f"count_x_{suffix}"] = X["prior_training_count"] * X[suffix]

        # Interaction: cooperative × group adoption rate
        if "belong_to_cooperative" in X.columns:
            for col in X.columns:
                if "group_" in col and "_rate" in col:
                    X[f"coop_x_{col}"] = X["belong_to_cooperative"] * X[col]

        # Ratio: farmer adoption rate vs regional rate
        for target_suffix in ["07_days", "90_days", "120_days"]:
            farmer_col = f"prior_adopted_within_{target_suffix}_rate"
            county_col = f"county_adopted_within_{target_suffix}_rate"
            if farmer_col in X.columns and county_col in X.columns:
                # Ratio clipped to avoid division issues
                X[f"farmer_vs_county_{target_suffix}"] = X[farmer_col] / (
                    X[county_col] + 1e-6
                )

        return X
