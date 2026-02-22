"""
Plan 2 — CatBoost with combined Prior + Train data.

Key differences from Plan 1:
    - Combines Prior (44 K rows) with Train (13.5 K rows) for more data.
    - Passes categorical columns natively to CatBoost (no manual encoding).
    - CatBoost's ordered boosting provides built-in overfitting protection.
    - Prior samples are down-weighted to account for distribution differences.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from plan2.config import CATBOOST_PARAMS, EARLY_STOPPING_ROUNDS, PRIOR_SAMPLE_WEIGHT
from shared.calibration import Calibrator, CalibrationMethod
from shared.constants import (
    CATEGORICAL_FEATURES,
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


class CatBoostModel:
    """CatBoost classifier trained on combined Prior + Train data.

    Instead of inheriting from ``BaseModel`` (which assumes a single
    training DataFrame), this class manages its own pipeline because it
    merges two data sources with different weights.

    Parameters
    ----------
    params : dict, optional
        CatBoost hyperparameters.
    prior_weight : float
        Sample weight for rows originating from Prior.csv.

    Example
    -------
    >>> model = CatBoostModel()
    >>> submission_path = model.run()
    """

    def __init__(
        self,
        params: dict | None = None,
        prior_weight: float = PRIOR_SAMPLE_WEIGHT,
        cv_strategy: str = CV_STRATEGY_DEFAULT,
        cv_time_cutoff: str = CV_TIME_CUTOFF_DEFAULT,
    ) -> None:
        self.params = params or CATBOOST_PARAMS.copy()
        self.prior_weight = prior_weight
        self.cv_strategy = cv_strategy
        self.cv_time_cutoff = cv_time_cutoff
        self.plan_name = "plan2_catboost"

    # ── Main pipeline ──────────────────────────────────────────────────

    def run(self, submission_filename: str | None = None) -> "Path":
        """Execute the full Plan 2 pipeline.

        Steps:
            1. Load and normalise data.
            2. Build features (shared feature engineer).
            3. Combine Train + Prior with source weights.
            4. Cross-validate on Train portion only (no data leakage).
            5. Train final model on combined data.
            6. Calibrate, enforce hierarchy, and write submission.
        """
        from pathlib import Path

        logger.info("=" * 60)
        logger.info("Running %s", self.plan_name)
        logger.info("=" * 60)

        # 1. Load
        loader = DataLoader()
        train_df, test_df, prior_df, sample_sub = loader.load_all()

        # 2. Feature engineering
        fe = FeatureEngineer(prior_df)
        fe.fit(train_df)

        X_train = fe.transform(train_df)
        X_test = fe.transform(test_df)
        X_prior = fe.transform(prior_df)

        # 3. Combine Train + Prior
        X_combined = pd.concat([X_train, X_prior], ignore_index=True)

        # Sample weights: Train = 1.0, Prior = prior_weight
        weights = np.concatenate([
            np.ones(len(X_train)),
            np.full(len(X_prior), self.prior_weight),
        ])

        # Add source indicator feature
        X_combined["source_is_prior"] = np.concatenate([
            np.zeros(len(X_train)),
            np.ones(len(X_prior)),
        ]).astype(int)
        X_test["source_is_prior"] = 0

        # Identify categorical feature indices for CatBoost
        cat_feature_names = [
            col for col in X_combined.columns
            if X_combined[col].dtype in ("int8", "int16", "int32", "int64", "category")
            and col in CATEGORICAL_FEATURES
        ]
        # Use column index positions for CatBoost
        cat_indices = [
            X_combined.columns.get_loc(c)
            for c in cat_feature_names
            if c in X_combined.columns
        ]

        # 4. CV on Train portion only + 5. Train final + 6. Predict
        validator = Validator(n_splits=5)
        cv_result = CVResult()
        predictions: dict[str, np.ndarray] = {}

        for target in TARGET_COLS:
            y_train_full = train_df[target]
            y_prior = prior_df[target]
            y_combined = pd.concat([y_train_full, y_prior], ignore_index=True)

            logger.info(
                "── Target: %s (train_pos=%.4f, prior_pos=%.4f) ──",
                target,
                y_train_full.mean(),
                y_prior.mean(),
            )

            # CV on Train subset only
            oof_preds = np.zeros(len(y_train_full))
            oof_mask = np.zeros(len(y_train_full), dtype=bool)
            splits = list(
                validator.cv_splits(
                    train_df,
                    y_train_full,
                    strategy=self.cv_strategy,
                    cutoff=self.cv_time_cutoff,
                )
            )
            for fold_idx, (tr_idx, va_idx) in enumerate(splits):
                model = self._build_catboost()
                train_pool = Pool(
                    X_train.iloc[tr_idx],
                    label=y_train_full.iloc[tr_idx],
                    cat_features=cat_indices[:len([c for c in cat_feature_names if c in X_train.columns])],
                )
                val_pool = Pool(
                    X_train.iloc[va_idx],
                    label=y_train_full.iloc[va_idx],
                    cat_features=cat_indices[:len([c for c in cat_feature_names if c in X_train.columns])],
                )
                model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
                oof_preds[va_idx] = model.predict_proba(X_train.iloc[va_idx])[:, 1]
                oof_mask[va_idx] = True

                fold_result = validator.evaluate(
                    y_train_full.iloc[va_idx].values,
                    oof_preds[va_idx],
                    target,
                    fold_idx,
                )
                cv_result.add(fold_result)

            # Calibrate on OOF
            calibrator = Calibrator(CalibrationMethod.ISOTONIC)
            calibrator.fit(y_train_full.values[oof_mask], oof_preds[oof_mask])

            # Train final model on combined data
            # Disable use_best_model for final fit (no eval set)
            final_model = self._build_catboost()
            final_model.set_params(use_best_model=False)
            combined_pool = Pool(
                X_combined,
                label=y_combined,
                weight=weights,
                cat_features=cat_indices,
            )
            final_model.fit(combined_pool)

            # Predict & calibrate
            raw_preds = final_model.predict_proba(X_test)[:, 1]
            predictions[target] = calibrator.transform(raw_preds)

        # Enforce hierarchy
        p07, p90, p120 = Calibrator.enforce_hierarchy(
            predictions[TARGET_COLS[0]],
            predictions[TARGET_COLS[1]],
            predictions[TARGET_COLS[2]],
        )
        predictions[TARGET_COLS[0]] = p07
        predictions[TARGET_COLS[1]] = p90
        predictions[TARGET_COLS[2]] = p120

        # Log CV summary
        summary = cv_result.summary()
        logger.info("\n%s CV Summary:\n%s", self.plan_name, summary.to_string())

        # Write submission
        filename = submission_filename or f"{self.plan_name}_submission.csv"
        gen = SubmissionGenerator(sample_sub)
        path = gen.generate(test_ids=test_df[ID_COL], predictions=predictions, filename=filename)

        logger.info("✓ %s complete → %s", self.plan_name, path)
        return path

    # ── Helpers ────────────────────────────────────────────────────────

    def _build_catboost(self) -> CatBoostClassifier:
        """Instantiate a fresh CatBoostClassifier with the configured params."""
        return CatBoostClassifier(**self.params)
