"""
Plan 5 — Prior-history feature-dominant + simple calibrated model.

Philosophy: since 62.7 % of test farmers have history in Prior, and
``has_topic_trained_on == 0`` means zero adoption, the signal is
largely in feature engineering rather than model complexity.

A regularised Logistic Regression is inherently well-calibrated (no
post-hoc calibration needed) and immune to overfitting on small data,
making it ideal for the 75 % Log Loss evaluation weight.

Additional enrichment:
    - Per-farmer interaction features (history × current context).
    - Polynomial features for key interactions (degree 2).
    - Standardised inputs for stable optimisation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from plan5.config import ELASTIC_NET_PARAMS, N_CV_FOLDS
from shared.calibration import Calibrator
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

        for target in TARGET_COLS:
            y = train_df[target]
            logger.info("── Target: %s (pos_rate=%.4f) ──", target, y.mean())

            # CV evaluation (no calibration needed — LR is already calibrated)
            oof_preds = np.zeros(len(y))
            splits = list(
                validator.cv_splits(
                    train_df,
                    y,
                    strategy=self.cv_strategy,
                    cutoff=self.cv_time_cutoff,
                )
            )
            for fold_idx, (tr_idx, va_idx) in enumerate(splits):
                model = LogisticRegression(**self.params)
                model.fit(X_train_scaled.iloc[tr_idx], y.iloc[tr_idx])
                oof_preds[va_idx] = model.predict_proba(
                    X_train_scaled.iloc[va_idx]
                )[:, 1]

                fold_result = validator.evaluate(
                    y.iloc[va_idx].values, oof_preds[va_idx], target, fold_idx
                )
                cv_result.add(fold_result)

            # Train final model on all data
            final_model = LogisticRegression(**self.params)
            final_model.fit(X_train_scaled, y)

            # Log top features by coefficient magnitude
            coef = pd.Series(
                final_model.coef_[0], index=X_train_scaled.columns
            ).abs().sort_values(ascending=False)
            logger.info(
                "  Top 10 features:\n%s",
                coef.head(10).to_string(),
            )

            # Predict on test (no additional calibration)
            predictions[target] = np.clip(
                final_model.predict_proba(X_test_scaled)[:, 1], 0.001, 0.999
            )

        # Enforce hierarchy
        p07, p90, p120 = Calibrator.enforce_hierarchy(
            predictions[TARGET_COLS[0]],
            predictions[TARGET_COLS[1]],
            predictions[TARGET_COLS[2]],
        )
        predictions[TARGET_COLS[0]] = p07
        predictions[TARGET_COLS[1]] = p90
        predictions[TARGET_COLS[2]] = p120

        # Log summary
        summary = cv_result.summary()
        logger.info("\n%s CV Summary:\n%s", self.plan_name, summary.to_string())

        # Write submission
        filename = submission_filename or f"{self.plan_name}_submission.csv"
        gen = SubmissionGenerator(sample_sub)
        path = gen.generate(
            test_ids=test_df[ID_COL], predictions=predictions, filename=filename
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
