"""
Plan 3 — Stacking Ensemble with diverse base learners.

Architecture:
    Level 0 (base):  LightGBM, XGBoost, CatBoost, Random Forest, Log. Reg.
    Level 1 (meta):  Logistic Regression on out-of-fold base predictions.

The meta-learner receives 5 probability features (one per base model) and
learns an optimal linear combination that is naturally well-calibrated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from plan3.config import (
    CATBOOST_PARAMS,
    EARLY_STOPPING_ROUNDS,
    LGBM_PARAMS,
    LR_PARAMS,
    META_LR_PARAMS,
    N_CV_FOLDS,
    NUM_BOOST_ROUND,
    RF_PARAMS,
    XGB_PARAMS,
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


# ── Base learner protocol ──────────────────────────────────────────────

class BaseLearner(Protocol):
    """Duck-typed interface that every base learner must satisfy."""

    name: str

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
    ) -> None: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


# ── Concrete base learners ─────────────────────────────────────────────

class LGBMLearner:
    """LightGBM base learner wrapper."""

    name: str = "lgbm"

    def __init__(self) -> None:
        self._model: lgb.Booster | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        train_ds = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        callbacks = [lgb.log_evaluation(period=0)]
        valid_sets = [train_ds]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_ds = lgb.Dataset(X_val, label=y_val, free_raw_data=False)
            valid_sets.append(val_ds)
            valid_names.append("valid")
            callbacks.append(lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False))

        self._model = lgb.train(
            LGBM_PARAMS,
            train_ds,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X, num_iteration=self._model.best_iteration)


class XGBLearner:
    """XGBoost base learner wrapper."""

    name: str = "xgb"

    def __init__(self, scale_pos_weight: float = 1.0) -> None:
        self._model: xgb.Booster | None = None
        self._params = XGB_PARAMS.copy()
        self._params["scale_pos_weight"] = scale_pos_weight

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = [(dtrain, "train")]

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "valid"))

        self._model = xgb.train(
            self._params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=evals,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        dmat = xgb.DMatrix(X)
        return self._model.predict(dmat, iteration_range=(0, self._model.best_iteration))


class CatBoostLearner:
    """CatBoost base learner wrapper."""

    name: str = "catboost"

    def __init__(self) -> None:
        self._model: CatBoostClassifier | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        self._model = CatBoostClassifier(**CATBOOST_PARAMS)
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = Pool(X_val, label=y_val)
        self._model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]


class RFLearner:
    """Random Forest base learner wrapper."""

    name: str = "rf"

    def __init__(self) -> None:
        self._model: RandomForestClassifier | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        self._model = RandomForestClassifier(**RF_PARAMS)
        self._model.fit(X_train, y_train)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]


class LRLearner:
    """Logistic Regression base learner wrapper with built-in scaling."""

    name: str = "lr"

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(**LR_PARAMS)),
        ])
        self._pipeline.fit(X_train, y_train)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._pipeline.predict_proba(X)[:, 1]


# ── Stacking ensemble ─────────────────────────────────────────────────

class StackingEnsemble:
    """Two-level stacking ensemble.

    Level-0 base learners produce out-of-fold (OOF) predictions via
    stratified K-fold CV.  Level-1 meta-learner (Logistic Regression)
    is trained on these OOF predictions to learn the optimal blend.

    The meta-learner naturally outputs calibrated probabilities, which
    is ideal given the 75 % Log Loss weight in the competition metric.

    Parameters
    ----------
    n_folds : int
        Number of CV folds for the stacking protocol.

    Example
    -------
    >>> ensemble = StackingEnsemble()
    >>> path = ensemble.run()
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
        self.plan_name = "plan3_stacking"

    def _create_base_learners(self, pos_rate: float) -> list:
        """Instantiate a fresh set of diverse base learners.

        Parameters
        ----------
        pos_rate : float
            Positive class rate for the current target, used to set
            ``scale_pos_weight`` for XGBoost.
        """
        spw = (1.0 - pos_rate) / max(pos_rate, 1e-6)
        return [
            LGBMLearner(),
            XGBLearner(scale_pos_weight=spw),
            CatBoostLearner(),
            RFLearner(),
            LRLearner(),
        ]

    def run(
        self,
        submission_filename: str | None = None,
        write_train_submission: bool = False,
    ) -> "Path":
        """Execute the full stacking pipeline.

        Steps:
            1. Load data and engineer features.
            2. For each target:
                a. Generate OOF predictions from each base learner.
                b. Train meta-learner on OOF matrix.
                c. Generate test predictions from averaged base models.
                d. Apply meta-learner to test meta-features.
            3. Enforce hierarchy and write submission.
        """
        from pathlib import Path

        logger.info("=" * 60)
        logger.info("Running %s", self.plan_name)
        logger.info("=" * 60)

        # 1. Load & features
        loader = DataLoader()
        train_df, test_df, prior_df, sample_sub = loader.load_all()

        fe = FeatureEngineer(prior_df)
        fe.fit(train_df)
        X_train = fe.transform(train_df)
        X_test = fe.transform(test_df)

        validator = Validator(n_splits=self.n_folds)
        cv_result = CVResult()
        predictions: dict[str, np.ndarray] = {}
        oof_predictions: dict[str, np.ndarray] = {}
        oof_mask_per_target: dict[str, np.ndarray] = {}

        for target in TARGET_COLS:
            y = train_df[target]
            pos_rate = y.mean()
            logger.info("── Target: %s (pos_rate=%.4f) ──", target, pos_rate)

            base_learners = self._create_base_learners(pos_rate)
            n_base = len(base_learners)

            # OOF matrix: (n_train, n_base)
            oof_matrix = np.zeros((len(y), n_base))
            oof_mask = np.zeros(len(y), dtype=bool)
            # Test predictions: (n_base, n_folds, n_test) → averaged
            test_preds_per_base = np.zeros((n_base, len(X_test)))

            # 2a. Generate OOF predictions
            folds = list(
                validator.cv_splits(
                    train_df,
                    y,
                    strategy=self.cv_strategy,
                    cutoff=self.cv_time_cutoff,
                )
            )
            n_folds_eff = len(folds)
            for fold_idx, (tr_idx, va_idx) in enumerate(folds):
                X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
                y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
                oof_mask[va_idx] = True

                fold_learners = self._create_base_learners(pos_rate)
                for i, learner in enumerate(fold_learners):
                    learner.fit(X_tr, y_tr, X_va, y_va)
                    oof_matrix[va_idx, i] = learner.predict_proba(X_va)
                    test_preds_per_base[i] += (
                        learner.predict_proba(X_test) / n_folds_eff
                    )

            # Evaluate OOF ensemble (simple average first)
            oof_avg = oof_matrix.mean(axis=1)
            fold_result = validator.evaluate(
                y.values[oof_mask], oof_avg[oof_mask], target, fold=99
            )
            cv_result.add(fold_result)

            # 2b. Train meta-learner on OOF matrix
            meta = LogisticRegression(**META_LR_PARAMS)
            meta.fit(oof_matrix, y)
            logger.info(
                "  Meta-learner coefficients: %s",
                dict(zip(
                    [l.name for l in base_learners],
                    np.round(meta.coef_[0], 4),
                )),
            )

            # 2c. Build test meta-features & predict
            test_meta = test_preds_per_base.T  # (n_test, n_base)
            predictions[target] = meta.predict_proba(test_meta)[:, 1]
            if write_train_submission:
                oof_predictions[target] = meta.predict_proba(oof_matrix)[:, 1]
                oof_mask_per_target[target] = oof_mask.copy()

        # 3. Enforce hierarchy
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

        if write_train_submission and oof_predictions:
            mask_final = (
                oof_mask_per_target[TARGET_COLS[0]]
                & oof_mask_per_target[TARGET_COLS[1]]
                & oof_mask_per_target[TARGET_COLS[2]]
            )
            oof_p07, oof_p90, oof_p120 = Calibrator.enforce_hierarchy(
                oof_predictions[TARGET_COLS[0]],
                oof_predictions[TARGET_COLS[1]],
                oof_predictions[TARGET_COLS[2]],
            )
            train_sub_dir = Path(SUBMISSION_DIR) / "train"
            train_sub_dir.mkdir(parents=True, exist_ok=True)
            train_ids = train_df.loc[mask_final, ID_COL]
            sub = pd.DataFrame({ID_COL: train_ids.values})
            for i, tc in enumerate(TARGET_COLS):
                auc_col, ll_col = SUBMISSION_MAP[tc]
                p = (oof_p07, oof_p90, oof_p120)[i]
                sub[ll_col] = p[mask_final]
                sub[auc_col] = p[mask_final]
            out = train_sub_dir / f"{self.plan_name}_submission.csv"
            sub.to_csv(out, index=False)
            logger.info("Train (OOF) submission written to %s  (%d rows)", out, len(sub))

        # Write submission
        filename = submission_filename or f"{self.plan_name}_submission.csv"
        gen = SubmissionGenerator(sample_sub)
        path = gen.generate(test_ids=test_df[ID_COL], predictions=predictions, filename=filename)

        logger.info("✓ %s complete → %s", self.plan_name, path)
        return path
