#!/usr/bin/env python3
"""
E10 — Dual-head HPO harness (E10.0 lock benchmark + E10.1 LGBM dual HPO).

E10.0: Lock benchmark
  - E0 harness: forward non-overlap folds, as-of prior, fixed seed.
  - Plan4 architecture: dual columns (LL head + AUC head per target).
  - Minimal features, dual-column OOF eval.
  - Pass: reproducibility delta < 0.002 over 3 runs; baseline frozen to e10_baseline.json.

E10.1: LGBM dual HPO (minimal features)
  - 2 Optuna studies per target (LL head + AUC head).
  - Optimize: OOF weighted score from 6 columns.
  - Guardrails: 7d LL not worse, worst-fold not worse, fold std controlled.
  - Pass rule: Δweighted >= +0.004 vs E10.0.

Usage:
  python e10_dual_hpo.py baseline [--runs 3] [--dry-run]   # E10.0
  python e10_dual_hpo.py hpo [--n-trials 50] [--study 7d]  # E10.1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.calibration import Calibrator, CalibrationMethod
from shared.constants import (
    DATE_COL,
    PROB_CLIP_MAX,
    PROB_CLIP_MIN,
    RANDOM_SEED,
    TARGET_COLS,
)
from shared.data_loader import DataLoader
from shared.evaluation import calculate_weighted_score
from shared.feature_engineering import (
    MINIMAL_FEATURE_GROUPS,
    FeatureEngineer,
    VETTED_FEATURE_GROUPS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Locked E10 config (aligned with E0) ───────────────────────────────────
E10_N_SPLITS = 5
E10_MIN_TRAIN_SIZE = 0.3
E10_CLIP = (PROB_CLIP_MIN, PROB_CLIP_MAX)
E10_SEED = RANDOM_SEED
E10_BASELINE_PATH = Path(__file__).resolve().parent / "e10_baseline.json"
E10_1_BEST_PATH = Path(__file__).resolve().parent / "e10_1_best.json"
E10_PASS_DELTA_REPRO = 0.002
E10_PASS_DELTA_HPO = 0.004

# Optuna (why same score repeatedly / slow increase):
# - Default sampler is TPESampler (Tree-structured Parzen Estimator). It fits
#   distributions to "good" and "bad" trials and samples next params from
#   regions where good trials are more likely. So it exploits more than
#   explores: many trials are near previous good ones → similar scores;
#   occasional jumps when a better region is found. That's by design, not a bug.
# - Pruning (cutting off bad trials early) is opt-in (MedianPruner etc.); we
#   don't use it here, so every trial runs to completion and is fully evaluated.
# - For more random exploration use RandomSampler() in create_study(sampler=...).


def rolling_forward_splits(
    df: pd.DataFrame,
    n_splits: int = E10_N_SPLITS,
    min_train_size: float = E10_MIN_TRAIN_SIZE,
) -> list[dict[str, Any]]:
    """Non-overlapping forward-time splits. Train = past, val = next window."""
    if DATE_COL not in df.columns:
        raise ValueError(f"{DATE_COL} missing")

    ordered = df[[DATE_COL]].copy()
    ordered["_pos"] = np.arange(len(df), dtype=int)
    ordered = ordered.sort_values(DATE_COL)

    min_date = ordered[DATE_COL].min()
    max_date = ordered[DATE_COL].max()

    cutoff_fracs = [
        min_train_size + (i + 1) * (1 - min_train_size) / (n_splits + 1)
        for i in range(n_splits + 1)
    ]
    cutoffs = [min_date + (max_date - min_date) * frac for frac in cutoff_fracs]

    splits = []
    for i in range(n_splits):
        start, end = cutoffs[i], cutoffs[i + 1]
        train_mask = ordered[DATE_COL] < start
        val_mask = (ordered[DATE_COL] >= start) & (ordered[DATE_COL] < end)
        train_pos = ordered.loc[train_mask, "_pos"].to_numpy(dtype=int)
        val_pos = ordered.loc[val_mask, "_pos"].to_numpy(dtype=int)
        if len(train_pos) == 0 or len(val_pos) == 0:
            continue
        splits.append({
            "fold": i + 1,
            "train_pos": train_pos,
            "val_pos": val_pos,
            "train_cutoff": start,
            "val_end": end,
        })
    if not splits:
        raise ValueError("No valid forward splits")
    return splits


# Default LGBM params (Plan4-style: LL conservative, AUC aggressive)
def _default_ll_params(seed: int = E10_SEED) -> dict:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 20,
        "max_depth": 5,
        "min_child_samples": 50,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "lambda_l1": 0.5,
        "lambda_l2": 2.0,
        "is_unbalance": True,
        "verbosity": -1,
        "random_state": seed,
        "n_jobs": -1,
    }


def _default_auc_params(seed: int = E10_SEED) -> dict:
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 7,
        "min_child_samples": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "lambda_l1": 0.05,
        "lambda_l2": 0.5,
        "is_unbalance": True,
        "verbosity": -1,
        "random_state": seed,
        "n_jobs": -1,
    }


EARLY_STOPPING_ROUNDS = 50
NUM_BOOST_ROUND = 1000


def run_single_e10_harness(
    train_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    n_splits: int = E10_N_SPLITS,
    seed: int = E10_SEED,
    *,
    ll_params_per_target: dict[str, dict] | None = None,
    auc_params_per_target: dict[str, dict] | None = None,
    num_boost_round: int | None = None,
    feature_groups: dict[str, bool] | None = None,
    ll_calibration_per_target: dict[str, CalibrationMethod] | None = None,
) -> dict[str, Any]:
    """
    One run of E10 harness: single model per target (one column per target),
    E0-style forward folds and as-of features. Same prediction used for both
    LL and AUC in the weighted score (after calibration).
    feature_groups: if None, uses MINIMAL_FEATURE_GROUPS (caller should set
      DIGICOW_MINIMAL_FEATURES=1 for minimal). Pass VETTED_FEATURE_GROUPS for E10.2.
    ll_calibration_per_target: if provided, use this CalibrationMethod per target
      (keys = TARGET_COLS). If None, use ISOTONIC for all.
    auc_params_per_target: ignored (kept for API compatibility).
    Returns weighted_score, per_fold_weighted_scores, fold_std, last_fold_score,
    oof_07_ll, oof_07_auc, ... and baseline_guardrail metrics.
    """
    import lightgbm as lgb
    from sklearn.metrics import log_loss, roc_auc_score

    np.random.seed(seed)
    train_df = train_df.copy()
    prior_df = prior_df.copy()
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])

    if ll_params_per_target is None:
        ll_params_per_target = {t: _default_ll_params(seed) for t in TARGET_COLS}

    n_rounds = num_boost_round if num_boost_round is not None else NUM_BOOST_ROUND

    splits = rolling_forward_splits(
        train_df, n_splits=n_splits, min_train_size=E10_MIN_TRAIN_SIZE
    )
    n_rows = len(train_df)

    oof_ll: dict[str, np.ndarray] = {t: np.zeros(n_rows, dtype=float) for t in TARGET_COLS}
    oof_mask = np.zeros(n_rows, dtype=bool)
    per_fold_weighted_scores: list[float] = []

    for split in splits:
        fold = split["fold"]
        train_pos = split["train_pos"]
        val_pos = split["val_pos"]
        train_cutoff = split["train_cutoff"]

        prior_asof = prior_df[prior_df[DATE_COL] < train_cutoff].copy()
        fe = FeatureEngineer(prior_asof)
        fe.set_feature_groups(feature_groups if feature_groups is not None else MINIMAL_FEATURE_GROUPS)
        fe.fit(train_df.iloc[train_pos])

        X_tr = fe.transform(train_df.iloc[train_pos])
        X_va = fe.transform(train_df.iloc[val_pos])
        fold_metrics: dict[str, float] = {}

        for target in TARGET_COLS:
            y = train_df[target]
            y_tr = y.iloc[train_pos].values
            y_va = y.iloc[val_pos].values

            train_set = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
            val_set = lgb.Dataset(X_va, label=y_va, free_raw_data=False)
            booster = lgb.train(
                params=ll_params_per_target[target],
                train_set=train_set,
                num_boost_round=n_rounds,
                valid_sets=[train_set, val_set],
                valid_names=["train", "valid"],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            pred_va = booster.predict(X_va, num_iteration=booster.best_iteration)
            pred_va = np.clip(pred_va, *E10_CLIP)
            oof_ll[target][val_pos] = pred_va
            oof_mask[val_pos] = True

            fold_metrics[f"{target}_ll"] = log_loss(y_va, pred_va)
            fold_metrics[f"{target}_auc"] = (
                roc_auc_score(y_va, pred_va) if len(np.unique(y_va)) > 1 else 0.5
            )

        fold_weighted = calculate_weighted_score(
            fold_metrics["adopted_within_07_days_auc"],
            fold_metrics["adopted_within_07_days_ll"],
            fold_metrics["adopted_within_90_days_auc"],
            fold_metrics["adopted_within_90_days_ll"],
            fold_metrics["adopted_within_120_days_auc"],
            fold_metrics["adopted_within_120_days_ll"],
        )
        per_fold_weighted_scores.append(fold_weighted)
        logger.info(
            "  Fold %d: train=%d val=%d weighted=%.6f",
            fold, len(train_pos), len(val_pos), fold_weighted,
        )

    _cal_method = ll_calibration_per_target or {}
    def _method(t: str) -> CalibrationMethod:
        return _cal_method.get(t, CalibrationMethod.ISOTONIC)

    oof_07_ll_raw = oof_ll[TARGET_COLS[0]][oof_mask]
    oof_90_ll_raw = oof_ll[TARGET_COLS[1]][oof_mask]
    oof_120_ll_raw = oof_ll[TARGET_COLS[2]][oof_mask]
    y_07 = train_df[TARGET_COLS[0]].values[oof_mask]
    y_90 = train_df[TARGET_COLS[1]].values[oof_mask]
    y_120 = train_df[TARGET_COLS[2]].values[oof_mask]

    cal_07 = Calibrator(_method(TARGET_COLS[0]))
    cal_07.fit(y_07, oof_07_ll_raw)
    oof_07_ll_cal = cal_07.transform(oof_07_ll_raw)
    cal_90 = Calibrator(_method(TARGET_COLS[1]))
    cal_90.fit(y_90, oof_90_ll_raw)
    oof_90_ll_cal = cal_90.transform(oof_90_ll_raw)
    cal_120 = Calibrator(_method(TARGET_COLS[2]))
    cal_120.fit(y_120, oof_120_ll_raw)
    oof_120_ll_cal = cal_120.transform(oof_120_ll_raw)

    oof_07_ll_cal, oof_90_ll_cal, oof_120_ll_cal = Calibrator.enforce_hierarchy(
        oof_07_ll_cal, oof_90_ll_cal, oof_120_ll_cal
    )
    oof_07_auc = oof_07_ll_cal
    oof_90_auc = oof_90_ll_cal
    oof_120_auc = oof_120_ll_cal

    oof_07_ll = log_loss(y_07, oof_07_ll_cal)
    oof_07_auc_score = (
        roc_auc_score(y_07, oof_07_auc) if len(np.unique(y_07)) > 1 else 0.5
    )
    oof_90_ll = log_loss(y_90, oof_90_ll_cal)
    oof_90_auc_score = (
        roc_auc_score(y_90, oof_90_auc) if len(np.unique(y_90)) > 1 else 0.5
    )
    oof_120_ll = log_loss(y_120, oof_120_ll_cal)
    oof_120_auc_score = (
        roc_auc_score(y_120, oof_120_auc) if len(np.unique(y_120)) > 1 else 0.5
    )

    weighted_score = calculate_weighted_score(
        oof_07_auc_score,
        oof_07_ll,
        oof_90_auc_score,
        oof_90_ll,
        oof_120_auc_score,
        oof_120_ll,
    )
    fold_std = (
        float(np.std(per_fold_weighted_scores))
        if len(per_fold_weighted_scores) > 1
        else 0.0
    )
    worst_fold_score = (
        min(per_fold_weighted_scores) if per_fold_weighted_scores else 0.0
    )
    last_fold_score = (
        per_fold_weighted_scores[-1] if per_fold_weighted_scores else 0.0
    )

    return {
        "weighted_score": weighted_score,
        "per_fold_weighted_scores": per_fold_weighted_scores,
        "fold_std": fold_std,
        "worst_fold_score": worst_fold_score,
        "last_fold_score": last_fold_score,
        "oof_07_ll": oof_07_ll,
        "oof_07_auc": oof_07_auc_score,
        "oof_90_ll": oof_90_ll,
        "oof_90_auc": oof_90_auc_score,
        "oof_120_ll": oof_120_ll,
        "oof_120_auc": oof_120_auc_score,
    }


# ── XGBoost / CatBoost dual harness (E10.4 / E10.5, vetted features) ──

def _default_xgb_ll_params(seed: int = E10_SEED) -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.03,
        "max_depth": 5,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "gamma": 0.0,
        "scale_pos_weight": 1,
        "verbosity": 0,
        "random_state": seed,
        "n_jobs": -1,
    }


def _default_xgb_auc_params(seed: int = E10_SEED) -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": 0.04,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 0.5,
        "gamma": 0.0,
        "scale_pos_weight": 1,
        "verbosity": 0,
        "random_state": seed,
        "n_jobs": -1,
    }


def _default_catboost_ll_params(seed: int = E10_SEED) -> dict:
    return {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "bagging_temperature": 0.2,
        "random_strength": 0.1,
        "border_count": 128,
        "iterations": NUM_BOOST_ROUND,
        "random_seed": seed,
        "verbose": 0,
        "allow_writing_files": False,
        "use_best_model": True,
    }


def _default_catboost_auc_params(seed: int = E10_SEED) -> dict:
    return {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "learning_rate": 0.06,
        "depth": 7,
        "l2_leaf_reg": 1.0,
        "bagging_temperature": 0.2,
        "random_strength": 0.1,
        "border_count": 128,
        "iterations": NUM_BOOST_ROUND,
        "random_seed": seed,
        "verbose": 0,
        "allow_writing_files": False,
        "use_best_model": True,
    }


def run_single_e10_harness_xgb(
    train_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    n_splits: int = E10_N_SPLITS,
    seed: int = E10_SEED,
    *,
    ll_params_per_target: dict[str, dict] | None = None,
    auc_params_per_target: dict[str, dict] | None = None,
    num_boost_round: int | None = None,
    feature_groups: dict[str, bool] | None = None,
    ll_calibration_per_target: dict[str, CalibrationMethod] | None = None,
) -> dict[str, Any]:
    """Single model per target (XGBoost); vetted features (E10.4). Same prediction for LL and AUC."""
    import xgboost as xgb
    from sklearn.metrics import log_loss, roc_auc_score

    np.random.seed(seed)
    train_df = train_df.copy()
    prior_df = prior_df.copy()
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    if ll_params_per_target is None:
        ll_params_per_target = {t: _default_xgb_ll_params(seed) for t in TARGET_COLS}
    n_rounds = num_boost_round if num_boost_round is not None else NUM_BOOST_ROUND
    feature_groups = feature_groups or VETTED_FEATURE_GROUPS

    splits = rolling_forward_splits(
        train_df, n_splits=n_splits, min_train_size=E10_MIN_TRAIN_SIZE
    )
    n_rows = len(train_df)
    oof_ll = {t: np.zeros(n_rows, dtype=float) for t in TARGET_COLS}
    oof_mask = np.zeros(n_rows, dtype=bool)
    per_fold_weighted_scores: list[float] = []

    for split in splits:
        fold, train_pos, val_pos, train_cutoff = (
            split["fold"], split["train_pos"], split["val_pos"], split["train_cutoff"],
        )
        prior_asof = prior_df[prior_df[DATE_COL] < train_cutoff].copy()
        fe = FeatureEngineer(prior_asof)
        fe.set_feature_groups(feature_groups)
        fe.fit(train_df.iloc[train_pos])
        X_tr = fe.transform(train_df.iloc[train_pos])
        X_va = fe.transform(train_df.iloc[val_pos])
        fold_metrics: dict[str, float] = {}
        for target in TARGET_COLS:
            y_tr = train_df[target].iloc[train_pos].values
            y_va = train_df[target].iloc[val_pos].values
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_va, label=y_va)
            p = {k: v for k, v in ll_params_per_target[target].items() if k != "n_estimators"}
            booster = xgb.train(
                p, dtrain, num_boost_round=n_rounds,
                evals=[(dval, "val")], early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose_eval=False,
            )
            best_it = getattr(booster, "best_iteration", None) or (n_rounds - 1)
            pred_va = np.clip(booster.predict(dval, iteration_range=(0, best_it + 1)), *E10_CLIP)
            oof_ll[target][val_pos] = pred_va
            oof_mask[val_pos] = True
            fold_metrics[f"{target}_ll"] = float(log_loss(y_va, pred_va))
            fold_metrics[f"{target}_auc"] = (
                float(roc_auc_score(y_va, pred_va)) if len(np.unique(y_va)) > 1 else 0.5
            )
        fw = calculate_weighted_score(
            fold_metrics["adopted_within_07_days_auc"], fold_metrics["adopted_within_07_days_ll"],
            fold_metrics["adopted_within_90_days_auc"], fold_metrics["adopted_within_90_days_ll"],
            fold_metrics["adopted_within_120_days_auc"], fold_metrics["adopted_within_120_days_ll"],
        )
        per_fold_weighted_scores.append(fw)
        logger.info("  Fold %d: train=%d val=%d weighted=%.6f", fold, len(train_pos), len(val_pos), fw)

    _cal = ll_calibration_per_target or {}
    def _m(t: str) -> CalibrationMethod:
        return _cal.get(t, CalibrationMethod.ISOTONIC)
    oof_07_ll_r = oof_ll[TARGET_COLS[0]][oof_mask]
    oof_90_ll_r = oof_ll[TARGET_COLS[1]][oof_mask]
    oof_120_ll_r = oof_ll[TARGET_COLS[2]][oof_mask]
    y07 = train_df[TARGET_COLS[0]].values[oof_mask]
    y90 = train_df[TARGET_COLS[1]].values[oof_mask]
    y120 = train_df[TARGET_COLS[2]].values[oof_mask]
    c07 = Calibrator(_m(TARGET_COLS[0]))
    c07.fit(y07, oof_07_ll_r)
    c90 = Calibrator(_m(TARGET_COLS[1]))
    c90.fit(y90, oof_90_ll_r)
    c120 = Calibrator(_m(TARGET_COLS[2]))
    c120.fit(y120, oof_120_ll_r)
    oof_07_ll_cal = c07.transform(oof_07_ll_r)
    oof_90_ll_cal = c90.transform(oof_90_ll_r)
    oof_120_ll_cal = c120.transform(oof_120_ll_r)
    oof_07_ll_cal, oof_90_ll_cal, oof_120_ll_cal = Calibrator.enforce_hierarchy(
        oof_07_ll_cal, oof_90_ll_cal, oof_120_ll_cal
    )
    oof_07_auc = oof_07_ll_cal
    oof_90_auc = oof_90_ll_cal
    oof_120_auc = oof_120_ll_cal
    oof_07_ll = float(log_loss(y07, oof_07_ll_cal))
    oof_07_auc_s = float(roc_auc_score(y07, oof_07_auc)) if len(np.unique(y07)) > 1 else 0.5
    oof_90_ll = float(log_loss(y90, oof_90_ll_cal))
    oof_90_auc_s = float(roc_auc_score(y90, oof_90_auc)) if len(np.unique(y90)) > 1 else 0.5
    oof_120_ll = float(log_loss(y120, oof_120_ll_cal))
    oof_120_auc_s = float(roc_auc_score(y120, oof_120_auc)) if len(np.unique(y120)) > 1 else 0.5
    ws = calculate_weighted_score(oof_07_auc_s, oof_07_ll, oof_90_auc_s, oof_90_ll, oof_120_auc_s, oof_120_ll)
    fold_std = float(np.std(per_fold_weighted_scores)) if len(per_fold_weighted_scores) > 1 else 0.0
    worst = min(per_fold_weighted_scores) if per_fold_weighted_scores else 0.0
    last = per_fold_weighted_scores[-1] if per_fold_weighted_scores else 0.0
    return {
        "weighted_score": ws,
        "per_fold_weighted_scores": per_fold_weighted_scores,
        "fold_std": fold_std,
        "worst_fold_score": worst,
        "last_fold_score": last,
        "oof_07_ll": oof_07_ll,
        "oof_07_auc": oof_07_auc_s,
        "oof_90_ll": oof_90_ll,
        "oof_90_auc": oof_90_auc_s,
        "oof_120_ll": oof_120_ll,
        "oof_120_auc": oof_120_auc_s,
    }


def run_single_e10_harness_catboost(
    train_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    n_splits: int = E10_N_SPLITS,
    seed: int = E10_SEED,
    *,
    ll_params_per_target: dict[str, dict] | None = None,
    auc_params_per_target: dict[str, dict] | None = None,
    num_boost_round: int | None = None,
    feature_groups: dict[str, bool] | None = None,
    ll_calibration_per_target: dict[str, CalibrationMethod] | None = None,
) -> dict[str, Any]:
    """Single model per target (CatBoost); vetted features (E10.5). Same prediction for LL and AUC."""
    from catboost import CatBoostClassifier
    from sklearn.metrics import log_loss, roc_auc_score

    np.random.seed(seed)
    train_df = train_df.copy()
    prior_df = prior_df.copy()
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    if ll_params_per_target is None:
        ll_params_per_target = {t: _default_catboost_ll_params(seed) for t in TARGET_COLS}
    n_rounds = num_boost_round if num_boost_round is not None else NUM_BOOST_ROUND
    feature_groups = feature_groups or VETTED_FEATURE_GROUPS

    splits = rolling_forward_splits(
        train_df, n_splits=n_splits, min_train_size=E10_MIN_TRAIN_SIZE
    )
    n_rows = len(train_df)
    oof_ll = {t: np.zeros(n_rows, dtype=float) for t in TARGET_COLS}
    oof_mask = np.zeros(n_rows, dtype=bool)
    per_fold_weighted_scores: list[float] = []

    for split in splits:
        fold, train_pos, val_pos, train_cutoff = (
            split["fold"], split["train_pos"], split["val_pos"], split["train_cutoff"],
        )
        prior_asof = prior_df[prior_df[DATE_COL] < train_cutoff].copy()
        fe = FeatureEngineer(prior_asof)
        fe.set_feature_groups(feature_groups)
        fe.fit(train_df.iloc[train_pos])
        X_tr = fe.transform(train_df.iloc[train_pos])
        X_va = fe.transform(train_df.iloc[val_pos])
        fold_metrics: dict[str, float] = {}
        for target in TARGET_COLS:
            y_tr = train_df[target].iloc[train_pos].values
            y_va = train_df[target].iloc[val_pos].values
            p = dict(ll_params_per_target[target])
            p["iterations"] = p.get("iterations", n_rounds)
            p["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
            model = CatBoostClassifier(**p)
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
            pred_va = np.clip(model.predict_proba(X_va)[:, 1], *E10_CLIP)
            oof_ll[target][val_pos] = pred_va
            oof_mask[val_pos] = True
            fold_metrics[f"{target}_ll"] = float(log_loss(y_va, pred_va))
            fold_metrics[f"{target}_auc"] = (
                float(roc_auc_score(y_va, pred_va)) if len(np.unique(y_va)) > 1 else 0.5
            )
        fw = calculate_weighted_score(
            fold_metrics["adopted_within_07_days_auc"], fold_metrics["adopted_within_07_days_ll"],
            fold_metrics["adopted_within_90_days_auc"], fold_metrics["adopted_within_90_days_ll"],
            fold_metrics["adopted_within_120_days_auc"], fold_metrics["adopted_within_120_days_ll"],
        )
        per_fold_weighted_scores.append(fw)
        logger.info("  Fold %d: train=%d val=%d weighted=%.6f", fold, len(train_pos), len(val_pos), fw)

    _cal = ll_calibration_per_target or {}
    def _m(t: str) -> CalibrationMethod:
        return _cal.get(t, CalibrationMethod.ISOTONIC)
    oof_07_ll_r = oof_ll[TARGET_COLS[0]][oof_mask]
    oof_90_ll_r = oof_ll[TARGET_COLS[1]][oof_mask]
    oof_120_ll_r = oof_ll[TARGET_COLS[2]][oof_mask]
    y07 = train_df[TARGET_COLS[0]].values[oof_mask]
    y90 = train_df[TARGET_COLS[1]].values[oof_mask]
    y120 = train_df[TARGET_COLS[2]].values[oof_mask]
    c07, c90, c120 = Calibrator(_m(TARGET_COLS[0])), Calibrator(_m(TARGET_COLS[1])), Calibrator(_m(TARGET_COLS[2]))
    c07.fit(y07, oof_07_ll_r)
    c90.fit(y90, oof_90_ll_r)
    c120.fit(y120, oof_120_ll_r)
    oof_07_ll_cal = c07.transform(oof_07_ll_r)
    oof_90_ll_cal = c90.transform(oof_90_ll_r)
    oof_120_ll_cal = c120.transform(oof_120_ll_r)
    oof_07_ll_cal, oof_90_ll_cal, oof_120_ll_cal = Calibrator.enforce_hierarchy(
        oof_07_ll_cal, oof_90_ll_cal, oof_120_ll_cal
    )
    oof_07_auc = oof_07_ll_cal
    oof_90_auc = oof_90_ll_cal
    oof_120_auc = oof_120_ll_cal
    oof_07_ll = float(log_loss(y07, oof_07_ll_cal))
    oof_07_auc_s = float(roc_auc_score(y07, oof_07_auc)) if len(np.unique(y07)) > 1 else 0.5
    oof_90_ll = float(log_loss(y90, oof_90_ll_cal))
    oof_90_auc_s = float(roc_auc_score(y90, oof_90_auc)) if len(np.unique(y90)) > 1 else 0.5
    oof_120_ll = float(log_loss(y120, oof_120_ll_cal))
    oof_120_auc_s = float(roc_auc_score(y120, oof_120_auc)) if len(np.unique(y120)) > 1 else 0.5
    ws = calculate_weighted_score(oof_07_auc_s, oof_07_ll, oof_90_auc_s, oof_90_ll, oof_120_auc_s, oof_120_ll)
    fold_std = float(np.std(per_fold_weighted_scores)) if len(per_fold_weighted_scores) > 1 else 0.0
    worst = min(per_fold_weighted_scores) if per_fold_weighted_scores else 0.0
    last = per_fold_weighted_scores[-1] if per_fold_weighted_scores else 0.0
    return {
        "weighted_score": ws,
        "per_fold_weighted_scores": per_fold_weighted_scores,
        "fold_std": fold_std,
        "worst_fold_score": worst,
        "last_fold_score": last,
        "oof_07_ll": oof_07_ll,
        "oof_07_auc": oof_07_auc_s,
        "oof_90_ll": oof_90_ll,
        "oof_90_auc": oof_90_auc_s,
        "oof_120_ll": oof_120_ll,
        "oof_120_auc": oof_120_auc_s,
    }


def run_baseline(args: argparse.Namespace) -> int:
    """E10.0: Lock benchmark, 3 runs, pass if delta < 0.002, save baseline."""
    os.environ["DIGICOW_MINIMAL_FEATURES"] = "1"

    logger.info("=" * 70)
    logger.info("E10.0 LOCK BENCHMARK")
    logger.info(
        "  Config: Plan4 dual heads + minimal features, E0 forward folds, as-of prior"
    )
    logger.info("  Seed=%s, n_splits=%s, clip=%s", E10_SEED, args.n_splits, E10_CLIP)
    logger.info("  Pass: max(run scores) - min(run scores) < %s", args.threshold)
    logger.info("=" * 70)

    loader = DataLoader()
    train_df, _test_df, prior_df, _ = loader.load_all()
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])

    if args.dry_run:
        result = run_single_e10_harness(
            train_df, prior_df, n_splits=args.n_splits, seed=E10_SEED
        )
        logger.info("  Weighted score (primary): %.6f", result["weighted_score"])
        logger.info(
            "  7d  LogLoss=%.6f  AUC=%.6f",
            result["oof_07_ll"],
            result["oof_07_auc"],
        )
        logger.info(
            "  90d LogLoss=%.6f  AUC=%.6f",
            result["oof_90_ll"],
            result["oof_90_auc"],
        )
        logger.info(
            "  120d LogLoss=%.6f AUC=%.6f",
            result["oof_120_ll"],
            result["oof_120_auc"],
        )
        logger.info("  Fold std (guardrail):      %.6f", result["fold_std"])
        logger.info("  Worst fold score:          %.6f", result["worst_fold_score"])
        logger.info("  Last fold score:           %.6f", result["last_fold_score"])
        return 0

    scores: list[float] = []
    results_list: list[dict[str, Any]] = []
    for r in range(args.runs):
        logger.info("\n--- Run %d / %d ---", r + 1, args.runs)
        result = run_single_e10_harness(
            train_df, prior_df, n_splits=args.n_splits, seed=E10_SEED
        )
        scores.append(result["weighted_score"])
        results_list.append(result)
        logger.info(
            "  Run %d weighted=%.6f 7d_ll=%.6f 7d_auc=%.6f (fold_std=%.6f)",
            r + 1,
            result["weighted_score"],
            result["oof_07_ll"],
            result["oof_07_auc"],
            result["fold_std"],
        )

    max_delta = max(scores) - min(scores)
    passed = max_delta < args.threshold

    logger.info("\n" + "=" * 70)
    logger.info("E10.0 RESULT")
    logger.info("  Scores: %s", [round(s, 6) for s in scores])
    logger.info("  Max delta: %.6f", max_delta)
    logger.info("  Threshold: %.6f", args.threshold)
    if passed:
        logger.info("  PASS: Reproducible baseline (delta < threshold)")
        baseline_score = float(np.mean(scores))
        logger.info("  Baseline score (mean): %.6f", baseline_score)
        # Use first run for guardrail reference (deterministic)
        ref = results_list[0]
        baseline_payload = {
            "weighted_score": baseline_score,
            "oof_07_ll": ref["oof_07_ll"],
            "oof_07_auc": ref["oof_07_auc"],
            "oof_90_ll": ref["oof_90_ll"],
            "oof_90_auc": ref["oof_90_auc"],
            "oof_120_ll": ref["oof_120_ll"],
            "oof_120_auc": ref["oof_120_auc"],
            "per_fold_weighted_scores": ref["per_fold_weighted_scores"],
            "fold_std": ref["fold_std"],
            "worst_fold_score": ref["worst_fold_score"],
        }
        E10_BASELINE_PATH.write_text(
            json.dumps(baseline_payload, indent=2), encoding="utf-8"
        )
        logger.info("  Baseline frozen to %s", E10_BASELINE_PATH)
    else:
        logger.warning(
            "  FAIL: Score changed by >= %.6f across runs. Fix before E10.1.",
            args.threshold,
        )
    logger.info("=" * 70)

    return 0 if passed else 1


def _load_baseline() -> dict[str, Any]:
    if not E10_BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"E10 baseline not found at {E10_BASELINE_PATH}. Run E10.0 first: "
            "python e10_dual_hpo.py baseline  (or use hpo --run-baseline-if-missing)"
        )
    return json.loads(E10_BASELINE_PATH.read_text(encoding="utf-8"))


def _run_and_save_baseline_once(
    train_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    n_splits: int = E10_N_SPLITS,
) -> dict[str, Any]:
    """Run one E10 harness and save result as baseline (e.g. for Colab when no prior run)."""
    logger.info("Running one E10.0 harness to create baseline at %s ...", E10_BASELINE_PATH)
    result = run_single_e10_harness(
        train_df, prior_df, n_splits=n_splits, seed=E10_SEED
    )
    payload = {
        "weighted_score": result["weighted_score"],
        "oof_07_ll": result["oof_07_ll"],
        "oof_07_auc": result["oof_07_auc"],
        "oof_90_ll": result["oof_90_ll"],
        "oof_90_auc": result["oof_90_auc"],
        "oof_120_ll": result["oof_120_ll"],
        "oof_120_auc": result["oof_120_auc"],
        "per_fold_weighted_scores": result["per_fold_weighted_scores"],
        "fold_std": result["fold_std"],
        "worst_fold_score": result["worst_fold_score"],
    }
    E10_BASELINE_PATH.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    logger.info("  Baseline saved: weighted=%.6f", result["weighted_score"])
    return payload


def run_hpo(args: argparse.Namespace, sampler=None) -> int:
    """E10.1: LGBM dual HPO with Optuna, guardrails, pass rule Δ >= +0.004.

    sampler: Optuna sampler (e.g. RandomSampler(seed=...) for random search).
             If None, uses default TPESampler.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    os.environ["DIGICOW_MINIMAL_FEATURES"] = "1"

    loader = DataLoader()
    train_df, _test_df, prior_df, _ = loader.load_all()
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])

    if not E10_BASELINE_PATH.exists() and getattr(args, "run_baseline_if_missing", False):
        _run_and_save_baseline_once(train_df, prior_df, n_splits=args.n_splits)
    baseline = _load_baseline()
    baseline_weighted = baseline["weighted_score"]
    baseline_7d_ll = baseline["oof_07_ll"]
    baseline_worst_fold = baseline["worst_fold_score"]
    baseline_fold_std = baseline["fold_std"]

    logger.info("=" * 70)
    logger.info("E10.1 LGBM DUAL HPO (minimal features)")
    logger.info("  Baseline weighted: %.6f (from %s)", baseline_weighted, E10_BASELINE_PATH)
    logger.info("  Guardrails: 7d_ll not worse, worst_fold not worse, fold_std controlled")
    logger.info("  Pass rule: Δweighted >= +%.4f", E10_PASS_DELTA_HPO)
    logger.info("=" * 70)

    # Quiet per-fold and feature-engineering logs during HPO (summary only)
    _noisy_loggers = [
        logging.getLogger("shared.feature_engineering"),
        logging.getLogger(__name__),
    ]
    _saved_levels = [log.level for log in _noisy_loggers]
    for log in _noisy_loggers:
        log.setLevel(logging.WARNING)

    # Study key: which target to tune (7d, 90d, 120d). Single model per target.
    def _make_objective(study_key: str):
        """Build Optuna objective for one target. Only that target's params are tuned."""

        def objective(trial: optuna.Trial) -> float:
            ll_params_per_target = {t: _default_ll_params(E10_SEED) for t in TARGET_COLS}
            num_rounds = trial.suggest_int("num_boost_round", 300, 1500)
            base = _default_ll_params(E10_SEED)
            tuned = {
                **base,
                "learning_rate": trial.suggest_float("lr", 0.01, 0.08),
                "num_leaves": trial.suggest_int("num_leaves", 15, 45),
                "max_depth": trial.suggest_int("max_depth", 4, 7),
                "min_child_samples": trial.suggest_int("min_child_samples", 30, 80),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.1, 2.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.5, 3.0),
            }
            if study_key == "7d":
                ll_params_per_target[TARGET_COLS[0]] = tuned
            elif study_key == "90d":
                ll_params_per_target[TARGET_COLS[1]] = tuned
            else:
                ll_params_per_target[TARGET_COLS[2]] = tuned

            result = run_single_e10_harness(
                train_df,
                prior_df,
                n_splits=args.n_splits,
                seed=E10_SEED,
                ll_params_per_target=ll_params_per_target,
                num_boost_round=num_rounds,
            )
            score = result["weighted_score"]
            if result["oof_07_ll"] > baseline_7d_ll + 1e-6:
                return 0.0
            if result["worst_fold_score"] < baseline_worst_fold - 1e-6:
                return 0.0
            if result["fold_std"] > baseline_fold_std * 1.5 + 0.01:
                return 0.0
            return score

        return objective

    studies_to_run = (
        ["7d", "90d", "120d"]
        if args.study == "all"
        else [args.study]
    )

    best_overall = baseline_weighted
    best_params: dict[str, dict] = {}

    for sk in studies_to_run:
        logger.info("\n--- Optuna study: %s (n_trials=%d) ---", sk, args.n_trials)
        study = optuna.create_study(
            direction="maximize",
            study_name=f"e10_{sk}",
            sampler=sampler,
        )
        study.optimize(_make_objective(sk), n_trials=args.n_trials, show_progress_bar=True)
        if study.best_trial is not None and study.best_value > baseline_weighted:
            logger.info(
                "  Best %s: weighted=%.6f (Δ=%.6f)",
                sk,
                study.best_value,
                study.best_value - baseline_weighted,
            )
            if study.best_value > best_overall:
                best_overall = study.best_value
            best_params[sk] = study.best_params
        else:
            logger.info("  No improvement over baseline for %s", sk)

    # Restore log levels
    for log, level in zip(_noisy_loggers, _saved_levels):
        log.setLevel(level)

    delta = best_overall - baseline_weighted
    passed = delta >= E10_PASS_DELTA_HPO

    logger.info("\n" + "=" * 70)
    logger.info("E10.1 RESULT")
    logger.info("  Baseline weighted: %.6f", baseline_weighted)
    logger.info("  Best weighted:     %.6f", best_overall)
    logger.info("  Δ:                 %.6f", delta)
    logger.info("  Pass (Δ >= %.4f): %s", E10_PASS_DELTA_HPO, "PASS" if passed else "FAIL")
    if best_params:
        logger.info("  Best params keys: %s", list(best_params.keys()))
    # Persist E10.1 best for E10.2 comparison (keep only if better than E10.2)
    E10_1_BEST_PATH.write_text(
        json.dumps({"weighted_score": best_overall, "best_params": best_params}, indent=2),
        encoding="utf-8",
    )
    logger.info("  E10.1 best saved to %s", E10_1_BEST_PATH)
    logger.info("=" * 70)

    return 0 if passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E10 dual-head HPO: E10.0 baseline lock, E10.1 LGBM HPO"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # E10.0
    p_baseline = sub.add_parser("baseline", help="E10.0: Lock benchmark (3 runs, save baseline)")
    p_baseline.add_argument("--runs", type=int, default=3)
    p_baseline.add_argument(
        "--threshold",
        type=float,
        default=E10_PASS_DELTA_REPRO,
        help="Max score delta for reproducibility pass",
    )
    p_baseline.add_argument("--dry-run", action="store_true", help="Single run, no pass/fail")
    p_baseline.add_argument("--n-splits", type=int, default=E10_N_SPLITS)

    # E10.1
    p_hpo = sub.add_parser("hpo", help="E10.1: LGBM dual HPO with Optuna")
    p_hpo.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Optuna trials per study",
    )
    p_hpo.add_argument(
        "--study",
        type=str,
        default="7d",
        choices=["7d", "90d", "120d", "all"],
        help="Which head(s) to tune (one study per head if 'all')",
    )
    p_hpo.add_argument("--n-splits", type=int, default=E10_N_SPLITS)
    p_hpo.add_argument(
        "--run-baseline-if-missing",
        action="store_true",
        help="If e10_baseline.json is missing, run one E10.0 harness and save it (e.g. on Colab)",
    )

    args = parser.parse_args()

    if args.command == "baseline":
        sys.exit(run_baseline(args))
    if args.command == "hpo":
        sys.exit(run_hpo(args))
    sys.exit(1)


if __name__ == "__main__":
    main()
