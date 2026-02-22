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
  python e10_dual_hpo.py hpo [--n-trials 50] [--study 7d_ll]  # E10.1
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
from shared.feature_engineering import MINIMAL_FEATURE_GROUPS, FeatureEngineer

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
E10_PASS_DELTA_REPRO = 0.002
E10_PASS_DELTA_HPO = 0.004


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
) -> dict[str, Any]:
    """
    One run of E10 harness: Plan4-style dual heads (LL + AUC per target),
    E0-style forward folds and as-of features, minimal features.
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
    if auc_params_per_target is None:
        auc_params_per_target = {t: _default_auc_params(seed) for t in TARGET_COLS}

    n_rounds = num_boost_round if num_boost_round is not None else NUM_BOOST_ROUND

    splits = rolling_forward_splits(
        train_df, n_splits=n_splits, min_train_size=E10_MIN_TRAIN_SIZE
    )
    n_rows = len(train_df)

    # OOF arrays: LL and AUC head per target
    oof_ll: dict[str, np.ndarray] = {t: np.zeros(n_rows, dtype=float) for t in TARGET_COLS}
    oof_auc: dict[str, np.ndarray] = {t: np.zeros(n_rows, dtype=float) for t in TARGET_COLS}
    oof_mask = np.zeros(n_rows, dtype=bool)

    per_fold_weighted_scores: list[float] = []

    for split in splits:
        fold = split["fold"]
        train_pos = split["train_pos"]
        val_pos = split["val_pos"]
        train_cutoff = split["train_cutoff"]

        prior_asof = prior_df[prior_df[DATE_COL] < train_cutoff].copy()
        fe = FeatureEngineer(prior_asof)
        fe.set_feature_groups(MINIMAL_FEATURE_GROUPS)
        fe.fit(train_df.iloc[train_pos])

        X_tr = fe.transform(train_df.iloc[train_pos])
        X_va = fe.transform(train_df.iloc[val_pos])

        fold_metrics: dict[str, float] = {}

        for target in TARGET_COLS:
            y = train_df[target]
            y_tr = y.iloc[train_pos].values
            y_va = y.iloc[val_pos].values

            # LL head
            train_set_ll = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
            val_set_ll = lgb.Dataset(X_va, label=y_va, free_raw_data=False)
            booster_ll = lgb.train(
                params=ll_params_per_target[target],
                train_set=train_set_ll,
                num_boost_round=n_rounds,
                valid_sets=[train_set_ll, val_set_ll],
                valid_names=["train", "valid"],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            pred_ll_va = booster_ll.predict(X_va, num_iteration=booster_ll.best_iteration)
            pred_ll_va = np.clip(pred_ll_va, *E10_CLIP)
            oof_ll[target][val_pos] = pred_ll_va

            # AUC head
            train_set_auc = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
            val_set_auc = lgb.Dataset(X_va, label=y_va, free_raw_data=False)
            booster_auc = lgb.train(
                params=auc_params_per_target[target],
                train_set=train_set_auc,
                num_boost_round=n_rounds,
                valid_sets=[train_set_auc, val_set_auc],
                valid_names=["train", "valid"],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            pred_auc_va = booster_auc.predict(X_va, num_iteration=booster_auc.best_iteration)
            pred_auc_va = np.clip(pred_auc_va, *E10_CLIP)
            oof_auc[target][val_pos] = pred_auc_va

            oof_mask[val_pos] = True

            ll_val = log_loss(y_va, pred_ll_va)
            auc_val = (
                roc_auc_score(y_va, pred_auc_va)
                if len(np.unique(y_va)) > 1
                else 0.5
            )
            fold_metrics[f"{target}_ll"] = ll_val
            fold_metrics[f"{target}_auc"] = auc_val

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

    # Calibrate LL OOF per target (isotonic like Plan4)
    oof_07_ll_raw = oof_ll[TARGET_COLS[0]][oof_mask]
    oof_90_ll_raw = oof_ll[TARGET_COLS[1]][oof_mask]
    oof_120_ll_raw = oof_ll[TARGET_COLS[2]][oof_mask]
    y_07 = train_df[TARGET_COLS[0]].values[oof_mask]
    y_90 = train_df[TARGET_COLS[1]].values[oof_mask]
    y_120 = train_df[TARGET_COLS[2]].values[oof_mask]

    cal_07 = Calibrator(CalibrationMethod.ISOTONIC)
    cal_07.fit(y_07, oof_07_ll_raw)
    oof_07_ll_cal = cal_07.transform(oof_07_ll_raw)
    cal_90 = Calibrator(CalibrationMethod.ISOTONIC)
    cal_90.fit(y_90, oof_90_ll_raw)
    oof_90_ll_cal = cal_90.transform(oof_90_ll_raw)
    cal_120 = Calibrator(CalibrationMethod.ISOTONIC)
    cal_120.fit(y_120, oof_120_ll_raw)
    oof_120_ll_cal = cal_120.transform(oof_120_ll_raw)

    oof_07_auc = np.clip(oof_auc[TARGET_COLS[0]][oof_mask], *E10_CLIP)
    oof_90_auc = np.clip(oof_auc[TARGET_COLS[1]][oof_mask], *E10_CLIP)
    oof_120_auc = np.clip(oof_auc[TARGET_COLS[2]][oof_mask], *E10_CLIP)

    # Enforce hierarchy on calibrated LL and AUC
    oof_07_ll_cal, oof_90_ll_cal, oof_120_ll_cal = Calibrator.enforce_hierarchy(
        oof_07_ll_cal, oof_90_ll_cal, oof_120_ll_cal
    )
    oof_07_auc, oof_90_auc, oof_120_auc = Calibrator.enforce_hierarchy(
        oof_07_auc, oof_90_auc, oof_120_auc
    )

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
            "python e10_dual_hpo.py baseline"
        )
    return json.loads(E10_BASELINE_PATH.read_text(encoding="utf-8"))


def run_hpo(args: argparse.Namespace) -> int:
    """E10.1: LGBM dual HPO with Optuna, guardrails, pass rule Δ >= +0.004."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    os.environ["DIGICOW_MINIMAL_FEATURES"] = "1"

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

    loader = DataLoader()
    train_df, _test_df, prior_df, _ = loader.load_all()
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])

    # Study key: which head to tune (one at a time; others use defaults)
    def _make_objective(study_key: str):
        """Build Optuna objective for a single head (e.g. 7d_ll). Only that head is tuned."""

        def objective(trial: optuna.Trial) -> float:
            ll_params_per_target = {t: _default_ll_params(E10_SEED) for t in TARGET_COLS}
            auc_params_per_target = {t: _default_auc_params(E10_SEED) for t in TARGET_COLS}
            num_rounds = trial.suggest_int("num_boost_round", 300, 1500)

            if study_key == "7d_ll":
                ll_params_per_target[TARGET_COLS[0]] = {
                    **_default_ll_params(E10_SEED),
                    "learning_rate": trial.suggest_float("lr", 0.01, 0.08),
                    "num_leaves": trial.suggest_int("num_leaves", 15, 45),
                    "max_depth": trial.suggest_int("max_depth", 4, 7),
                    "min_child_samples": trial.suggest_int("min_child_samples", 30, 80),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                    "lambda_l1": trial.suggest_float("lambda_l1", 0.1, 2.0),
                    "lambda_l2": trial.suggest_float("lambda_l2", 0.5, 3.0),
                }
            elif study_key == "7d_auc":
                auc_params_per_target[TARGET_COLS[0]] = {
                    **_default_auc_params(E10_SEED),
                    "learning_rate": trial.suggest_float("lr", 0.02, 0.08),
                    "num_leaves": trial.suggest_int("num_leaves", 40, 90),
                    "max_depth": trial.suggest_int("max_depth", 5, 8),
                    "min_child_samples": trial.suggest_int("min_child_samples", 15, 40),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                    "lambda_l1": trial.suggest_float("lambda_l1", 0.02, 0.3),
                    "lambda_l2": trial.suggest_float("lambda_l2", 0.2, 1.0),
                }
            elif study_key == "90d_ll":
                ll_params_per_target[TARGET_COLS[1]] = {
                    **_default_ll_params(E10_SEED),
                    "learning_rate": trial.suggest_float("lr", 0.01, 0.08),
                    "num_leaves": trial.suggest_int("num_leaves", 15, 45),
                    "max_depth": trial.suggest_int("max_depth", 4, 7),
                    "min_child_samples": trial.suggest_int("min_child_samples", 30, 80),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                    "lambda_l1": trial.suggest_float("lambda_l1", 0.1, 2.0),
                    "lambda_l2": trial.suggest_float("lambda_l2", 0.5, 3.0),
                }
            elif study_key == "90d_auc":
                auc_params_per_target[TARGET_COLS[1]] = {
                    **_default_auc_params(E10_SEED),
                    "learning_rate": trial.suggest_float("lr", 0.02, 0.08),
                    "num_leaves": trial.suggest_int("num_leaves", 40, 90),
                    "max_depth": trial.suggest_int("max_depth", 5, 8),
                    "min_child_samples": trial.suggest_int("min_child_samples", 15, 40),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                    "lambda_l1": trial.suggest_float("lambda_l1", 0.02, 0.3),
                    "lambda_l2": trial.suggest_float("lambda_l2", 0.2, 1.0),
                }
            elif study_key == "120d_ll":
                ll_params_per_target[TARGET_COLS[2]] = {
                    **_default_ll_params(E10_SEED),
                    "learning_rate": trial.suggest_float("lr", 0.01, 0.08),
                    "num_leaves": trial.suggest_int("num_leaves", 15, 45),
                    "max_depth": trial.suggest_int("max_depth", 4, 7),
                    "min_child_samples": trial.suggest_int("min_child_samples", 30, 80),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                    "lambda_l1": trial.suggest_float("lambda_l1", 0.1, 2.0),
                    "lambda_l2": trial.suggest_float("lambda_l2", 0.5, 3.0),
                }
            elif study_key == "120d_auc":
                auc_params_per_target[TARGET_COLS[2]] = {
                    **_default_auc_params(E10_SEED),
                    "learning_rate": trial.suggest_float("lr", 0.02, 0.08),
                    "num_leaves": trial.suggest_int("num_leaves", 40, 90),
                    "max_depth": trial.suggest_int("max_depth", 5, 8),
                    "min_child_samples": trial.suggest_int("min_child_samples", 15, 40),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                    "lambda_l1": trial.suggest_float("lambda_l1", 0.02, 0.3),
                    "lambda_l2": trial.suggest_float("lambda_l2", 0.2, 1.0),
                }

            result = run_single_e10_harness(
                train_df,
                prior_df,
                n_splits=args.n_splits,
                seed=E10_SEED,
                ll_params_per_target=ll_params_per_target,
                auc_params_per_target=auc_params_per_target,
                num_boost_round=num_rounds,
            )
            score = result["weighted_score"]
            # Guardrails: fail trial if constraints violated
            if result["oof_07_ll"] > baseline_7d_ll + 1e-6:
                return 0.0
            if result["worst_fold_score"] < baseline_worst_fold - 1e-6:
                return 0.0
            max_std = baseline_fold_std * 1.5 + 0.01
            if result["fold_std"] > max_std:
                return 0.0
            return score

        return objective

    studies_to_run = (
        ["7d_ll", "7d_auc", "90d_ll", "90d_auc", "120d_ll", "120d_auc"]
        if args.study == "all"
        else [args.study]
    )

    best_overall = baseline_weighted
    best_params: dict[str, dict] = {}

    for sk in studies_to_run:
        logger.info("\n--- Optuna study: %s (n_trials=%d) ---", sk, args.n_trials)
        study = optuna.create_study(direction="maximize", study_name=f"e10_{sk}")
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
        default="7d_ll",
        choices=[
            "7d_ll",
            "7d_auc",
            "90d_ll",
            "90d_auc",
            "120d_ll",
            "120d_auc",
            "all",
        ],
        help="Which head(s) to tune (one study per head if 'all')",
    )
    p_hpo.add_argument("--n-splits", type=int, default=E10_N_SPLITS)

    args = parser.parse_args()

    if args.command == "baseline":
        sys.exit(run_baseline(args))
    if args.command == "hpo":
        sys.exit(run_hpo(args))
    sys.exit(1)


if __name__ == "__main__":
    main()
