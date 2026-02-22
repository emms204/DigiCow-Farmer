#!/usr/bin/env python3
"""
E5 Leakage-safe encodings: as-of Bayesian target encodings with min-support and uncertainty.

Hypothesis E5: Safe target encodings (per-fold as-of, min-support, SE/lower bound) help.
Experiment: Add Bayesian-smoothed adoption rates (per trainer/group/county) computed only
            on train data before fold cutoff; min-support; SE and lower-bound features.
Primary pass metric: Weighted score gain and fold stability.
Pass: ΔWeighted ≥ +0.003 and ΔStd ≤ -0.002 (fold std of per_fold_weighted_scores).
If fail: Drop encoding block.

Usage:
    python e5_safe_encodings.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.constants import DATE_COL, TARGET_COLS
from shared.data_loader import DataLoader
from shared.evaluation import calculate_weighted_score
from shared.feature_engineering import MINIMAL_FEATURE_GROUPS, FeatureEngineer
from sklearn.metrics import log_loss, roc_auc_score

from e0_reproducibility import (
    E0_CLIP,
    E0_MIN_TRAIN_SIZE,
    E0_N_SPLITS,
    E0_SEED,
    rolling_forward_splits,
    run_single_e0_harness,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

E5_WEIGHTED_DELTA_MIN = 0.003   # weighted score must improve by at least this
E5_STD_DELTA_MAX = -0.002       # fold std must decrease by at least 0.002 (more stable)
E5_MIN_SUPPORT = 10             # min samples per group to use group encoding; else global
E5_ENCODING_GROUPS = ("trainer", "group_name", "county")  # columns for target encoding


def _build_asof_target_encodings(
    ref: pd.DataFrame,
    group_col: str,
    prefix: str,
    min_support: int = E5_MIN_SUPPORT,
) -> pd.DataFrame:
    """Build Bayesian-smoothed target encodings from ref (as-of train slice).

    Returns DataFrame with columns: group_col, and for each target:
      {prefix}_{short}_rate, {prefix}_{short}_se, {prefix}_{short}_lower95
    where short is 07, 90, 120. Groups with count < min_support get global rate and high SE.
    """
    if group_col not in ref.columns:
        return pd.DataFrame()

    grouped = ref.groupby(group_col)
    stats = pd.DataFrame({group_col: list(grouped.groups.keys())})
    stats = stats.set_index(group_col)

    short_names = {"adopted_within_07_days": "07", "adopted_within_90_days": "90", "adopted_within_120_days": "120"}
    global_rates = {col: ref[col].mean() for col in TARGET_COLS if col in ref.columns}

    for col in TARGET_COLS:
        if col not in ref.columns:
            continue
        short = short_names.get(col, col)
        smoothed_rates = []
        ses = []
        lower_bounds = []
        counts = []

        for group_name, group_data in grouped:
            n = len(group_data)
            counts.append(n)
            successes = group_data[col].sum()
            rate = successes / n if n > 0 else 0.0
            global_rate = global_rates[col]
            alpha_prior = global_rate * 10
            beta_prior = (1 - global_rate) * 10
            alpha_post = alpha_prior + successes
            beta_post = beta_prior + (n - successes)
            smoothed_rate = alpha_post / (alpha_post + beta_post)
            se = np.sqrt(smoothed_rate * (1 - smoothed_rate) / (alpha_post + beta_post))
            if n < min_support:
                smoothed_rate = global_rate
                se = 1.0  # high uncertainty
            smoothed_rates.append(smoothed_rate)
            ses.append(se)
            lower_bounds.append(max(0.0, smoothed_rate - 1.96 * se))

        stats[f"{prefix}_{short}_rate"] = smoothed_rates
        stats[f"{prefix}_{short}_se"] = ses
        stats[f"{prefix}_{short}_lower95"] = lower_bounds
        stats[f"{prefix}_{short}_count"] = counts

    stats = stats.reset_index()
    return stats


def _add_encoding_features(
    df: pd.DataFrame,
    encoding_dfs: dict[str, pd.DataFrame],
    group_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Merge encoding tables onto df. Missing groups get NaN; caller can fill with global."""
    out = pd.DataFrame(index=df.index)
    for group_col in group_columns:
        enc = encoding_dfs.get(group_col)
        if enc is None or enc.empty or group_col not in df.columns:
            continue
        merge_cols = [c for c in enc.columns if c != group_col]
        merged = df[[group_col]].merge(enc, on=group_col, how="left")
        for c in merge_cols:
            out[c] = merged[c].values
    return out


def run_e5_encoding_harness(
    train_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    n_splits: int = E0_N_SPLITS,
    seed: int = E0_SEED,
) -> dict[str, Any]:
    """Run E0-style harness with minimal features + as-of Bayesian target encodings per fold."""
    import lightgbm as lgb
    from plan1.config import EARLY_STOPPING_ROUNDS, LGBM_PARAMS, NUM_BOOST_ROUND

    np.random.seed(seed)
    train_df = train_df.copy()
    prior_df = prior_df.copy()
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])

    splits = rolling_forward_splits(train_df, n_splits=n_splits, min_train_size=E0_MIN_TRAIN_SIZE)
    n_rows = len(train_df)
    oof_preds = {t: np.zeros(n_rows, dtype=float) for t in TARGET_COLS}
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

        X_min_tr = fe.transform(train_df.iloc[train_pos])
        X_min_va = fe.transform(train_df.iloc[val_pos])

        # As-of target encodings from train slice only (no leakage)
        ref = train_df.iloc[train_pos]
        encoding_dfs: dict[str, pd.DataFrame] = {}
        for group_col in E5_ENCODING_GROUPS:
            if group_col not in ref.columns:
                continue
            encoding_dfs[group_col] = _build_asof_target_encodings(
                ref, group_col, f"te_{group_col}", min_support=E5_MIN_SUPPORT
            )

        enc_tr = _add_encoding_features(train_df.iloc[train_pos], encoding_dfs, E5_ENCODING_GROUPS)
        enc_va = _add_encoding_features(train_df.iloc[val_pos], encoding_dfs, E5_ENCODING_GROUPS)
        # Fill NaN (unseen groups) with 0.5 rate and high SE
        enc_tr = enc_tr.fillna(0.5)
        enc_va = enc_va.fillna(0.5)

        X_tr = pd.concat([X_min_tr.reset_index(drop=True), enc_tr.reset_index(drop=True)], axis=1)
        X_va = pd.concat([X_min_va.reset_index(drop=True), enc_va.reset_index(drop=True)], axis=1)
        # Ensure numeric; LGBM can have issues with object columns
        X_tr = X_tr.astype(float)
        X_va = X_va.astype(float)

        fold_metrics: dict[str, float] = {}
        for target in TARGET_COLS:
            y_tr = train_df[target].iloc[train_pos].values
            y_va = train_df[target].iloc[val_pos].values

            train_set = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
            val_set = lgb.Dataset(X_va, label=y_va, free_raw_data=False)
            booster = lgb.train(
                params={**LGBM_PARAMS, "random_state": seed},
                train_set=train_set,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[train_set, val_set],
                valid_names=["train", "valid"],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            pred_va = booster.predict(X_va, num_iteration=booster.best_iteration)
            pred_va = np.clip(pred_va, *E0_CLIP)

            oof_preds[target][val_pos] = pred_va
            oof_mask[val_pos] = True

            ll = log_loss(y_va, pred_va)
            auc = roc_auc_score(y_va, pred_va) if len(np.unique(y_va)) > 1 else 0.5
            fold_metrics[f"{target}_ll"] = ll
            fold_metrics[f"{target}_auc"] = auc

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
            "  Fold %d: train=%d val=%d weighted=%.6f (minimal+encodings)",
            fold, len(train_pos), len(val_pos), fold_weighted,
        )

    y_07 = train_df[TARGET_COLS[0]].values[oof_mask]
    y_90 = train_df[TARGET_COLS[1]].values[oof_mask]
    y_120 = train_df[TARGET_COLS[2]].values[oof_mask]
    p_07 = oof_preds[TARGET_COLS[0]][oof_mask]
    p_90 = oof_preds[TARGET_COLS[1]][oof_mask]
    p_120 = oof_preds[TARGET_COLS[2]][oof_mask]

    oof_07_ll = log_loss(y_07, p_07)
    oof_07_auc = roc_auc_score(y_07, p_07) if len(np.unique(y_07)) > 1 else 0.5
    oof_90_ll = log_loss(y_90, p_90)
    oof_90_auc = roc_auc_score(y_90, p_90) if len(np.unique(y_90)) > 1 else 0.5
    oof_120_ll = log_loss(y_120, p_120)
    oof_120_auc = roc_auc_score(y_120, p_120) if len(np.unique(y_120)) > 1 else 0.5

    weighted_score = calculate_weighted_score(
        oof_07_auc, oof_07_ll, oof_90_auc, oof_90_ll, oof_120_auc, oof_120_ll,
    )
    fold_std = float(np.std(per_fold_weighted_scores)) if len(per_fold_weighted_scores) > 1 else 0.0

    return {
        "weighted_score": weighted_score,
        "per_fold_weighted_scores": per_fold_weighted_scores,
        "fold_std": fold_std,
    }


def get_e5_oof(
    train_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    n_splits: int = E0_N_SPLITS,
    seed: int = E0_SEED,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Run E5 encoding harness and return OOF dict (target -> array) and oof_mask. For E6 blending."""
    import lightgbm as lgb
    from plan1.config import EARLY_STOPPING_ROUNDS, LGBM_PARAMS, NUM_BOOST_ROUND

    np.random.seed(seed)
    train_df = train_df.copy()
    prior_df = prior_df.copy()
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])

    splits = rolling_forward_splits(train_df, n_splits=n_splits, min_train_size=E0_MIN_TRAIN_SIZE)
    n_rows = len(train_df)
    oof_preds = {t: np.zeros(n_rows, dtype=float) for t in TARGET_COLS}
    oof_mask = np.zeros(n_rows, dtype=bool)

    for split in splits:
        train_pos = split["train_pos"]
        val_pos = split["val_pos"]
        train_cutoff = split["train_cutoff"]

        prior_asof = prior_df[prior_df[DATE_COL] < train_cutoff].copy()
        fe = FeatureEngineer(prior_asof)
        fe.set_feature_groups(MINIMAL_FEATURE_GROUPS)
        fe.fit(train_df.iloc[train_pos])

        X_min_tr = fe.transform(train_df.iloc[train_pos])
        X_min_va = fe.transform(train_df.iloc[val_pos])

        ref = train_df.iloc[train_pos]
        encoding_dfs = {}
        for group_col in E5_ENCODING_GROUPS:
            if group_col not in ref.columns:
                continue
            encoding_dfs[group_col] = _build_asof_target_encodings(
                ref, group_col, f"te_{group_col}", min_support=E5_MIN_SUPPORT
            )

        enc_tr = _add_encoding_features(train_df.iloc[train_pos], encoding_dfs, E5_ENCODING_GROUPS)
        enc_va = _add_encoding_features(train_df.iloc[val_pos], encoding_dfs, E5_ENCODING_GROUPS)
        enc_tr = enc_tr.fillna(0.5)
        enc_va = enc_va.fillna(0.5)

        X_tr = pd.concat([X_min_tr.reset_index(drop=True), enc_tr.reset_index(drop=True)], axis=1)
        X_va = pd.concat([X_min_va.reset_index(drop=True), enc_va.reset_index(drop=True)], axis=1)
        X_tr = X_tr.astype(float)
        X_va = X_va.astype(float)

        for target in TARGET_COLS:
            y_tr = train_df[target].iloc[train_pos].values
            y_va = train_df[target].iloc[val_pos].values
            train_set = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
            val_set = lgb.Dataset(X_va, label=y_va, free_raw_data=False)
            booster = lgb.train(
                params={**LGBM_PARAMS, "random_state": seed},
                train_set=train_set,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[train_set, val_set],
                valid_names=["train", "valid"],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            pred_va = booster.predict(X_va, num_iteration=booster.best_iteration)
            oof_preds[target][val_pos] = np.clip(pred_va, *E0_CLIP)
            oof_mask[val_pos] = True

    return oof_preds, oof_mask


def main() -> None:
    os.environ["DIGICOW_MINIMAL_FEATURES"] = "1"

    logger.info("=" * 70)
    logger.info("E5 LEAKAGE-SAFE ENCODINGS EXPERIMENT")
    logger.info("  Pass: ΔWeighted >= +%.3f and ΔStd <= %.3f", E5_WEIGHTED_DELTA_MIN, E5_STD_DELTA_MAX)
    logger.info("=" * 70)

    loader = DataLoader()
    train_df, _test_df, prior_df, _ = loader.load_all()
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])

    logger.info("\n--- E0 baseline (minimal only) ---")
    baseline = run_single_e0_harness(train_df, prior_df, n_splits=E0_N_SPLITS, seed=E0_SEED)
    baseline_weighted = baseline["weighted_score"]
    baseline_std = baseline["fold_std"]
    logger.info("  Baseline weighted=%.6f  fold_std=%.6f", baseline_weighted, baseline_std)

    logger.info("\n--- E5 (minimal + as-of target encodings) ---")
    e5 = run_e5_encoding_harness(train_df, prior_df, n_splits=E0_N_SPLITS, seed=E0_SEED)
    e5_weighted = e5["weighted_score"]
    e5_std = e5["fold_std"]
    logger.info("  E5     weighted=%.6f  fold_std=%.6f", e5_weighted, e5_std)

    d_weighted = e5_weighted - baseline_weighted
    d_std = e5_std - baseline_std
    pass_weighted = d_weighted >= E5_WEIGHTED_DELTA_MIN
    pass_std = d_std <= E5_STD_DELTA_MAX  # std should decrease (more negative delta)
    primary_pass = pass_weighted and pass_std

    logger.info("\n" + "=" * 70)
    logger.info("E5 RESULT")
    logger.info("  Δweighted = %.6f  (need >= +%.3f)  %s", d_weighted, E5_WEIGHTED_DELTA_MIN, "OK" if pass_weighted else "FAIL")
    logger.info("  Δfold_std = %.6f  (need <= %.3f)   %s", d_std, E5_STD_DELTA_MAX, "OK" if pass_std else "FAIL")
    if primary_pass:
        logger.info("  PASS: Adopt leakage-safe encodings.")
    else:
        logger.warning("  FAIL: Drop encoding block.")
    logger.info("=" * 70)

    sys.exit(0 if primary_pass else 1)


if __name__ == "__main__":
    main()
