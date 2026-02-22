"""
Plan 8 — Cohort-Aware Mixture + Multi-Calibration.

Two experts (warm/cold), gating on has_prior_history-like signal,
cohort-wise calibration (4 cohorts: has_prior x has_topic).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from shared.constants import DATE_COL, FARMER_COL, TARGET_COLS, TOPICS_COL
from shared.feature_engineering import FeatureEngineer
from shared.calibration import Calibrator, CalibrationMethod

from plan8.config import (
    CALIBRATOR_OPTIONS,
    COLD_FEATURE_GROUPS,
    EARLY_STOPPING_ROUNDS,
    NUM_BOOST_ROUND,
    LGBM_PARAMS,
    PROB_CLIP,
    SEED,
    WARM_FEATURE_GROUPS,
)
from plan8.features import build_gating_features, cohort_id, compute_has_prior_history

logger = logging.getLogger(__name__)


def _train_experts(
    X_tr: np.ndarray,
    y_tr_list: list[np.ndarray],
    X_va: np.ndarray,
    y_va_list: list[np.ndarray],
    seed: int,
) -> list[Any]:
    """Train one LGBM per target; return list of boosters."""
    import lightgbm as lgb

    boosters = []
    for y_tr, y_va in zip(y_tr_list, y_va_list):
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
        boosters.append(booster)
    return boosters


def _predict_experts(boosters: list[Any], X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p07 = np.clip(boosters[0].predict(X), *PROB_CLIP)
    p90 = np.clip(boosters[1].predict(X), *PROB_CLIP)
    p120 = np.clip(boosters[2].predict(X), *PROB_CLIP)
    return p07, p90, p120


def run_one_fold(
    train_pos: np.ndarray,
    val_pos: np.ndarray,
    train_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    train_cutoff: Any,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run warm/cold experts + gating for one fold. Returns val_pos, p07, p90, p120, cohort (for val rows)."""
    import lightgbm as lgb

    prior_asof = prior_df[prior_df[DATE_COL] < train_cutoff].copy()
    train_slice = train_df.iloc[train_pos]
    val_slice = train_df.iloc[val_pos]

    # Has prior (for gating label and cohort)
    has_prior_train = compute_has_prior_history(train_slice, prior_asof)
    has_prior_val = compute_has_prior_history(val_slice, prior_asof)
    cohort_val = cohort_id(
        has_prior_val,
        val_slice[TOPICS_COL].apply(lambda x: 1.0 if isinstance(x, list) and len(x) > 0 else 0.0).values,
    )

    # Warm expert features
    fe_warm = FeatureEngineer(prior_asof)
    fe_warm.set_feature_groups(WARM_FEATURE_GROUPS)
    fe_warm.fit(train_slice)
    X_warm_tr = fe_warm.transform(train_slice)
    X_warm_va = fe_warm.transform(val_slice)
    if hasattr(X_warm_tr, "values"):
        X_warm_tr = X_warm_tr.values
        X_warm_va = X_warm_va.values
    X_warm_tr = np.asarray(X_warm_tr, dtype=np.float64)
    X_warm_va = np.asarray(X_warm_va, dtype=np.float64)

    # Cold expert features
    fe_cold = FeatureEngineer(prior_asof)
    fe_cold.set_feature_groups(COLD_FEATURE_GROUPS)
    fe_cold.fit(train_slice)
    X_cold_tr = fe_cold.transform(train_slice)
    X_cold_va = fe_cold.transform(val_slice)
    if hasattr(X_cold_tr, "values"):
        X_cold_tr = X_cold_tr.values
        X_cold_va = X_cold_va.values
    X_cold_tr = np.asarray(X_cold_tr, dtype=np.float64)
    X_cold_va = np.asarray(X_cold_va, dtype=np.float64)

    # Gating features
    X_gate_tr, encoders = build_gating_features(train_slice, prior_asof, fit_encoders=None)
    X_gate_va, _ = build_gating_features(val_slice, prior_asof, fit_encoders=encoders)

    y_tr_list = [train_slice[TARGET_COLS[i]].values for i in range(3)]
    y_va_list = [val_slice[TARGET_COLS[i]].values for i in range(3)]

    # Train experts
    warm_boosters = _train_experts(X_warm_tr, y_tr_list, X_warm_va, y_va_list, seed)
    cold_boosters = _train_experts(X_cold_tr, y_tr_list, X_cold_va, y_va_list, seed)

    p_warm_07, p_warm_90, p_warm_120 = _predict_experts(warm_boosters, X_warm_va)
    p_cold_07, p_cold_90, p_cold_120 = _predict_experts(cold_boosters, X_cold_va)

    # Gating: predict P(warm)
    train_set_g = lgb.Dataset(X_gate_tr, label=has_prior_train, free_raw_data=False)
    val_set_g = lgb.Dataset(X_gate_va, free_raw_data=False)
    gating = lgb.train(
        params={**LGBM_PARAMS, "random_state": seed},
        train_set=train_set_g,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_set_g, val_set_g],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    w_val = np.clip(gating.predict(X_gate_va), 0.0, 1.0)

    # Gated blend
    p07 = w_val * p_warm_07 + (1 - w_val) * p_cold_07
    p90 = w_val * p_warm_90 + (1 - w_val) * p_cold_90
    p120 = w_val * p_warm_120 + (1 - w_val) * p_cold_120
    p07 = np.clip(p07, *PROB_CLIP)
    p90 = np.clip(p90, *PROB_CLIP)
    p120 = np.clip(p120, *PROB_CLIP)

    return val_pos, p07, p90, p120, cohort_val


def run_plan8_oof(
    train_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    splits: list[dict[str, Any]],
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run Plan 8 over all folds; apply cohort-wise calibration; return OOF (p07, p90, p120) and oof_mask."""
    from sklearn.metrics import log_loss

    train_df = train_df.copy()
    prior_df = prior_df.copy()
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])

    n_rows = len(train_df)
    oof_p07 = np.zeros(n_rows)
    oof_p90 = np.zeros(n_rows)
    oof_p120 = np.zeros(n_rows)
    oof_cohort = np.full(n_rows, -1, dtype=int)
    oof_mask = np.zeros(n_rows, dtype=bool)

    for split in splits:
        val_pos, p07, p90, p120, cohort_val = run_one_fold(
            split["train_pos"],
            split["val_pos"],
            train_df,
            prior_df,
            split["train_cutoff"],
            seed,
        )
        oof_p07[val_pos] = p07
        oof_p90[val_pos] = p90
        oof_p120[val_pos] = p120
        oof_cohort[val_pos] = cohort_val
        oof_mask[val_pos] = True

    # Cohort-wise calibration: 4 cohorts x 3 targets -> choose calibrator by OOF LL
    method_map = {"none": CalibrationMethod.NONE, "platt": CalibrationMethod.PLATT, "isotonic": CalibrationMethod.ISOTONIC}
    calibrators: dict[int, tuple[Calibrator, Calibrator, Calibrator]] = {}

    for cohort in range(4):
        mask_c = oof_mask & (oof_cohort == cohort)
        if mask_c.sum() < 10:
            calibrators[cohort] = (Calibrator(CalibrationMethod.NONE), Calibrator(CalibrationMethod.NONE), Calibrator(CalibrationMethod.NONE))
            continue
        y_07 = train_df[TARGET_COLS[0]].values[mask_c]
        y_90 = train_df[TARGET_COLS[1]].values[mask_c]
        y_120 = train_df[TARGET_COLS[2]].values[mask_c]
        p_07 = oof_p07[mask_c]
        p_90 = oof_p90[mask_c]
        p_120 = oof_p120[mask_c]

        best_ll = 1e9
        best_cals = (Calibrator(CalibrationMethod.NONE), Calibrator(CalibrationMethod.NONE), Calibrator(CalibrationMethod.NONE))
        for opt in CALIBRATOR_OPTIONS:
            method = method_map[opt]
            c07 = Calibrator(method)
            c90 = Calibrator(method)
            c120 = Calibrator(method)
            c07.fit(y_07, p_07)
            c90.fit(y_90, p_90)
            c120.fit(y_120, p_120)
            pp07 = c07.transform(p_07)
            pp90 = c90.transform(p_90)
            pp120 = c120.transform(p_120)
            pp07, pp90, pp120 = Calibrator.enforce_hierarchy(pp07, pp90, pp120)
            ll = log_loss(y_07, pp07) + log_loss(y_90, pp90) + log_loss(y_120, pp120)
            if ll < best_ll:
                best_ll = ll
                best_cals = (c07, c90, c120)
        calibrators[cohort] = best_cals

    # Apply cohort calibrators to full OOF
    out_07 = oof_p07.copy()
    out_90 = oof_p90.copy()
    out_120 = oof_p120.copy()
    for cohort in range(4):
        mask_c = oof_mask & (oof_cohort == cohort)
        if mask_c.sum() == 0:
            continue
        c07, c90, c120 = calibrators.get(cohort, (Calibrator(CalibrationMethod.NONE), Calibrator(CalibrationMethod.NONE), Calibrator(CalibrationMethod.NONE)))
        out_07[mask_c] = c07.transform(oof_p07[mask_c])
        out_90[mask_c] = c90.transform(oof_p90[mask_c])
        out_120[mask_c] = c120.transform(oof_p120[mask_c])

    out_07, out_90, out_120 = Calibrator.enforce_hierarchy(out_07, out_90, out_120)
    out_07 = np.clip(out_07, *PROB_CLIP)
    out_90 = np.clip(out_90, *PROB_CLIP)
    out_120 = np.clip(out_120, *PROB_CLIP)

    return out_07, out_90, out_120, oof_mask
