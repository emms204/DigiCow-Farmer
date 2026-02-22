#!/usr/bin/env python3
"""
E3 Calibration blend experiment: OOF ensemble of none/platt/isotonic/beta;
optimize blend on forward OOF; compare to best single calibrator.

Hypothesis E3: Calibrator choice is fold-dependent; blending calibrators is better.
Experiment: OOF calibration ensemble; optimize blend on forward OOF.
Primary pass metric: 7d LogLoss.
Guardrails: AUC drop <= 0.005.
Pass threshold: Δ7dLL <= -0.002 and non-negative weighted delta.
If fail: Use best single calibrator.

Usage:
    python e3_calibration_blend.py
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

from shared.calibration import Calibrator, CalibrationMethod
from shared.constants import DATE_COL, PROB_CLIP_MAX, PROB_CLIP_MIN, TARGET_COLS
from shared.data_loader import DataLoader
from shared.evaluation import calculate_weighted_score
from shared.feature_engineering import MINIMAL_FEATURE_GROUPS, FeatureEngineer
from sklearn.metrics import log_loss, roc_auc_score
from scipy.optimize import minimize
from scipy.stats import beta as scipy_beta

from e0_reproducibility import (
    E0_CLIP,
    E0_MIN_TRAIN_SIZE,
    E0_N_SPLITS,
    E0_SEED,
    rolling_forward_splits,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# E3 pass thresholds
E3_D7D_LL_MAX = -0.002   # 7d LogLoss must improve by at least 0.002
E3_WEIGHTED_MIN = 0.0   # weighted delta must be non-negative
E3_AUC_DROP_TOL = 0.005 # guardrail: 7d AUC drop <= 0.005


def get_raw_oof(
    train_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    n_splits: int = E0_N_SPLITS,
    seed: int = E0_SEED,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Run E0-style LGBM, return raw OOF predictions (clipped) and mask. No calibration."""
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

        X_tr = fe.transform(train_df.iloc[train_pos])
        X_va = fe.transform(train_df.iloc[val_pos])

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


def get_raw_oof_and_test(
    train_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_splits: int = E0_N_SPLITS,
    seed: int = E0_SEED,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    """Run E0-style LGBM per fold; return raw OOF, mask, and test preds (averaged over folds)."""
    import lightgbm as lgb
    from plan1.config import EARLY_STOPPING_ROUNDS, LGBM_PARAMS, NUM_BOOST_ROUND

    np.random.seed(seed)
    train_df = train_df.copy()
    prior_df = prior_df.copy()
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])

    splits = rolling_forward_splits(train_df, n_splits=n_splits, min_train_size=E0_MIN_TRAIN_SIZE)
    n_rows = len(train_df)
    n_test = len(test_df)
    oof_preds = {t: np.zeros(n_rows, dtype=float) for t in TARGET_COLS}
    oof_mask = np.zeros(n_rows, dtype=bool)
    test_sum = {t: np.zeros(n_test, dtype=float) for t in TARGET_COLS}

    for split in splits:
        train_pos = split["train_pos"]
        val_pos = split["val_pos"]
        train_cutoff = split["train_cutoff"]

        prior_asof = prior_df[prior_df[DATE_COL] < train_cutoff].copy()
        fe = FeatureEngineer(prior_asof)
        fe.set_feature_groups(MINIMAL_FEATURE_GROUPS)
        fe.fit(train_df.iloc[train_pos])

        X_tr = fe.transform(train_df.iloc[train_pos])
        X_va = fe.transform(train_df.iloc[val_pos])
        X_te = fe.transform(test_df)

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
            pred_te = booster.predict(X_te, num_iteration=booster.best_iteration)
            test_sum[target] += np.clip(pred_te, *E0_CLIP)
        oof_mask[val_pos] = True

    n_folds = len(splits)
    test_preds = {t: test_sum[t] / n_folds for t in TARGET_COLS}
    return oof_preds, oof_mask, test_preds


def beta_calibrate_fit(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Fit Beta CDF calibration: calibrated = Beta(alpha, beta).cdf(p). Returns (alpha, beta)."""
    # Minimize log loss of (y_true, clip(beta.cdf(y_prob)))
    def neg_ll(params: np.ndarray) -> float:
        a, b = float(params[0]), float(params[1])
        if a <= 0 or b <= 0:
            return 1e10
        try:
            p_cal = np.clip(scipy_beta.cdf(y_prob, a, b), 1e-6, 1 - 1e-6)
            return log_loss(y_true, p_cal)
        except Exception:
            return 1e10

    res = minimize(
        neg_ll,
        x0=[2.0, 2.0],
        method="L-BFGS-B",
        bounds=[(0.1, 50), (0.1, 50)],
    )
    alpha, beta = float(res.x[0]), float(res.x[1])
    return alpha, beta


def beta_calibrate_transform(y_prob: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    out = scipy_beta.cdf(y_prob, alpha, beta)
    return np.clip(out, PROB_CLIP_MIN, PROB_CLIP_MAX)


def apply_calibrators(
    oof_preds: dict[str, np.ndarray],
    oof_mask: np.ndarray,
    train_df: pd.DataFrame,
) -> tuple[list[str], dict[str, dict[str, np.ndarray]]]:
    """Apply none/platt/isotonic/beta per target. Return calibrator names and {target: {cal_name: calibrated_oof}}."""
    names = ["none", "platt", "isotonic", "beta"]
    out: dict[str, dict[str, np.ndarray]] = {t: {} for t in TARGET_COLS}
    beta_params: dict[str, tuple[float, float]] = {}

    for target in TARGET_COLS:
        y = train_df[target].values[oof_mask]
        p_raw = oof_preds[target][oof_mask]

        # none: clip only (full length to match other calibrators)
        out[target]["none"] = np.clip(oof_preds[target].copy(), PROB_CLIP_MIN, PROB_CLIP_MAX)

        # platt
        cal_platt = Calibrator(CalibrationMethod.PLATT)
        cal_platt.fit(y, p_raw)
        out[target]["platt"] = cal_platt.transform(oof_preds[target].copy())

        # isotonic
        cal_iso = Calibrator(CalibrationMethod.ISOTONIC)
        cal_iso.fit(y, p_raw)
        out[target]["isotonic"] = cal_iso.transform(oof_preds[target].copy())

        # beta
        alpha, beta = beta_calibrate_fit(y, p_raw)
        beta_params[target] = (alpha, beta)
        out[target]["beta"] = beta_calibrate_transform(oof_preds[target].copy(), alpha, beta)

    return names, out


def metrics_from_oof(
    p07: np.ndarray, p90: np.ndarray, p120: np.ndarray,
    y07: np.ndarray, y90: np.ndarray, y120: np.ndarray,
) -> dict[str, float]:
    """Enforce hierarchy then compute 7d_ll, weighted_score, 7d_auc."""
    p07, p90, p120 = Calibrator.enforce_hierarchy(p07, p90, p120)
    ll_07 = log_loss(y07, p07)
    ll_90 = log_loss(y90, p90)
    ll_120 = log_loss(y120, p120)
    auc_07 = roc_auc_score(y07, p07) if len(np.unique(y07)) > 1 else 0.5
    auc_90 = roc_auc_score(y90, p90) if len(np.unique(y90)) > 1 else 0.5
    auc_120 = roc_auc_score(y120, p120) if len(np.unique(y120)) > 1 else 0.5
    ws = calculate_weighted_score(auc_07, ll_07, auc_90, ll_90, auc_120, ll_120)
    return {"7d_ll": ll_07, "weighted_score": ws, "7d_auc": auc_07}


def blend_weights_to_oof(
    w: np.ndarray,
    cal_name_list: list[str],
    calibrated: dict[str, dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """w: (4,) weights for [none, platt, isotonic, beta]. Returns blended p07, p90, p120 (full length)."""
    p07 = np.zeros_like(calibrated[TARGET_COLS[0]]["none"])
    p90 = np.zeros_like(calibrated[TARGET_COLS[1]]["none"])
    p120 = np.zeros_like(calibrated[TARGET_COLS[2]]["none"])
    for i, name in enumerate(cal_name_list):
        p07 += w[i] * calibrated[TARGET_COLS[0]][name]
        p90 += w[i] * calibrated[TARGET_COLS[1]][name]
        p120 += w[i] * calibrated[TARGET_COLS[2]][name]
    return p07, p90, p120


def main() -> None:
    os.environ["DIGICOW_MINIMAL_FEATURES"] = "1"

    logger.info("=" * 70)
    logger.info("E3 CALIBRATION BLEND EXPERIMENT")
    logger.info("  Pass: Δ7dLL <= %.3f and weighted delta >= %.1f", E3_D7D_LL_MAX, E3_WEIGHTED_MIN)
    logger.info("  Guardrail: 7d AUC drop <= %.3f", E3_AUC_DROP_TOL)
    logger.info("=" * 70)

    loader = DataLoader()
    train_df, _test_df, prior_df, _ = loader.load_all()
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])

    # 1. Raw OOF
    logger.info("\n--- Getting raw OOF (E0-style LGBM, no calibration) ---")
    oof_preds, oof_mask = get_raw_oof(train_df, prior_df, n_splits=E0_N_SPLITS, seed=E0_SEED)
    logger.info("  Raw OOF collected (mask sum=%d)", oof_mask.sum())

    # 2. Four calibrators per target
    logger.info("\n--- Fitting none/platt/isotonic/beta per target ---")
    cal_name_list, calibrated = apply_calibrators(oof_preds, oof_mask, train_df)

    y07 = train_df[TARGET_COLS[0]].values[oof_mask]
    y90 = train_df[TARGET_COLS[1]].values[oof_mask]
    y120 = train_df[TARGET_COLS[2]].values[oof_mask]

    # 3. Baseline = best single calibrator (by 7d_ll)
    single_metrics: dict[str, dict[str, float]] = {}
    for name in cal_name_list:
        p07 = calibrated[TARGET_COLS[0]][name][oof_mask]
        p90 = calibrated[TARGET_COLS[1]][name][oof_mask]
        p120 = calibrated[TARGET_COLS[2]][name][oof_mask]
        single_metrics[name] = metrics_from_oof(p07, p90, p120, y07, y90, y120)

    best_name = min(cal_name_list, key=lambda n: single_metrics[n]["7d_ll"])
    baseline_7d_ll = single_metrics[best_name]["7d_ll"]
    baseline_weighted = single_metrics[best_name]["weighted_score"]
    baseline_7d_auc = single_metrics[best_name]["7d_auc"]

    logger.info("  Best single calibrator (by 7d_ll): %s  7d_ll=%.6f  weighted=%.6f  7d_auc=%.6f",
                best_name, baseline_7d_ll, baseline_weighted, baseline_7d_auc)
    for n in cal_name_list:
        logger.info("    %s: 7d_ll=%.6f  weighted=%.6f", n, single_metrics[n]["7d_ll"], single_metrics[n]["weighted_score"])

    # 4. Optimize blend weights (minimize 7d_ll on OOF)
    def objective(w: np.ndarray) -> float:
        w = np.clip(w, 0, 1)
        w = w / w.sum()
        p07, p90, p120 = blend_weights_to_oof(w, cal_name_list, calibrated)
        p07_m = p07[oof_mask]
        p90_m = p90[oof_mask]
        p120_m = p120[oof_mask]
        p07_m, p90_m, p120_m = Calibrator.enforce_hierarchy(p07_m, p90_m, p120_m)
        return log_loss(y07, p07_m)

    w0 = np.ones(4) / 4
    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=[(0, 1)] * 4,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
    )
    w_opt = np.clip(res.x, 0, 1)
    w_opt = w_opt / w_opt.sum()
    logger.info("  Blend weights (none/platt/isotonic/beta): %s", [round(x, 4) for x in w_opt])

    # 5. Blend metrics
    p07_b, p90_b, p120_b = blend_weights_to_oof(w_opt, cal_name_list, calibrated)
    blend_metrics = metrics_from_oof(
        p07_b[oof_mask], p90_b[oof_mask], p120_b[oof_mask],
        y07, y90, y120,
    )
    blend_7d_ll = blend_metrics["7d_ll"]
    blend_weighted = blend_metrics["weighted_score"]
    blend_7d_auc = blend_metrics["7d_auc"]

    d_7d_ll = blend_7d_ll - baseline_7d_ll
    d_weighted = blend_weighted - baseline_weighted
    d_7d_auc = blend_7d_auc - baseline_7d_auc

    pass_7d_ll = d_7d_ll <= E3_D7D_LL_MAX
    pass_weighted = d_weighted >= E3_WEIGHTED_MIN
    guardrail_auc = d_7d_auc >= -E3_AUC_DROP_TOL
    primary_pass = pass_7d_ll and pass_weighted and guardrail_auc

    logger.info("\n--- E3 blend (optimized on OOF) ---")
    logger.info("  Blend 7d_ll=%.6f  weighted=%.6f  7d_auc=%.6f", blend_7d_ll, blend_weighted, blend_7d_auc)
    logger.info("\n" + "=" * 70)
    logger.info("E3 RESULT")
    logger.info("  Δ7dLL     = %.6f  (need <= %.3f)  %s", d_7d_ll, E3_D7D_LL_MAX, "OK" if pass_7d_ll else "FAIL")
    logger.info("  ΔWeighted = %.6f  (need >= %.1f)  %s", d_weighted, E3_WEIGHTED_MIN, "OK" if pass_weighted else "FAIL")
    logger.info("  7d AUC drop = %.6f  (guardrail <= %.3f)  %s", -d_7d_auc, E3_AUC_DROP_TOL, "OK" if guardrail_auc else "FAIL")
    if primary_pass:
        logger.info("  PASS: Adopt calibration blend.")
    else:
        logger.warning("  FAIL: Use best single calibrator (%s).", best_name)
    logger.info("=" * 70)

    sys.exit(0 if primary_pass else 1)


if __name__ == "__main__":
    main()
