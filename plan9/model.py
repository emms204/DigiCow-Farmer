"""
Plan 9 — Multi-View Ensemble: View A (tabular) + View B (text) + View C (graph), then constrained blend.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from shared.constants import DATE_COL, TARGET_COLS
from shared.evaluation import calculate_weighted_score
from sklearn.metrics import log_loss, roc_auc_score

from e0_reproducibility import E0_MIN_TRAIN_SIZE, E0_N_SPLITS, E0_SEED, rolling_forward_splits
from plan9.config import BLEND_PENALTY_FOLD_STD, PROB_CLIP, SEED
from plan9.view_a import get_view_a_oof
from plan9.view_b import get_view_b_oof
from plan9.view_c import get_view_c_oof

logger = logging.getLogger(__name__)


def _weighted_score_from_predictions(
    y_07: np.ndarray, y_90: np.ndarray, y_120: np.ndarray,
    p_07: np.ndarray, p_90: np.ndarray, p_120: np.ndarray,
) -> float:
    ll_07 = log_loss(y_07, p_07)
    ll_90 = log_loss(y_90, p_90)
    ll_120 = log_loss(y_120, p_120)
    auc_07 = roc_auc_score(y_07, p_07) if len(np.unique(y_07)) > 1 else 0.5
    auc_90 = roc_auc_score(y_90, p_90) if len(np.unique(y_90)) > 1 else 0.5
    auc_120 = roc_auc_score(y_120, p_120) if len(np.unique(y_120)) > 1 else 0.5
    return calculate_weighted_score(auc_07, ll_07, auc_90, ll_90, auc_120, ll_120)


def _worst_fold_score(
    p_07: np.ndarray, p_90: np.ndarray, p_120: np.ndarray,
    train_df: pd.DataFrame,
    splits: list[dict[str, Any]],
) -> float:
    scores = []
    for split in splits:
        val_pos = split["val_pos"]
        y_07 = train_df[TARGET_COLS[0]].iloc[val_pos].values
        y_90 = train_df[TARGET_COLS[1]].iloc[val_pos].values
        y_120 = train_df[TARGET_COLS[2]].iloc[val_pos].values
        scores.append(_weighted_score_from_predictions(
            y_07, y_90, y_120,
            p_07[val_pos], p_90[val_pos], p_120[val_pos],
        ))
    return float(min(scores))


def _fold_std(
    p_07: np.ndarray, p_90: np.ndarray, p_120: np.ndarray,
    train_df: pd.DataFrame,
    splits: list[dict[str, Any]],
) -> float:
    scores = []
    for split in splits:
        val_pos = split["val_pos"]
        y_07 = train_df[TARGET_COLS[0]].iloc[val_pos].values
        y_90 = train_df[TARGET_COLS[1]].iloc[val_pos].values
        y_120 = train_df[TARGET_COLS[2]].iloc[val_pos].values
        scores.append(_weighted_score_from_predictions(
            y_07, y_90, y_120,
            p_07[val_pos], p_90[val_pos], p_120[val_pos],
        ))
    return float(np.std(scores)) if len(scores) > 1 else 0.0


def _blend(w: np.ndarray, stack_07: list[np.ndarray], stack_90: list[np.ndarray], stack_120: list[np.ndarray]):
    w = np.clip(w, 0.0, 1.0)
    w = w / w.sum()
    n = len(stack_07[0])
    p07 = sum(w[k] * stack_07[k] for k in range(len(w)))
    p90 = sum(w[k] * stack_90[k] for k in range(len(w)))
    p120 = sum(w[k] * stack_120[k] for k in range(len(w)))
    return np.array(p07), np.array(p90), np.array(p120)


def run_plan9_oof(
    train_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    splits: list[dict[str, Any]] | None = None,
    seed: int = SEED,
    baseline_worst_fold: float | None = None,
    return_weights: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run Plan 9: View A + B + C OOF, then constrained blend. Returns (p07, p90, p120, oof_mask[, w_opt])."""
    train_df = train_df.copy()
    prior_df = prior_df.copy()
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])

    if splits is None:
        splits = rolling_forward_splits(train_df, n_splits=E0_N_SPLITS, min_train_size=E0_MIN_TRAIN_SIZE)

    # View A: E6 isotonic (needs minimal features)
    os.environ["DIGICOW_MINIMAL_FEATURES"] = "1"
    logger.info("Plan 9: View A (tabular isotonic) ...")
    a_07, a_90, a_120, oof_mask = get_view_a_oof(train_df, prior_df, splits, seed=seed)

    # View B: topic text
    logger.info("Plan 9: View B (topic text) ...")
    b_07, b_90, b_120, mask_b = get_view_b_oof(train_df, splits, seed=seed)
    assert np.array_equal(mask_b, oof_mask), "View B oof_mask mismatch"

    # View C: graph
    logger.info("Plan 9: View C (graph) ...")
    c_07, c_90, c_120, mask_c = get_view_c_oof(train_df, splits, seed=seed)
    assert np.array_equal(mask_c, oof_mask), "View C oof_mask mismatch"

    stack_07 = [a_07, b_07, c_07]
    stack_90 = [a_90, b_90, c_90]
    stack_120 = [a_120, b_120, c_120]
    y_07 = train_df[TARGET_COLS[0]].values[oof_mask]
    y_90 = train_df[TARGET_COLS[1]].values[oof_mask]
    y_120 = train_df[TARGET_COLS[2]].values[oof_mask]

    # Constrained blender: max weighted_score - penalty * fold_std, s.t. sum(w)=1, w>=0, worst_fold >= baseline
    def objective(w: np.ndarray) -> float:
        p07, p90, p120 = _blend(w, stack_07, stack_90, stack_120)
        ws = _weighted_score_from_predictions(y_07, y_90, y_120, p07[oof_mask], p90[oof_mask], p120[oof_mask])
        fold_std = _fold_std(p07, p90, p120, train_df, splits)
        return -(ws - BLEND_PENALTY_FOLD_STD * fold_std)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if baseline_worst_fold is not None:
        def constraint_worst(w: np.ndarray) -> float:
            p07, p90, p120 = _blend(w, stack_07, stack_90, stack_120)
            return _worst_fold_score(p07, p90, p120, train_df, splits) - baseline_worst_fold
        constraints.append({"type": "ineq", "fun": constraint_worst})

    w0 = np.ones(3) / 3
    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 3,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-9},
    )
    w_opt = np.clip(res.x, 0.0, 1.0)
    w_opt = w_opt / w_opt.sum()
    logger.info("Plan 9 blend weights: A=%.3f B=%.3f C=%.3f", w_opt[0], w_opt[1], w_opt[2])

    p07, p90, p120 = _blend(w_opt, stack_07, stack_90, stack_120)
    p07 = np.clip(p07, *PROB_CLIP)
    p90 = np.clip(p90, *PROB_CLIP)
    p120 = np.clip(p120, *PROB_CLIP)
    # Enforce hierarchy
    from shared.calibration import Calibrator
    p07, p90, p120 = Calibrator.enforce_hierarchy(p07, p90, p120)
    p07 = np.clip(p07, *PROB_CLIP)
    p90 = np.clip(p90, *PROB_CLIP)
    p120 = np.clip(p120, *PROB_CLIP)

    if return_weights:
        return p07, p90, p120, oof_mask, w_opt
    return p07, p90, p120, oof_mask
