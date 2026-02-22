"""
View A: Current best tabular (E6 isotonic-style).

Reuses E0-style LGBM + minimal features, then isotonic calibration per target.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from shared.constants import DATE_COL, TARGET_COLS

from e0_reproducibility import E0_MIN_TRAIN_SIZE, E0_N_SPLITS, E0_SEED
from e3_calibration_blend import apply_calibrators, get_raw_oof


def get_view_a_oof(
    train_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    splits: list[dict[str, Any]],
    seed: int = E0_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (p07, p90, p120, oof_mask) for View A = raw LGBM + isotonic calibration."""
    train_df = train_df.copy()
    prior_df = prior_df.copy()
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])

    oof_preds, oof_mask = get_raw_oof(train_df, prior_df, n_splits=len(splits), seed=seed)
    _, calibrated = apply_calibrators(oof_preds, oof_mask, train_df)

    p07 = calibrated[TARGET_COLS[0]]["isotonic"]
    p90 = calibrated[TARGET_COLS[1]]["isotonic"]
    p120 = calibrated[TARGET_COLS[2]]["isotonic"]
    return p07, p90, p120, oof_mask


def get_view_a_test(
    train_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    test_df: pd.DataFrame,
    splits: list[dict[str, Any]],
    seed: int = E0_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (test_p07, test_p90, test_p120) for View A: raw LGBM test preds (fold-avg) + isotonic calibrator fit on OOF."""
    from shared.calibration import Calibrator, CalibrationMethod

    train_df = train_df.copy()
    prior_df = prior_df.copy()
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])

    from e3_calibration_blend import get_raw_oof_and_test
    oof_preds, oof_mask, test_preds = get_raw_oof_and_test(
        train_df, prior_df, test_df, n_splits=len(splits), seed=seed
    )
    _, calibrated = apply_calibrators(oof_preds, oof_mask, train_df)
    # Fit isotonic on OOF (raw vs true) and apply to averaged test preds
    test_p07 = np.clip(test_preds[TARGET_COLS[0]].astype(float), 1e-6, 1 - 1e-6)
    test_p90 = np.clip(test_preds[TARGET_COLS[1]].astype(float), 1e-6, 1 - 1e-6)
    test_p120 = np.clip(test_preds[TARGET_COLS[2]].astype(float), 1e-6, 1 - 1e-6)
    cal_07 = Calibrator(CalibrationMethod.ISOTONIC)
    cal_07.fit(train_df[TARGET_COLS[0]].values[oof_mask], oof_preds[TARGET_COLS[0]][oof_mask])
    test_p07 = cal_07.transform(test_p07)
    cal_90 = Calibrator(CalibrationMethod.ISOTONIC)
    cal_90.fit(train_df[TARGET_COLS[1]].values[oof_mask], oof_preds[TARGET_COLS[1]][oof_mask])
    test_p90 = cal_90.transform(test_p90)
    cal_120 = Calibrator(CalibrationMethod.ISOTONIC)
    cal_120.fit(train_df[TARGET_COLS[2]].values[oof_mask], oof_preds[TARGET_COLS[2]][oof_mask])
    test_p120 = cal_120.transform(test_p120)
    return np.clip(test_p07, 1e-6, 1 - 1e-6), np.clip(test_p90, 1e-6, 1 - 1e-6), np.clip(test_p120, 1e-6, 1 - 1e-6)
