#!/usr/bin/env python3
"""
Generate E6 (tabular proxy), Plan 9 (multi-view), and stacked submissions.

- E6 submission: View A test preds only (E6-style tabular + isotonic).
- Plan 9 submission: Blend of View A + View B + View C test preds (View C from cache).
- Stacked submission: 0.5 * E6 + 0.5 * Plan 9 (then clip + hierarchy).

Requires View C n2v cache (run scripts/precompute_view_c_n2v.py once).
Writes to submissions/ then runs evaluate_submissions.py.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.constants import (
    DATE_COL,
    ID_COL,
    PROB_CLIP_MAX,
    PROB_CLIP_MIN,
    SUBMISSION_MAP,
    TARGET_COLS,
)
from shared.data_loader import DataLoader
from shared.submission import SubmissionGenerator

from e0_reproducibility import E0_MIN_TRAIN_SIZE, E0_N_SPLITS, E0_SEED, rolling_forward_splits
from plan9.config import PROB_CLIP
from plan9.model import run_plan9_oof
from plan9.view_a import get_view_a_oof, get_view_a_test
from plan9.view_b import get_view_b_test
from plan9.view_c import _get_view_c_n2v_artifact_dir, _save_view_c_n2v_test, load_view_c_n2v_test
from plan9.view_c_n2v import run_view_c_node2v_oof_and_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

STACKED_E6_WEIGHT = 0.5  # stacked = STACKED_E6_WEIGHT * E6 + (1 - STACKED_E6_WEIGHT) * Plan9


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-train", action="store_true", help="Also write train (OOF) submissions to submissions/train/ for evaluate_submissions.py --train")
    args = parser.parse_args()

    os.environ["DIGICOW_MINIMAL_FEATURES"] = "1"

    logger.info("Loading data ...")
    loader = DataLoader()
    train_df, test_df, prior_df, sample_sub = loader.load_all()
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])

    splits = rolling_forward_splits(
        train_df, n_splits=E0_N_SPLITS, min_train_size=E0_MIN_TRAIN_SIZE
    )

    # Plan 9 OOF + blend weights (View C from cache)
    logger.info("Running Plan 9 OOF to get blend weights (View C from cache) ...")
    result = run_plan9_oof(
        train_df, prior_df, splits=splits, seed=E0_SEED,
        baseline_worst_fold=None, return_weights=True
    )
    p07_oof, p90_oof, p120_oof, oof_mask, w_opt = result
    logger.info("Plan 9 blend weights: A=%.3f B=%.3f C=%.3f", w_opt[0], w_opt[1], w_opt[2])

    # View A test (E6 proxy)
    logger.info("View A test preds (E6 proxy) ...")
    e6_p07, e6_p90, e6_p120 = get_view_a_test(train_df, prior_df, test_df, splits, seed=E0_SEED)
    e6_p07 = np.clip(e6_p07, *PROB_CLIP)
    e6_p90 = np.clip(e6_p90, *PROB_CLIP)
    e6_p120 = np.clip(e6_p120, *PROB_CLIP)
    from shared.calibration import Calibrator
    e6_p07, e6_p90, e6_p120 = Calibrator.enforce_hierarchy(e6_p07, e6_p90, e6_p120)
    e6_p07 = np.clip(e6_p07, *PROB_CLIP)
    e6_p90 = np.clip(e6_p90, *PROB_CLIP)
    e6_p120 = np.clip(e6_p120, *PROB_CLIP)

    # View B test
    logger.info("View B test preds ...")
    b_p07, b_p90, b_p120 = get_view_b_test(train_df, test_df, splits, seed=E0_SEED)

    # View C test (from cache or compute once and save)
    c_test = load_view_c_n2v_test()
    if c_test is None:
        logger.info("View C n2v test cache missing; running Node2Vec+test once (slow) ...")
        _, _, _, _, c_p07, c_p90, c_p120 = run_view_c_node2v_oof_and_test(
            train_df, splits, test_df, seed=E0_SEED
        )
        artifact_dir = _get_view_c_n2v_artifact_dir()
        _save_view_c_n2v_test(artifact_dir, c_p07, c_p90, c_p120, len(test_df))
    else:
        c_p07, c_p90, c_p120 = c_test

    # Plan 9 test blend
    p9_p07 = w_opt[0] * e6_p07 + w_opt[1] * b_p07 + w_opt[2] * c_p07
    p9_p90 = w_opt[0] * e6_p90 + w_opt[1] * b_p90 + w_opt[2] * c_p90
    p9_p120 = w_opt[0] * e6_p120 + w_opt[1] * b_p120 + w_opt[2] * c_p120
    p9_p07 = np.clip(p9_p07, *PROB_CLIP)
    p9_p90 = np.clip(p9_p90, *PROB_CLIP)
    p9_p120 = np.clip(p9_p120, *PROB_CLIP)
    p9_p07, p9_p90, p9_p120 = Calibrator.enforce_hierarchy(p9_p07, p9_p90, p9_p120)
    p9_p07 = np.clip(p9_p07, *PROB_CLIP)
    p9_p90 = np.clip(p9_p90, *PROB_CLIP)
    p9_p120 = np.clip(p9_p120, *PROB_CLIP)

    # Stacked: alpha * E6 + (1-alpha) * Plan9
    alpha = STACKED_E6_WEIGHT
    stack_p07 = alpha * e6_p07 + (1 - alpha) * p9_p07
    stack_p90 = alpha * e6_p90 + (1 - alpha) * p9_p90
    stack_p120 = alpha * e6_p120 + (1 - alpha) * p9_p120
    stack_p07 = np.clip(stack_p07, *PROB_CLIP)
    stack_p90 = np.clip(stack_p90, *PROB_CLIP)
    stack_p120 = np.clip(stack_p120, *PROB_CLIP)
    stack_p07, stack_p90, stack_p120 = Calibrator.enforce_hierarchy(stack_p07, stack_p90, stack_p120)
    stack_p07 = np.clip(stack_p07, *PROB_CLIP)
    stack_p90 = np.clip(stack_p90, *PROB_CLIP)
    stack_p120 = np.clip(stack_p120, *PROB_CLIP)

    gen = SubmissionGenerator(sample_sub)

    preds_e6 = {
        TARGET_COLS[0]: e6_p07,
        TARGET_COLS[1]: e6_p90,
        TARGET_COLS[2]: e6_p120,
    }
    preds_p9 = {
        TARGET_COLS[0]: p9_p07,
        TARGET_COLS[1]: p9_p90,
        TARGET_COLS[2]: p9_p120,
    }
    preds_stack = {
        TARGET_COLS[0]: stack_p07,
        TARGET_COLS[1]: stack_p90,
        TARGET_COLS[2]: stack_p120,
    }

    root = Path(__file__).resolve().parent.parent
    gen.generate(test_ids=test_df[ID_COL], predictions=preds_e6, filename="e6_submission.csv")
    gen.generate(test_ids=test_df[ID_COL], predictions=preds_p9, filename="plan9_submission.csv")
    gen.generate(test_ids=test_df[ID_COL], predictions=preds_stack, filename="e6_plan9_stacked_submission.csv")

    if args.write_train:
        # Write train (OOF) submissions so evaluate_submissions.py --train can report real weighted scores
        train_sub_dir = root / "submissions" / "train"
        train_sub_dir.mkdir(parents=True, exist_ok=True)
        n_oof = int(oof_mask.sum())
        train_ids = train_df.loc[oof_mask, ID_COL]
        a_07, a_90, a_120, _ = get_view_a_oof(train_df, prior_df, splits, seed=E0_SEED)

        def _write_train_sub(filename: str, p07: np.ndarray, p90: np.ndarray, p120: np.ndarray) -> None:
            p07 = np.clip(p07, PROB_CLIP_MIN, PROB_CLIP_MAX)
            p90 = np.clip(p90, PROB_CLIP_MIN, PROB_CLIP_MAX)
            p120 = np.clip(p120, PROB_CLIP_MIN, PROB_CLIP_MAX)
            sub = pd.DataFrame({ID_COL: train_ids.values})
            for i, tc in enumerate(TARGET_COLS):
                auc_col, ll_col = SUBMISSION_MAP[tc]
                p = (p07, p90, p120)[i]
                sub[auc_col] = p
                sub[ll_col] = p
            sub.to_csv(train_sub_dir / filename, index=False)
            logger.info("Train submission written to %s  (%d rows)", train_sub_dir / filename, len(sub))

        _write_train_sub("e6_submission.csv", a_07[oof_mask], a_90[oof_mask], a_120[oof_mask])
        _write_train_sub("plan9_submission.csv", p07_oof[oof_mask], p90_oof[oof_mask], p120_oof[oof_mask])
        stack_p07 = np.clip(alpha * a_07[oof_mask] + (1 - alpha) * p07_oof[oof_mask], *PROB_CLIP)
        stack_p90 = np.clip(alpha * a_90[oof_mask] + (1 - alpha) * p90_oof[oof_mask], *PROB_CLIP)
        stack_p120 = np.clip(alpha * a_120[oof_mask] + (1 - alpha) * p120_oof[oof_mask], *PROB_CLIP)
        _write_train_sub("e6_plan9_stacked_submission.csv", stack_p07, stack_p90, stack_p120)
        logger.info("Train (OOF) submissions written to %s", train_sub_dir)

    logger.info("Submissions written. Running evaluate_submissions.py ...")
    code = subprocess.run(
        [sys.executable, str(root / "evaluate_submissions.py"), "--submissions-dir", str(root / "submissions")],
        cwd=str(root),
    ).returncode
    if code != 0:
        logger.warning("evaluate_submissions.py exited with code %s", code)
    else:
        logger.info("evaluate_submissions.py finished.")


if __name__ == "__main__":
    main()
