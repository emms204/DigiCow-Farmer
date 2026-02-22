#!/usr/bin/env python3
"""
Precompute View C Node2Vec artifact (view_c_n2v_v1).

Run once to build and cache:
  - OOF predictions (train) → plan9/artifacts/view_c_n2v_v1/oof.npz
  - Test predictions       → plan9/artifacts/view_c_n2v_v1/test.npz

Then E9 and blend/calibration iterate in seconds by loading this cache.
Rerun this script only when the graph schema or data changes; bump
VIEW_C_N2V_ARTIFACT_VERSION in plan9/config.py for a new artifact version.

Usage:
  python scripts/precompute_view_c_n2v.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.constants import DATE_COL
from shared.data_loader import DataLoader

from e0_reproducibility import E0_MIN_TRAIN_SIZE, E0_N_SPLITS, E0_SEED, rolling_forward_splits
from plan9.config import VIEW_C_N2V_ARTIFACT_VERSION
from plan9.view_c import _get_view_c_n2v_artifact_dir, _save_view_c_n2v_oof, _save_view_c_n2v_test
from plan9.view_c_n2v import run_view_c_node2v_oof_and_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Precomputing View C Node2Vec artifact (version=%s) ...", VIEW_C_N2V_ARTIFACT_VERSION)
    loader = DataLoader()
    train_df, test_df, prior_df, _ = loader.load_all()
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    splits = rolling_forward_splits(
        train_df, n_splits=E0_N_SPLITS, min_train_size=E0_MIN_TRAIN_SIZE
    )

    oof_p07, oof_p90, oof_p120, oof_mask, test_p07, test_p90, test_p120 = run_view_c_node2v_oof_and_test(
        train_df, splits, test_df, seed=E0_SEED
    )

    artifact_dir = _get_view_c_n2v_artifact_dir()
    _save_view_c_n2v_oof(
        artifact_dir, oof_p07, oof_p90, oof_p120, oof_mask, len(train_df)
    )
    _save_view_c_n2v_test(
        artifact_dir, test_p07, test_p90, test_p120, len(test_df)
    )
    logger.info("Done. Artifact at %s", artifact_dir)


if __name__ == "__main__":
    main()
