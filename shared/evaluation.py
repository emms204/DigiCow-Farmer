"""
Local evaluation function matching the competition's scoring formula.

Based on the competition clarification:
    - 7-day target: 65% LogLoss + 15% AUC = 80% of total score
    - 90-day target: 5% LogLoss + 5% AUC = 10% of total score
    - 120-day target: 5% LogLoss + 5% AUC = 10% of total score

LogLoss values are normalized using SCALING = 0.56926.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from shared.constants import SUBMISSION_DIR

logger = logging.getLogger(__name__)

# Competition scoring constants
SCALING: float = 0.56926
WEIGHTS: dict[str, float] = {
    "Target_07_LogLoss": 0.65,
    "Target_07_AUC": 0.15,
    "Target_90_LogLoss": 0.05,
    "Target_90_AUC": 0.05,
    "Target_120_LogLoss": 0.05,
    "Target_120_AUC": 0.05,
}


def calculate_weighted_score(
    target_07_auc: float,
    target_07_logloss: float,
    target_90_auc: float,
    target_90_logloss: float,
    target_120_auc: float,
    target_120_logloss: float,
) -> float:
    """Calculate the competition's weighted score.

    Parameters
    ----------
    target_07_auc, target_90_auc, target_120_auc : float
        ROC-AUC scores for each target.
    target_07_logloss, target_90_logloss, target_120_logloss : float
        Log Loss scores for each target.

    Returns
    -------
    float
        Weighted competition score (higher is better).
    """
    # Normalize LogLoss values
    target_07_logloss_norm = 1 - (target_07_logloss / SCALING)
    target_90_logloss_norm = 1 - (target_90_logloss / SCALING)
    target_120_logloss_norm = 1 - (target_120_logloss / SCALING)

    # Weighted sum
    weighted_score = (
        target_07_auc * WEIGHTS["Target_07_AUC"]
        + target_07_logloss_norm * WEIGHTS["Target_07_LogLoss"]
        + target_90_auc * WEIGHTS["Target_90_AUC"]
        + target_90_logloss_norm * WEIGHTS["Target_90_LogLoss"]
        + target_120_auc * WEIGHTS["Target_120_AUC"]
        + target_120_logloss_norm * WEIGHTS["Target_120_LogLoss"]
    )

    return weighted_score


def evaluate_submission(
    submission_df: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """Evaluate a submission file against reference labels.

    Parameters
    ----------
    submission_df : pd.DataFrame
        Submission CSV with columns: ID, Target_07_AUC, Target_07_LogLoss, etc.
    reference_df : pd.DataFrame, optional
        Reference CSV with true labels.  If None, only returns predictions
        (cannot compute LogLoss/AUC).
    verbose : bool
        Print detailed breakdown.

    Returns
    -------
    dict
        Dictionary with keys:
            - 'score': overall weighted score
            - 'target_07_auc', 'target_07_logloss', etc.: individual metrics
            - 'target_07_logloss_norm', etc.: normalized LogLoss values
    """
    results: dict[str, float] = {}

    if reference_df is None:
        logger.warning("No reference provided — cannot compute LogLoss/AUC")
        return results

    # Merge on ID
    merged = submission_df.merge(reference_df, on="ID", how="inner")
    if len(merged) != len(submission_df):
        logger.warning(
            "ID mismatch: submission has %d rows, merged has %d",
            len(submission_df),
            len(merged),
        )

    # Extract true labels
    y_true_07 = merged["adopted_within_07_days"].values
    y_true_90 = merged["adopted_within_90_days"].values
    y_true_120 = merged["adopted_within_120_days"].values

    # Extract predictions
    y_pred_07_ll = merged["Target_07_LogLoss"].values
    y_pred_90_ll = merged["Target_90_LogLoss"].values
    y_pred_120_ll = merged["Target_120_LogLoss"].values

    y_pred_07_auc = merged["Target_07_AUC"].values
    y_pred_90_auc = merged["Target_90_AUC"].values
    y_pred_120_auc = merged["Target_120_AUC"].values

    # Compute LogLoss
    ll_07 = log_loss(y_true_07, y_pred_07_ll)
    ll_90 = log_loss(y_true_90, y_pred_90_ll)
    ll_120 = log_loss(y_true_120, y_pred_120_ll)

    results["target_07_logloss"] = ll_07
    results["target_90_logloss"] = ll_90
    results["target_120_logloss"] = ll_120

    # Normalize LogLoss
    results["target_07_logloss_norm"] = 1 - (ll_07 / SCALING)
    results["target_90_logloss_norm"] = 1 - (ll_90 / SCALING)
    results["target_120_logloss_norm"] = 1 - (ll_120 / SCALING)

    # Compute AUC
    if len(np.unique(y_true_07)) < 2:
        auc_07 = 0.5
    else:
        auc_07 = roc_auc_score(y_true_07, y_pred_07_auc)

    if len(np.unique(y_true_90)) < 2:
        auc_90 = 0.5
    else:
        auc_90 = roc_auc_score(y_true_90, y_pred_90_auc)

    if len(np.unique(y_true_120)) < 2:
        auc_120 = 0.5
    else:
        auc_120 = roc_auc_score(y_true_120, y_pred_120_auc)

    results["target_07_auc"] = auc_07
    results["target_90_auc"] = auc_90
    results["target_120_auc"] = auc_120

    # Compute weighted score
    score = calculate_weighted_score(
        auc_07, ll_07, auc_90, ll_90, auc_120, ll_120
    )
    results["score"] = score

    if verbose:
        logger.info("=" * 80)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 80)
        logger.info("\n7-day target (80%% of score):")
        logger.info("  LogLoss: %.6f → normalized: %.6f (weight: 65%%)", ll_07, results["target_07_logloss_norm"])
        logger.info("  AUC:     %.6f (weight: 15%%)", auc_07)
        logger.info("  Contribution: %.6f", results["target_07_logloss_norm"] * 0.65 + auc_07 * 0.15)

        logger.info("\n90-day target (10%% of score):")
        logger.info("  LogLoss: %.6f → normalized: %.6f (weight: 5%%)", ll_90, results["target_90_logloss_norm"])
        logger.info("  AUC:     %.6f (weight: 5%%)", auc_90)
        logger.info("  Contribution: %.6f", results["target_90_logloss_norm"] * 0.05 + auc_90 * 0.05)

        logger.info("\n120-day target (10%% of score):")
        logger.info("  LogLoss: %.6f → normalized: %.6f (weight: 5%%)", ll_120, results["target_120_logloss_norm"])
        logger.info("  AUC:     %.6f (weight: 5%%)", auc_120)
        logger.info("  Contribution: %.6f", results["target_120_logloss_norm"] * 0.05 + auc_120 * 0.05)

        logger.info("\n" + "=" * 80)
        logger.info("FINAL WEIGHTED SCORE: %.6f", score)
        logger.info("=" * 80)

    return results


def evaluate_submission_file(
    submission_path: str | Path,
    reference_path: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """Load a submission CSV and evaluate it.

    Parameters
    ----------
    submission_path : str | Path
        Path to submission CSV.
    reference_path : str | Path, optional
        Path to reference CSV with true labels.  If None, tries to find
        a reference file automatically.
    verbose : bool
        Print detailed breakdown.

    Returns
    -------
    dict
        Evaluation results (see ``evaluate_submission``).
    """
    submission_df = pd.read_csv(submission_path)
    logger.info("Loaded submission: %s (%d rows)", submission_path, len(submission_df))

    if reference_path is None:
        # Try to find reference file
        ref_candidates = [
            Path(submission_path).parent.parent / "Train.csv",
            Path(submission_path).parent.parent / "reference.csv",
        ]
        for candidate in ref_candidates:
            if candidate.exists():
                reference_path = candidate
                break

    if reference_path is None:
        logger.warning("No reference file found — cannot compute metrics")
        return {}

    reference_df = pd.read_csv(reference_path)
    logger.info("Loaded reference: %s (%d rows)", reference_path, len(reference_df))

    return evaluate_submission(submission_df, reference_df, verbose=verbose)
