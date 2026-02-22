#!/usr/bin/env python3
"""
Evaluate all submission files using the competition scoring formula.

Since we don't have test labels, this script:
    1. Analyzes probability distributions in submissions
       (separately for LogLoss and AUC columns)
    2. Estimates scores based on expected LogLoss given the distributions
    3. Optionally evaluates on training set (if --train flag is used)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.evaluation import calculate_weighted_score, evaluate_submission, evaluate_submission_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def estimate_logloss_from_distribution(
    probs: np.ndarray, true_pos_rate: float
) -> float:
    """Estimate LogLoss given a probability distribution and true positive rate.

    This is a rough approximation assuming:
        - true_pos_rate fraction of samples are positive (prob = 1)
        - (1 - true_pos_rate) fraction are negative (prob = 0)

    Parameters
    ----------
    probs : np.ndarray
        Predicted probabilities.
    true_pos_rate : float
        True positive class rate.

    Returns
    -------
    float
        Estimated LogLoss.
    """
    # For negatives (1 - true_pos_rate of data): log(1 - p)
    neg_ll = -(1 - true_pos_rate) * np.mean(np.log(np.clip(1 - probs, 1e-15, 1)))

    # For positives (true_pos_rate of data): log(p)
    pos_ll = -true_pos_rate * np.mean(np.log(np.clip(probs, 1e-15, 1)))

    return neg_ll + pos_ll


def analyze_submission_file(submission_path: Path) -> dict:
    """Analyze a submission file's probability distributions."""
    df = pd.read_csv(submission_path)
    name = submission_path.stem

    results = {
        "name": name,
        "file": str(submission_path),
    }

    # True positive rates from training data
    true_rates = {"07": 0.0113, "90": 0.0158, "120": 0.0223}

    for target in ["07", "90", "120"]:
        col_ll = f"Target_{target}_LogLoss"
        col_auc = f"Target_{target}_AUC"

        probs_ll = df[col_ll].values
        probs_auc = df[col_auc].values if col_auc in df.columns else probs_ll

        # LogLoss-column distribution stats
        results[f"{target}_mean_ll"] = np.mean(probs_ll)
        results[f"{target}_median_ll"] = np.median(probs_ll)
        results[f"{target}_p95_ll"] = np.percentile(probs_ll, 95)
        results[f"{target}_max_ll"] = np.max(probs_ll)
        results[f"{target}_min_ll"] = np.min(probs_ll)

        # AUC-column distribution stats
        results[f"{target}_mean_auc_col"] = np.mean(probs_auc)
        results[f"{target}_median_auc_col"] = np.median(probs_auc)
        results[f"{target}_p95_auc_col"] = np.percentile(probs_auc, 95)
        results[f"{target}_max_auc_col"] = np.max(probs_auc)
        results[f"{target}_min_auc_col"] = np.min(probs_auc)

        # Divergence between submission columns for same target
        results[f"{target}_mean_abs_col_diff"] = np.mean(np.abs(probs_ll - probs_auc))

        # Estimate LogLoss
        true_rate = true_rates[target]
        est_ll = estimate_logloss_from_distribution(probs_ll, true_rate)
        results[f"{target}_est_logloss"] = est_ll

        # Estimate AUC (rough: based on how well probabilities separate)
        # Sort by probability and assume top true_rate fraction are positives
        sorted_idx = np.argsort(-probs_auc)
        n_pos = int(len(probs_auc) * true_rate)
        # Rough AUC estimate: if top predictions are well-separated, AUC is high
        top_mean = np.mean(probs_auc[sorted_idx[:n_pos]])
        bottom_mean = np.mean(probs_auc[sorted_idx[-n_pos:]])
        est_auc = 0.5 + 0.5 * (top_mean - bottom_mean)  # rough approximation
        results[f"{target}_est_auc"] = np.clip(est_auc, 0.5, 1.0)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate submission files")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Evaluate on training set (requires regenerating predictions)",
    )
    parser.add_argument(
        "--submissions-dir",
        type=str,
        default="submissions",
        help="Directory containing submission files",
    )
    args = parser.parse_args()

    submissions_dir = Path(args.submissions_dir)
    if not submissions_dir.exists():
        logger.error("Submissions directory not found: %s", submissions_dir)
        return

    # Find all submission CSVs
    submission_files = sorted(submissions_dir.glob("*_submission.csv"))
    if not submission_files:
        logger.error("No submission files found in %s", submissions_dir)
        return

    # When --train: load Train.csv and look for train (OOF) submission files
    train_dir = submissions_dir / "train"
    reference_df = None
    if args.train:
        root = submissions_dir.resolve().parent
        train_csv = root / "Train.csv"
        if train_csv.exists():
            reference_df = pd.read_csv(train_csv)
            logger.info("Loaded Train.csv (%d rows) for real score evaluation.", len(reference_df))
        else:
            logger.warning("Train.csv not found at %s — skipping train evaluation.", train_csv)
        if not train_dir.exists():
            logger.warning("Train submission dir not found: %s", train_dir)

    logger.info("=" * 90)
    logger.info("ANALYZING SUBMISSION FILES")
    logger.info("=" * 90)

    all_results = []

    for sub_file in submission_files:
        logger.info("\n" + "-" * 90)
        logger.info("Analyzing: %s", sub_file.name)
        logger.info("-" * 90)

        results = analyze_submission_file(sub_file)

        # Calculate estimated score
        est_score = calculate_weighted_score(
            results["07_est_auc"],
            results["07_est_logloss"],
            results["90_est_auc"],
            results["90_est_logloss"],
            results["120_est_auc"],
            results["120_est_logloss"],
        )
        results["estimated_score"] = est_score

        # Real weighted score on training set (OOF) when --train and train file exists
        results["train_score"] = None
        if args.train and reference_df is not None and train_dir.exists():
            train_file = train_dir / sub_file.name
            if train_file.exists():
                sub_df = pd.read_csv(train_file)
                eval_results = evaluate_submission(sub_df, reference_df=reference_df, verbose=False)
                if eval_results:
                    results["train_score"] = eval_results["score"]

        # Print summary
        logger.info("\n7-day target (80%% of score):")
        logger.info("  LogLoss col mean: %.6f (true rate: 0.0113)", results["07_mean_ll"])
        logger.info("  AUC col mean:     %.6f", results["07_mean_auc_col"])
        logger.info("  Mean |LL-AUC|:    %.6f", results["07_mean_abs_col_diff"])
        logger.info("  Est LogLoss: %.6f → normalized: %.6f", results["07_est_logloss"], 1 - (results["07_est_logloss"] / 0.56926))
        logger.info("  Est AUC: %.6f", results["07_est_auc"])

        logger.info("\n90-day target (10%% of score):")
        logger.info("  LogLoss col mean: %.6f (true rate: 0.0158)", results["90_mean_ll"])
        logger.info("  AUC col mean:     %.6f", results["90_mean_auc_col"])
        logger.info("  Mean |LL-AUC|:    %.6f", results["90_mean_abs_col_diff"])
        logger.info("  Est LogLoss: %.6f → normalized: %.6f", results["90_est_logloss"], 1 - (results["90_est_logloss"] / 0.56926))
        logger.info("  Est AUC: %.6f", results["90_est_auc"])

        logger.info("\n120-day target (10%% of score):")
        logger.info("  LogLoss col mean: %.6f (true rate: 0.0223)", results["120_mean_ll"])
        logger.info("  AUC col mean:     %.6f", results["120_mean_auc_col"])
        logger.info("  Mean |LL-AUC|:    %.6f", results["120_mean_abs_col_diff"])
        logger.info("  Est LogLoss: %.6f → normalized: %.6f", results["120_est_logloss"], 1 - (results["120_est_logloss"] / 0.56926))
        logger.info("  Est AUC: %.6f", results["120_est_auc"])

        logger.info("\n" + "=" * 90)
        logger.info("ESTIMATED COMPETITION SCORE: %.6f", est_score)
        if results["train_score"] is not None:
            logger.info("REAL WEIGHTED SCORE (train OOF): %.6f", results["train_score"])
        logger.info("=" * 90)

        all_results.append(results)

    # Summary table
    logger.info("\n\n" + "=" * 90)
    logger.info("SUMMARY TABLE")
    logger.info("=" * 90)
    if args.train and any(r.get("train_score") is not None for r in all_results):
        logger.info(
            f"{'Submission':<30} {'Train Score':<12} {'Est Score':<12} {'7d LL':<10} {'7d AUC':<10}"
        )
        logger.info("-" * 90)
        def sort_key(r):
            ts = r.get("train_score")
            return (ts if ts is not None else -1.0, r["estimated_score"])
        for r in sorted(all_results, key=sort_key, reverse=True):
            ts = r["train_score"] if r.get("train_score") is not None else float("nan")
            logger.info(
                f"{r['name']:<30} {ts:<12.6f} {r['estimated_score']:<12.6f} "
                f"{r['07_est_logloss']:<10.6f} {r['07_est_auc']:<10.6f}"
            )
    else:
        logger.info(
            f"{'Submission':<30} {'Est Score':<12} {'7d LL':<10} {'7d AUC':<10} "
            f"{'7d LLmean':<10} {'7d AUCmean':<10} {'7d |Δ|':<10}"
        )
        logger.info("-" * 90)
        for r in sorted(all_results, key=lambda x: x["estimated_score"], reverse=True):
            logger.info(
                f"{r['name']:<30} {r['estimated_score']:<12.6f} "
                f"{r['07_est_logloss']:<10.6f} {r['07_est_auc']:<10.6f} "
                f"{r['07_mean_ll']:<10.6f} {r['07_mean_auc_col']:<10.6f} "
                f"{r['07_mean_abs_col_diff']:<10.6f}"
            )

    logger.info("=" * 90)
    if args.train and not any(r.get("train_score") is not None for r in all_results):
        logger.info("\nNo train submission files found in %s", train_dir)
        logger.info("Generate them with: python scripts/generate_e6_plan9_submissions.py --write-train")
    else:
        logger.info("\nNote: Est Score = heuristic without test labels. Train Score = real weighted score on training (OOF) set.")


if __name__ == "__main__":
    main()
