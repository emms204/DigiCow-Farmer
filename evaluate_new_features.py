#!/usr/bin/env python3
"""
Dual-model feature ablation with strict forward time CV.

Protocol:
1. Two model probes:
   - Logistic Regression (scaled, class_weight=None)
   - LightGBM (is_unbalance disabled, no class weighting)
2. Non-overlapping forward validation windows.
3. Fold-wise as-of feature construction:
   FeatureEngineer receives only prior rows before fold cutoff.
4. Keep feature groups only if they improve weighted score while preserving
   7-day LogLoss and fold stability across both models.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import lightgbm as lgb

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from shared.constants import DATE_COL, TARGET_COLS
from shared.data_loader import DataLoader
from shared.evaluation import calculate_weighted_score
from shared.feature_engineering import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# New feature groups to test incrementally.
FEATURE_GROUP_ORDER: list[str] = [
    "confidence_aggregates",
    "interactions",
    "recency_intensity",
    "quality_features",
]


def parse_models(raw: str) -> list[str]:
    """Parse model names and validate availability."""
    models = [m.strip().lower() for m in raw.split(",") if m.strip()]
    valid = {"lr", "lgbm"}
    bad = [m for m in models if m not in valid]
    if bad:
        raise ValueError(f"Unknown models: {bad}. Supported: lr,lgbm")

    if "lgbm" in models and not HAS_LGBM:
        logger.warning("LightGBM is not installed; dropping 'lgbm' from models")
        models = [m for m in models if m != "lgbm"]

    if not models:
        raise ValueError("No valid models selected")
    return models


def build_baseline_group_config() -> dict[str, bool]:
    """Feature group config for minimal baseline used in this script."""
    return {
        "direct_features": True,
        "date_features": False,
        "topic_features": True,
        "trainer_features": False,
        "prior_farmer_history": False,
        "aggregation_features": False,
        "confidence_aggregates": False,
        "interactions": False,
        "recency_intensity": False,
        "quality_features": False,
    }


def rolling_forward_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    min_train_size: float = 0.3,
) -> list[dict[str, Any]]:
    """Build non-overlapping forward-time splits.

    Fold i:
      train = rows with date < cutoff_i
      val   = rows with cutoff_i <= date < cutoff_{i+1}
    """
    if DATE_COL not in df.columns:
        raise ValueError(f"{DATE_COL} missing from dataframe")

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

    splits: list[dict[str, Any]] = []
    for i in range(n_splits):
        start = cutoffs[i]
        end = cutoffs[i + 1]

        train_mask = ordered[DATE_COL] < start
        val_mask = (ordered[DATE_COL] >= start) & (ordered[DATE_COL] < end)

        train_pos = ordered.loc[train_mask, "_pos"].to_numpy(dtype=int)
        val_pos = ordered.loc[val_mask, "_pos"].to_numpy(dtype=int)

        if len(train_pos) == 0 or len(val_pos) == 0:
            continue

        splits.append(
            {
                "fold": i + 1,
                "train_pos": train_pos,
                "val_pos": val_pos,
                "train_cutoff": start,
                "val_end": end,
            }
        )

    if not splits:
        raise ValueError("No valid forward splits generated")
    return splits


def build_estimator(model_name: str) -> Any:
    """Create estimator instance for a model probe."""
    if model_name == "lr":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=8000,
                        C=0.25,
                        penalty="l2",
                        solver="lbfgs",
                        class_weight=None,
                        random_state=42,
                    ),
                ),
            ]
        )

    if model_name == "lgbm":
        return lgb.LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            boosting_type="gbdt",
            n_estimators=700,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            class_weight=None,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

    raise ValueError(f"Unsupported model '{model_name}'")


def evaluate_feature_set_for_model(
    model_name: str,
    group_config: dict[str, bool],
    train_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    y_dict: dict[str, pd.Series],
    n_splits: int,
    focus_target: str = "adopted_within_07_days",
) -> dict[str, Any]:
    """Evaluate one feature config with one model under strict forward CV."""
    logger.info("\n%s", "=" * 88)
    logger.info("Evaluating model=%s", model_name)
    logger.info("Enabled groups: %s", [k for k, v in group_config.items() if v])
    logger.info("%s", "=" * 88)

    splits = rolling_forward_splits(train_df, n_splits=n_splits)
    n_rows = len(train_df)
    oof_preds = {target: np.zeros(n_rows, dtype=float) for target in y_dict}
    oof_mask = np.zeros(n_rows, dtype=bool)

    fold_results: list[dict[str, Any]] = []

    for split in splits:
        fold = split["fold"]
        tr_pos = split["train_pos"]
        va_pos = split["val_pos"]
        train_cutoff = split["train_cutoff"]
        val_end = split["val_end"]

        logger.info(
            "\n  Fold %d: train<%s, val=[%s, %s)",
            fold,
            pd.Timestamp(train_cutoff).date(),
            pd.Timestamp(train_cutoff).date(),
            pd.Timestamp(val_end).date(),
        )
        logger.info("    Train: %d, Val: %d", len(tr_pos), len(va_pos))

        # Strict as-of prior data for this fold.
        prior_asof = prior_df[prior_df[DATE_COL] < train_cutoff].copy()

        tr_df = train_df.iloc[tr_pos]
        va_df = train_df.iloc[va_pos]

        fe = FeatureEngineer(prior_asof)
        fe.set_feature_groups(group_config)
        fe.fit(tr_df)

        X_tr = fe.transform(tr_df)
        X_va = fe.transform(va_df)

        fold_metrics: dict[str, Any] = {
            "fold": fold,
            "train_cutoff": str(pd.Timestamp(train_cutoff).date()),
            "val_end": str(pd.Timestamp(val_end).date()),
            "n_train": len(tr_pos),
            "n_val": len(va_pos),
        }

        for target in TARGET_COLS:
            if target not in y_dict:
                continue
            y = y_dict[target]
            y_tr = y.iloc[tr_pos]
            y_va = y.iloc[va_pos]

            est = build_estimator(model_name)
            est.fit(X_tr, y_tr)
            y_pred = est.predict_proba(X_va)[:, 1]

            oof_preds[target][va_pos] = y_pred

            ll = log_loss(y_va, y_pred)
            auc = roc_auc_score(y_va, y_pred) if len(np.unique(y_va)) > 1 else 0.5
            fold_metrics[f"{target}_ll"] = ll
            fold_metrics[f"{target}_auc"] = auc

        oof_mask[va_pos] = True

        fold_weighted = calculate_weighted_score(
            fold_metrics["adopted_within_07_days_auc"],
            fold_metrics["adopted_within_07_days_ll"],
            fold_metrics["adopted_within_90_days_auc"],
            fold_metrics["adopted_within_90_days_ll"],
            fold_metrics["adopted_within_120_days_auc"],
            fold_metrics["adopted_within_120_days_ll"],
        )
        fold_metrics["weighted_score"] = fold_weighted

        logger.info(
            "    %s: LogLoss=%.6f, AUC=%.6f, Weighted=%.6f",
            focus_target,
            fold_metrics[f"{focus_target}_ll"],
            fold_metrics[f"{focus_target}_auc"],
            fold_weighted,
        )
        fold_results.append(fold_metrics)

    results: dict[str, Any] = {
        "model": model_name,
        "fold_results": fold_results,
    }

    for target in TARGET_COLS:
        if target not in y_dict:
            continue
        y = y_dict[target].to_numpy()
        y_true = y[oof_mask]
        y_hat = oof_preds[target][oof_mask]

        oof_ll = log_loss(y_true, y_hat)
        oof_auc = roc_auc_score(y_true, y_hat) if len(np.unique(y_true)) > 1 else 0.5

        results[f"{target}_oof_ll"] = oof_ll
        results[f"{target}_oof_auc"] = oof_auc
        results[f"{target}_mean_ll"] = np.mean([r[f"{target}_ll"] for r in fold_results])
        results[f"{target}_std_ll"] = np.std([r[f"{target}_ll"] for r in fold_results])
        results[f"{target}_mean_auc"] = np.mean([r[f"{target}_auc"] for r in fold_results])
        results[f"{target}_std_auc"] = np.std([r[f"{target}_auc"] for r in fold_results])

    results["weighted_score"] = calculate_weighted_score(
        results["adopted_within_07_days_oof_auc"],
        results["adopted_within_07_days_oof_ll"],
        results["adopted_within_90_days_oof_auc"],
        results["adopted_within_90_days_oof_ll"],
        results["adopted_within_120_days_oof_auc"],
        results["adopted_within_120_days_oof_ll"],
    )
    results["weighted_mean"] = np.mean([r["weighted_score"] for r in fold_results])
    results["weighted_std"] = np.std([r["weighted_score"] for r in fold_results])

    logger.info("\n  OOF summary (%s):", model_name)
    logger.info(
        "    7d LL=%.6f, 7d AUC=%.6f",
        results["adopted_within_07_days_oof_ll"],
        results["adopted_within_07_days_oof_auc"],
    )
    logger.info(
        "    Weighted=%.6f, fold mean=%.6f±%.6f",
        results["weighted_score"],
        results["weighted_mean"],
        results["weighted_std"],
    )

    return results


def evaluate_feature_set_dual(
    group_config: dict[str, bool],
    model_names: list[str],
    train_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    y_dict: dict[str, pd.Series],
    n_splits: int,
) -> dict[str, dict[str, Any]]:
    """Evaluate a feature set with all selected model probes."""
    out: dict[str, dict[str, Any]] = {}
    for model_name in model_names:
        out[model_name] = evaluate_feature_set_for_model(
            model_name=model_name,
            group_config=group_config,
            train_df=train_df,
            prior_df=prior_df,
            y_dict=y_dict,
            n_splits=n_splits,
        )
    return out


def should_keep_group(
    prev_by_model: dict[str, dict[str, Any]],
    new_by_model: dict[str, dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Decision rule: keep only robust gains across all model probes."""
    reasons: list[str] = []
    has_any_gain = False

    for model_name in new_by_model:
        prev = prev_by_model[model_name]
        new = new_by_model[model_name]

        delta_score = new["weighted_score"] - prev["weighted_score"]
        delta_ll7 = (
            new["adopted_within_07_days_oof_ll"]
            - prev["adopted_within_07_days_oof_ll"]
        )
        delta_std = new["weighted_std"] - prev["weighted_std"]

        reasons.append(
            f"{model_name}: Δscore={delta_score:+.6f}, "
            f"Δ7dLL={delta_ll7:+.6f}, Δstd={delta_std:+.6f}"
        )

        # Hard fail conditions.
        if delta_score < -1e-4 or delta_ll7 > 5e-4 or delta_std > 2e-3:
            return False, reasons

        if delta_score > 1e-4:
            has_any_gain = True

    return has_any_gain, reasons


def print_feature_set_summary(
    name: str,
    results_by_model: dict[str, dict[str, Any]],
) -> None:
    """Print compact per-model summary for a feature set."""
    logger.info("\nFeature set: %s", name)
    logger.info(
        "%-8s  %-10s  %-10s  %-10s  %-10s",
        "Model",
        "7dLL",
        "7dAUC",
        "Weighted",
        "WStd",
    )
    logger.info("%s", "-" * 58)
    for model_name, res in results_by_model.items():
        logger.info(
            "%-8s  %-10.6f  %-10.6f  %-10.6f  %-10.6f",
            model_name,
            res["adopted_within_07_days_oof_ll"],
            res["adopted_within_07_days_oof_auc"],
            res["weighted_score"],
            res["weighted_std"],
        )


def run_incremental_ablation(
    train_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    model_names: list[str],
    groups_to_test: list[str],
    n_splits: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Run incremental keep/drop ablation with dual-model consensus."""
    y_dict = {target: train_df[target] for target in TARGET_COLS if target in train_df}

    all_results: dict[str, dict[str, dict[str, Any]]] = {}

    current_groups = build_baseline_group_config()
    baseline_res = evaluate_feature_set_dual(
        current_groups, model_names, train_df, prior_df, y_dict, n_splits
    )
    all_results["baseline"] = baseline_res
    print_feature_set_summary("baseline", baseline_res)

    current_name = "baseline"
    current_res = baseline_res

    for group_name in groups_to_test:
        if group_name not in FEATURE_GROUP_ORDER:
            logger.warning("Unknown group '%s' skipped", group_name)
            continue

        logger.info("\n%s", "=" * 88)
        logger.info("TESTING GROUP: %s", group_name)
        logger.info("%s", "=" * 88)

        test_groups = current_groups.copy()
        test_groups[group_name] = True
        
        # Special handling: interactions need base features to interact with
        if group_name == "interactions":
            # Enable required base features for interactions
            test_groups["prior_farmer_history"] = True  # For prior_{target}_rate
            test_groups["confidence_aggregates"] = True  # For group_{target}_rate_smoothed
            # recency_intensity is optional (for days_since interaction)
            logger.info("Note: Enabling required base features for interactions: prior_farmer_history, confidence_aggregates")

        test_res = evaluate_feature_set_dual(
            test_groups, model_names, train_df, prior_df, y_dict, n_splits
        )
        all_results[group_name] = test_res
        print_feature_set_summary(group_name, test_res)

        keep, reasons = should_keep_group(current_res, test_res)
        for line in reasons:
            logger.info("  %s", line)

        if keep:
            logger.info("✅ KEEP %s", group_name)
            current_groups = test_groups
            current_name = f"{current_name}+{group_name}"
            current_res = test_res
        else:
            logger.info("❌ SKIP %s", group_name)

    logger.info("\n%s", "=" * 88)
    logger.info("FINAL CHOSEN CONFIG: %s", [k for k, v in current_groups.items() if v])
    logger.info("FINAL MODEL SUMMARIES (%s)", current_name)
    print_feature_set_summary(current_name, current_res)
    logger.info("%s", "=" * 88)

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-model feature ablation")
    parser.add_argument(
        "--models",
        type=str,
        default="lr,lgbm",
        help="Comma-separated model probes: lr,lgbm",
    )
    parser.add_argument(
        "--group",
        type=str,
        help="Test one feature group only",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only run baseline",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of forward CV folds",
    )
    args = parser.parse_args()

    logger.info("%s", "=" * 88)
    logger.info("DUAL-MODEL FEATURE ABLATION")
    logger.info("%s", "=" * 88)

    model_names = parse_models(args.models)
    logger.info("Model probes: %s", model_names)

    loader = DataLoader()
    train_df, _test_df, prior_df, _ = loader.load_all()
    logger.info("Train rows=%d, Prior rows=%d", len(train_df), len(prior_df))

    if args.baseline_only:
        groups_to_test: list[str] = []
    elif args.group:
        groups_to_test = [args.group]
    else:
        groups_to_test = FEATURE_GROUP_ORDER

    run_incremental_ablation(
        train_df=train_df,
        prior_df=prior_df,
        model_names=model_names,
        groups_to_test=groups_to_test,
        n_splits=args.n_splits,
    )

    logger.info("\n%s", "=" * 88)
    logger.info("EVALUATION COMPLETE")
    logger.info("%s", "=" * 88)


if __name__ == "__main__":
    main()
