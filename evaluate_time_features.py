#!/usr/bin/env python3
"""
Time-Coded Features Evaluation with Strict Temporal Constraints

Builds rolling/lag features with strict event_date < training_day constraint.
Evaluates using time-based CV with rolling cutoffs.
Keeps features only if they improve weighted metric consistently, especially 7-day LogLoss.
Checks calibration shift: if AUC improves but 7d LogLoss worsens, drop feature.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.constants import DATE_COL, FARMER_COL, GROUP_COL, TARGET_COLS, TRAINER_COL
from shared.data_loader import DataLoader
from shared.evaluation import calculate_weighted_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Minimal feature set (baseline)
MINIMAL_FEATURES = [
    "has_topic_trained_on",
    "topic_count",
    "gender",
    "age",
    "county",
    "subcounty",
    "ward",
    "belong_to_cooperative",
    "registration",
]


class TimeFeatureBuilder:
    """Build time-coded features with strict temporal constraints."""

    def __init__(self, prior_df: pd.DataFrame, train_df: pd.DataFrame):
        """Initialize with prior and train data.
        
        Parameters
        ----------
        prior_df : pd.DataFrame
            Historical data for computing rolling statistics
        train_df : pd.DataFrame
            Training data (for reference)
        """
        self.prior_df = prior_df.copy()
        self.train_df = train_df.copy()
        
        # Combine for computing rolling stats
        self.combined_df = pd.concat([prior_df, train_df], ignore_index=True)
        self.combined_df = self.combined_df.sort_values(DATE_COL)
        
    def build_rolling_features(
        self, 
        df: pd.DataFrame,
        window_days: list[int] = [7, 14, 30, 60, 90]
    ) -> pd.DataFrame:
        """Build rolling window features with strict temporal constraints.
        
        Optimized version using vectorized operations where possible.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to add features to
        window_days : list[int]
            Rolling window sizes in days
            
        Returns
        -------
        pd.DataFrame
            DataFrame with rolling features added
        """
        features = pd.DataFrame(index=df.index)
        
        logger.info(f"Building rolling features for {len(df)} rows...")
        
        # Sort combined data for efficient lookups
        combined_sorted = self.combined_df.sort_values(DATE_COL).copy()
        combined_sorted['date_rank'] = combined_sorted.groupby(DATE_COL).ngroup()
        
        # Pre-compute date ranges for efficiency
        df_sorted = df.sort_values(DATE_COL).copy()
        
        for window in window_days:
            logger.info(f"  Processing {window}d window...")
            
            rolling_global = []
            rolling_farmer = []
            rolling_group = []
            
            for idx, row in df_sorted.iterrows():
                training_day = row[DATE_COL]
                cutoff = training_day - pd.Timedelta(days=window)
                
                # Strict constraint: only use events before training_day
                historical = combined_sorted[
                    (combined_sorted[DATE_COL] < training_day) &
                    (combined_sorted[DATE_COL] >= cutoff)
                ]
                
                # Global count
                rolling_global.append(len(historical))
                
                # Farmer-specific count
                if pd.notna(row[FARMER_COL]):
                    farmer_count = len(historical[historical[FARMER_COL] == row[FARMER_COL]])
                    rolling_farmer.append(farmer_count)
                else:
                    rolling_farmer.append(0)
                
                # Group-specific count
                if GROUP_COL in row.index and pd.notna(row.get(GROUP_COL)):
                    group_count = len(historical[historical[GROUP_COL] == row[GROUP_COL]])
                    rolling_group.append(group_count)
                else:
                    rolling_group.append(0)
            
            # Assign to original index order
            features.loc[df_sorted.index, f"rolling_training_count_{window}d"] = rolling_global
            features.loc[df_sorted.index, f"rolling_farmer_training_count_{window}d"] = rolling_farmer
            features.loc[df_sorted.index, f"rolling_group_training_count_{window}d"] = rolling_group
        
        logger.info(f"  Built {len(features.columns)} rolling features")
        return features
    
    def build_lag_features(
        self,
        df: pd.DataFrame,
        lag_days: list[int] = [7, 14, 30, 60, 90]
    ) -> pd.DataFrame:
        """Build lag features: days since last training event.
        
        Optimized version.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to add features to
        lag_days : list[int]
            Lag periods to check
            
        Returns
        -------
        pd.DataFrame
            DataFrame with lag features added
        """
        features = pd.DataFrame(index=df.index)
        
        logger.info(f"Building lag features for {len(df)} rows...")
        
        # Sort for efficiency
        combined_sorted = self.combined_df.sort_values(DATE_COL).copy()
        df_sorted = df.sort_values(DATE_COL).copy()
        
        for lag in lag_days:
            logger.info(f"  Processing {lag}d lag...")
            
            days_since_global_list = []
            days_since_farmer_list = []
            
            for idx, row in df_sorted.iterrows():
                training_day = row[DATE_COL]
                cutoff = training_day - pd.Timedelta(days=lag)
                
                # Strict constraint: only use events before training_day
                historical = combined_sorted[combined_sorted[DATE_COL] < training_day]
                
                if len(historical) == 0:
                    days_since_global_list.append(999)
                    days_since_farmer_list.append(999)
                    continue
                
                # Days since last global training
                last_global = historical[DATE_COL].max()
                days_since_global = (training_day - last_global).days
                
                # Check if there was training in last 'lag' days
                had_training = (historical[DATE_COL] >= cutoff).any()
                days_since_global_list.append(0 if had_training else days_since_global)
                
                # Days since farmer's last training
                if pd.notna(row[FARMER_COL]):
                    farmer_historical = historical[historical[FARMER_COL] == row[FARMER_COL]]
                    if len(farmer_historical) > 0:
                        last_farmer = farmer_historical[DATE_COL].max()
                        days_since_farmer = (training_day - last_farmer).days
                        had_farmer_training = (farmer_historical[DATE_COL] >= cutoff).any()
                        days_since_farmer_list.append(0 if had_farmer_training else days_since_farmer)
                    else:
                        days_since_farmer_list.append(999)
                else:
                    days_since_farmer_list.append(999)
            
            features.loc[df_sorted.index, f"days_since_last_training_{lag}d"] = days_since_global_list
            features.loc[df_sorted.index, f"days_since_farmer_last_training_{lag}d"] = days_since_farmer_list
        
        logger.info(f"  Built {len(features.columns)} lag features")
        return features


def rolling_cutoff_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    min_train_size: float = 0.3
) -> list[tuple[np.ndarray, np.ndarray, str, str]]:
    """Generate non-overlapping forward folds for time-based CV.
    
    Fold k: train on [min_date, cutoff_k), validate on [cutoff_k, cutoff_{k+1})
    This ensures no sample is predicted multiple times.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DATE_COL (must preserve original index)
    n_splits : int
        Number of splits
    min_train_size : float
        Minimum fraction of data for training
        
    Returns
    -------
    list[tuple[np.ndarray, np.ndarray, str, str]]
        List of (train_idx, val_idx, train_cutoff, val_cutoff) tuples
    """
    # Sort by date but preserve original index
    df_sorted = df.sort_values(DATE_COL)
    min_date = df_sorted[DATE_COL].min()
    max_date = df_sorted[DATE_COL].max()
    
    # Generate non-overlapping cutoffs
    total_range = max_date - min_date
    train_range = total_range * min_train_size
    
    splits = []
    prev_cutoff = min_date + train_range
    
    for i in range(n_splits):
        # Calculate validation window
        progress = (i + 1) / (n_splits + 1)
        val_cutoff = min_date + total_range * (
            min_train_size + progress * (1 - min_train_size)
        )
        
        # Train: [min_date, prev_cutoff)
        # Val: [prev_cutoff, val_cutoff)
        train_mask = (df_sorted[DATE_COL] >= min_date) & (df_sorted[DATE_COL] < prev_cutoff)
        val_mask = (df_sorted[DATE_COL] >= prev_cutoff) & (df_sorted[DATE_COL] < val_cutoff)
        
        # Use original indices from df_sorted (which preserves df.index)
        train_idx = df_sorted.index[train_mask].values
        val_idx = df_sorted.index[val_mask].values
        
        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((
                train_idx,
                val_idx,
                prev_cutoff.strftime("%Y-%m-%d"),
                val_cutoff.strftime("%Y-%m-%d")
            ))
        
        prev_cutoff = val_cutoff
    
    return splits


def evaluate_feature_set(
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    target: str,
    feature_set_name: str
) -> dict:
    """Evaluate a feature set using non-overlapping forward folds.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (must align with df.index)
    y : pd.Series
        Target values (must align with df.index)
    df : pd.DataFrame
        Original dataframe (for date-based splitting, preserves index)
    target : str
        Target name
    feature_set_name : str
        Name of feature set for logging
        
    Returns
    -------
    dict
        Evaluation results including competition-weighted score
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating: {feature_set_name}")
    logger.info(f"  Features: {len(X.columns)}")
    logger.info(f"  Target: {target}")
    logger.info(f"{'='*80}")
    
    # Ensure X and y align with df.index
    assert X.index.equals(df.index), "X.index must match df.index"
    assert y.index.equals(df.index), "y.index must match df.index"
    
    splits = rolling_cutoff_splits(df, n_splits=5)
    
    fold_results = []
    oof_preds = np.full(len(y), np.nan)  # Use NaN to detect unpredicted samples
    oof_mask = np.zeros(len(y), dtype=bool)
    
    # Competition weights (from shared/evaluation.py)
    SCALING = 0.56926
    if target == "adopted_within_07_days":
        logloss_weight = 0.65
        auc_weight = 0.15
    elif target == "adopted_within_90_days":
        logloss_weight = 0.05
        auc_weight = 0.05
    else:  # 120_days
        logloss_weight = 0.05
        auc_weight = 0.05
    
    for fold_idx, (tr_idx, va_idx, train_cutoff, val_cutoff) in enumerate(splits):
        logger.info(f"\n  Fold {fold_idx + 1}: train<{train_cutoff}, val=[{train_cutoff}, {val_cutoff})")
        logger.info(f"    Train: {len(tr_idx)} samples, Val: {len(va_idx)} samples")
        
        # Use .loc to ensure correct index alignment
        X_tr, X_va = X.loc[tr_idx], X.loc[va_idx]
        y_tr, y_va = y.loc[tr_idx], y.loc[va_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        
        # Train model (no class_weight, stronger regularization)
        model = LogisticRegression(
            max_iter=5000,
            C=0.1,  # Stronger regularization
            penalty='l2',
            solver='lbfgs',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_tr_scaled, y_tr)
        
        # Predict
        y_pred = model.predict_proba(X_va_scaled)[:, 1]
        oof_preds[va_idx] = y_pred
        oof_mask[va_idx] = True
        
        # Evaluate
        ll = log_loss(y_va, y_pred)
        auc = roc_auc_score(y_va, y_pred) if len(np.unique(y_va)) > 1 else 0.5
        
        # Competition-weighted score
        ll_norm = 1 - (ll / SCALING)
        weighted_score = auc_weight * auc + logloss_weight * ll_norm
        
        fold_results.append({
            'fold': fold_idx + 1,
            'train_cutoff': train_cutoff,
            'val_cutoff': val_cutoff,
            'log_loss': ll,
            'auc': auc,
            'weighted_score': weighted_score,
            'n_train': len(tr_idx),
            'n_val': len(va_idx),
        })
        
        logger.info(f"    LogLoss: {ll:.6f}, AUC: {auc:.6f}, Weighted: {weighted_score:.6f}")
    
    # Overall OOF metrics (only on samples that were predicted exactly once)
    oof_preds_clean = oof_preds[oof_mask]
    y_clean = y[oof_mask]
    
    oof_ll = log_loss(y_clean, oof_preds_clean)
    oof_auc = roc_auc_score(y_clean, oof_preds_clean) if len(np.unique(y_clean)) > 1 else 0.5
    
    # Competition-weighted OOF score
    oof_ll_norm = 1 - (oof_ll / SCALING)
    oof_weighted = auc_weight * oof_auc + logloss_weight * oof_ll_norm
    
    results = {
        'feature_set': feature_set_name,
        'n_features': len(X.columns),
        'fold_results': fold_results,
        'oof_log_loss': oof_ll,
        'oof_auc': oof_auc,
        'oof_weighted_score': oof_weighted,
        'mean_fold_ll': np.mean([r['log_loss'] for r in fold_results]),
        'std_fold_ll': np.std([r['log_loss'] for r in fold_results]),
        'mean_fold_auc': np.mean([r['auc'] for r in fold_results]),
        'std_fold_auc': np.std([r['auc'] for r in fold_results]),
        'mean_fold_weighted': np.mean([r['weighted_score'] for r in fold_results]),
        'std_fold_weighted': np.std([r['weighted_score'] for r in fold_results]),
    }
    
    logger.info(f"\n  Overall OOF: LogLoss={oof_ll:.6f}, AUC={oof_auc:.6f}, Weighted={oof_weighted:.6f}")
    logger.info(f"  Mean across folds: LogLoss={results['mean_fold_ll']:.6f}±{results['std_fold_ll']:.6f}, "
                f"AUC={results['mean_fold_auc']:.6f}±{results['std_fold_auc']:.6f}, "
                f"Weighted={results['mean_fold_weighted']:.6f}±{results['std_fold_weighted']:.6f}")
    
    return results


def build_minimal_features(df: pd.DataFrame, fe: Optional[object] = None) -> pd.DataFrame:
    """Build minimal feature set (baseline).
    
    Uses FeatureEngineer if provided to ensure consistent encoding.
    """
    if fe is not None:
        # Use existing feature engineer for consistent encoding
        X_all = fe.transform(df)
        # Select only minimal features
        minimal_cols = [col for col in MINIMAL_FEATURES if col in X_all.columns]
        return X_all[minimal_cols]
    else:
        # Manual encoding (fallback)
        features = pd.DataFrame(index=df.index)
        
        # Direct categorical features (need encoding)
        categoricals = ["gender", "registration", "age", "county", "subcounty", "ward"]
        for cat in categoricals:
            if cat in df.columns:
                # Simple integer encoding
                features[cat] = pd.Categorical(df[cat]).codes
        
        # Binary features
        if "belong_to_cooperative" in df.columns:
            features["belong_to_cooperative"] = df["belong_to_cooperative"].astype(int)
        
        # Topic features
        if "topics_list" in df.columns:
            features["has_topic_trained_on"] = df["topics_list"].apply(
                lambda x: 1 if isinstance(x, list) and len(x) > 0 else 0
            )
            features["topic_count"] = df["topics_list"].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        
        return features.fillna(0)


def main():
    """Main evaluation pipeline."""
    logger.info("="*80)
    logger.info("TIME-CODED FEATURES EVALUATION")
    logger.info("="*80)
    
    # Load data
    loader = DataLoader()
    train_df, test_df, prior_df, _ = loader.load_all()
    
    # Focus on 7-day target (most important)
    target = "adopted_within_07_days"
    y = train_df[target]
    
    logger.info(f"\nTarget: {target}")
    logger.info(f"Positive rate: {y.mean():.4f} ({y.sum()} / {len(y)})")
    
    # Build feature engineer for consistent encoding
    from shared.feature_engineering import FeatureEngineer
    fe = FeatureEngineer(prior_df)
    fe.fit(train_df)
    
    # Build minimal feature set (baseline)
    logger.info("\n" + "="*80)
    logger.info("1. Building Minimal Feature Set (Baseline)")
    logger.info("="*80)
    X_minimal = build_minimal_features(train_df, fe=fe)
    logger.info(f"Minimal features: {list(X_minimal.columns)}")
    
    # Evaluate minimal set
    results_minimal = evaluate_feature_set(
        X_minimal, y, train_df, target, "Minimal (9 features)"
    )
    
    # Build time-coded features
    logger.info("\n" + "="*80)
    logger.info("2. Building Time-Coded Features")
    logger.info("="*80)
    
    time_builder = TimeFeatureBuilder(prior_df, train_df)
    
    # Build time features (pruned to 2-4 non-redundant features)
    logger.info("\nBuilding pruned time features (2-4 non-redundant)...")
    
    # Only farmer-specific features (most signal, least collinear)
    rolling_features = time_builder.build_rolling_features(
        train_df,
        window_days=[30]  # Single window to reduce redundancy
    )
    
    # Only farmer lag (most informative)
    lag_features = time_builder.build_lag_features(
        train_df,
        lag_days=[30]  # Single lag to reduce redundancy
    )
    
    # Select only farmer-specific features (less collinear than global/group)
    time_feature_cols = [
        col for col in list(rolling_features.columns) + list(lag_features.columns)
        if 'farmer' in col  # Only farmer-specific
    ]
    
    time_features_pruned = pd.concat([
        rolling_features[[col for col in rolling_features.columns if col in time_feature_cols]],
        lag_features[[col for col in lag_features.columns if col in time_feature_cols]]
    ], axis=1)
    
    # Combine: minimal + pruned time features
    X_with_time = pd.concat([X_minimal, time_features_pruned], axis=1)
    logger.info(f"\nTotal features with time: {len(X_with_time.columns)}")
    logger.info(f"  Minimal: {len(X_minimal.columns)}")
    logger.info(f"  Time (pruned): {len(time_features_pruned.columns)}")
    logger.info(f"  Time features: {list(time_features_pruned.columns)}")
    
    # Evaluate with time features
    results_with_time = evaluate_feature_set(
        X_with_time, y, train_df, target, "Minimal + Time Features"
    )
    
    # Compare results
    logger.info("\n" + "="*80)
    logger.info("COMPARISON")
    logger.info("="*80)
    
    logger.info(f"\n{'Metric':<30} {'Minimal':<20} {'With Time':<20} {'Change':<20}")
    logger.info("-" * 90)
    
    metrics = [
        ('OOF Weighted Score', 'oof_weighted_score', False),  # Higher is better (competition metric)
        ('OOF LogLoss', 'oof_log_loss', True),  # Lower is better
        ('OOF AUC', 'oof_auc', False),  # Higher is better
        ('Mean Fold Weighted', 'mean_fold_weighted', False),
        ('Mean Fold LogLoss', 'mean_fold_ll', True),
        ('Mean Fold AUC', 'mean_fold_auc', False),
        ('Std Fold Weighted', 'std_fold_weighted', True),  # Lower std = more stable
    ]
    
    for metric_name, metric_key, lower_is_better in metrics:
        minimal_val = results_minimal[metric_key]
        time_val = results_with_time[metric_key]
        change = time_val - minimal_val
        
        if lower_is_better:
            change_str = f"{change:+.6f} ({'✅' if change < 0 else '❌'})"
        else:
            change_str = f"{change:+.6f} ({'✅' if change > 0 else '❌'})"
        
        logger.info(f"{metric_name:<30} {minimal_val:<20.6f} {time_val:<20.6f} {change_str:<20}")
    
    # Calibration shift check
    logger.info("\n" + "="*80)
    logger.info("CALIBRATION SHIFT CHECK")
    logger.info("="*80)
    
    weighted_change = results_with_time['oof_weighted_score'] - results_minimal['oof_weighted_score']
    ll_change = results_with_time['oof_log_loss'] - results_minimal['oof_log_loss']
    auc_change = results_with_time['oof_auc'] - results_minimal['oof_auc']
    std_change = results_with_time['std_fold_weighted'] - results_minimal['std_fold_weighted']
    
    logger.info(f"Weighted Score change: {weighted_change:+.6f} (competition metric)")
    logger.info(f"LogLoss change: {ll_change:+.6f}")
    logger.info(f"AUC change: {auc_change:+.6f}")
    logger.info(f"Stability (std) change: {std_change:+.6f} (lower is better)")
    
    if weighted_change > 0:
        logger.info("✅ Weighted score improved - time features are beneficial!")
    elif auc_change > 0 and ll_change > 0:
        logger.warning("⚠️  CALIBRATION SHIFT: AUC improved but LogLoss worsened!")
        logger.warning("   Recommendation: Drop or regularize time features.")
    else:
        logger.info("❌ Weighted score worsened - time features may not be helpful")
    
    if std_change > 0:
        logger.warning("⚠️  Stability decreased - time features increase variance")
    
    # Feature importance (if time features help)
    if weighted_change > 0:
        logger.info("\n" + "="*80)
        logger.info("TIME FEATURE IMPORTANCE")
        logger.info("="*80)
        
        # Train final model to see feature importance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_with_time)
        model = LogisticRegression(max_iter=5000, C=0.1, penalty='l2', solver='lbfgs', random_state=42, n_jobs=-1)
        model.fit(X_scaled, y)
        
        importances = pd.Series(
            np.abs(model.coef_[0]),
            index=X_with_time.columns
        ).sort_values(ascending=False)
        
        # Show which time features are most important
        time_feat_importances = importances[importances.index.isin(time_features_pruned.columns)]
        
        logger.info("\nTop 20 features by coefficient magnitude:")
        for i, (feat, imp) in enumerate(importances.head(20).items(), 1):
            is_time_feat = feat in time_features_pruned.columns
            marker = "⏰" if is_time_feat else "  "
            logger.info(f"  {i:2d}. {marker} {feat:50s} {imp:.6f}")
        
        if len(time_feat_importances) > 0:
            logger.info(f"\nTime feature importances:")
            for feat, imp in time_feat_importances.items():
                logger.info(f"  ⏰ {feat:50s} {imp:.6f}")
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()

