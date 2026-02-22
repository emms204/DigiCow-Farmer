#!/usr/bin/env python3
"""
Feature Analysis and Ablation Study Tool

This script helps evaluate:
1. Feature importance (via model coefficients/importance)
2. Feature ablation (remove features and measure impact)
3. Feature correlation analysis
4. Feature effectiveness per target

Usage:
    python analyze_features.py                    # Full analysis
    python analyze_features.py --ablation          # Run ablation study
    python analyze_features.py --importance        # Show feature importance only
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.constants import TARGET_COLS
from shared.data_loader import DataLoader
from shared.feature_engineering import FeatureEngineer
from shared.validation import Validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class FeatureAnalyzer:
    """Analyze feature effectiveness and importance."""

    def __init__(self):
        self.loader = DataLoader()
        self.validator = Validator(n_splits=5)
        self.fe = None
        self.X_train = None
        self.y_train = None
        self.feature_names = None

    def load_data(self):
        """Load and prepare data."""
        logger.info("Loading data...")
        train_df, test_df, prior_df, _ = self.loader.load_all()

        # Feature engineering
        self.fe = FeatureEngineer(prior_df)
        self.fe.fit(train_df)
        self.X_train = self.fe.transform(train_df)
        self.feature_names = self.X_train.columns.tolist()

        logger.info(f"Loaded {len(self.X_train)} samples with {len(self.feature_names)} features")
        return train_df

    def analyze_feature_importance(self, target: str = "adopted_within_07_days", method: str = "rf"):
        """Analyze feature importance using different methods.
        
        Parameters
        ----------
        target : str
            Target column name
        method : str
            'rf' for RandomForest, 'lr' for LogisticRegression coefficients
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"FEATURE IMPORTANCE ANALYSIS: {target}")
        logger.info(f"{'='*80}")
        
        train_df = self.load_data()
        y = train_df[target]
        
        if method == "rf":
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            model.fit(self.X_train, y)
            importances = pd.Series(model.feature_importances_, index=self.feature_names)
            importances = importances.sort_values(ascending=False)
            
        elif method == "lr":
            model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
            model.fit(self.X_train, y)
            importances = pd.Series(np.abs(model.coef_[0]), index=self.feature_names)
            importances = importances.sort_values(ascending=False)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"\nTop 20 Features by Importance ({method.upper()}):")
        logger.info("-" * 80)
        for i, (feat, imp) in enumerate(importances.head(20).items(), 1):
            logger.info(f"{i:2d}. {feat:50s} {imp:.6f}")
        
        return importances

    def categorize_features(self) -> dict:
        """Categorize features by type."""
        categories = {
            "direct_categorical": [],
            "direct_binary": [],
            "date_features": [],
            "topic_features": [],
            "trainer_features": [],
            "prior_farmer_history": [],
            "aggregation_features": [],
        }
        
        for feat in self.feature_names:
            if feat in ["gender", "registration", "age", "county", "subcounty", "ward"]:
                categories["direct_categorical"].append(feat)
            elif feat in ["belong_to_cooperative", "has_topic_trained_on"]:
                categories["direct_binary"].append(feat)
            elif feat.startswith(("month", "day_", "quarter", "week_", "is_")):
                categories["date_features"].append(feat)
            elif feat.startswith(("topic_", "topic_cat_")):
                categories["topic_features"].append(feat)
            elif feat.startswith(("trainer_", "has_multiple")):
                categories["trainer_features"].append(feat)
            elif feat.startswith("prior_"):
                categories["prior_farmer_history"].append(feat)
            elif any(x in feat for x in ["_rate", "_count"]) and not feat.startswith("prior_"):
                categories["aggregation_features"].append(feat)
        
        return categories

    def ablation_study(self, target: str = "adopted_within_07_days"):
        """Run ablation study: remove feature categories and measure impact.
        
        This answers: "What happens if we remove this feature category?"
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"ABLATION STUDY: {target}")
        logger.info(f"{'='*80}")
        
        train_df = self.load_data()
        y = train_df[target]
        categories = self.categorize_features()
        
        # Baseline: all features
        baseline_score = self._evaluate_features(self.X_train, y, "ALL FEATURES")
        
        results = [("Baseline (All Features)", baseline_score)]
        
        # Remove each category
        for cat_name, features in categories.items():
            if not features:
                continue
            
            X_ablated = self.X_train.drop(columns=features, errors="ignore")
            score = self._evaluate_features(X_ablated, y, f"Without {cat_name}")
            impact = score - baseline_score
            results.append((f"Without {cat_name}", score, impact))
        
        # Report results
        logger.info(f"\n{'='*80}")
        logger.info("ABLATION RESULTS (Lower Combined Score = Better)")
        logger.info(f"{'='*80}")
        logger.info(f"{'Configuration':<40} {'Combined Score':<15} {'Impact':<15}")
        logger.info("-" * 80)
        
        for result in results:
            if len(result) == 2:
                name, score = result
                logger.info(f"{name:<40} {score:<15.6f} {'(baseline)':<15}")
            else:
                name, score, impact = result
                impact_str = f"{impact:+.6f}" if impact != 0 else "0.000000"
                logger.info(f"{name:<40} {score:<15.6f} {impact_str:<15}")
        
        return results

    def _evaluate_features(self, X: pd.DataFrame, y: pd.Series, description: str) -> float:
        """Evaluate features using CV and return combined score."""
        logger.info(f"\nEvaluating: {description} ({X.shape[1]} features)")
        
        oof_preds = np.zeros(len(y))
        splits = list(self.validator.cv_splits(
            pd.DataFrame(index=y.index),  # Dummy df for splits
            y,
            strategy="stratified",
        ))
        
        for fold_idx, (tr_idx, va_idx) in enumerate(splits):
            model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            oof_preds[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        
        # Calculate metrics
        ll = log_loss(y, oof_preds)
        auc = roc_auc_score(y, oof_preds)
        combined = 0.75 * ll + 0.25 * (1.0 - auc)
        
        logger.info(f"  LogLoss: {ll:.6f}, AUC: {auc:.6f}, Combined: {combined:.6f}")
        return combined

    def analyze_feature_correlations(self, top_n: int = 20):
        """Analyze correlations between features and targets."""
        logger.info(f"\n{'='*80}")
        logger.info("FEATURE-TARGET CORRELATIONS")
        logger.info(f"{'='*80}")
        
        train_df = self.load_data()
        
        # Calculate correlations
        correlations = {}
        for target in TARGET_COLS:
            y = train_df[target]
            corr = self.X_train.apply(lambda col: col.corr(y))
            correlations[target] = corr.abs().sort_values(ascending=False)
        
        # Report top correlations per target
        for target in TARGET_COLS:
            logger.info(f"\n{target}:")
            logger.info("-" * 80)
            top_corr = correlations[target].head(top_n)
            for feat, corr_val in top_corr.items():
                logger.info(f"  {feat:50s} {corr_val:.6f}")
        
        return correlations

    def feature_category_effectiveness(self):
        """Analyze effectiveness of each feature category per target."""
        logger.info(f"\n{'='*80}")
        logger.info("FEATURE CATEGORY EFFECTIVENESS")
        logger.info(f"{'='*80}")
        
        train_df = self.load_data()
        categories = self.categorize_features()
        
        results = {}
        
        for target in TARGET_COLS:
            logger.info(f"\n{target}:")
            y = train_df[target]
            
            # Evaluate each category in isolation
            category_scores = {}
            for cat_name, features in categories.items():
                if not features:
                    continue
                
                X_cat = self.X_train[features]
                score = self._evaluate_features(X_cat, y, f"Only {cat_name}")
                category_scores[cat_name] = score
            
            results[target] = category_scores
            
            # Report
            logger.info("  Category performance (lower = better):")
            for cat, score in sorted(category_scores.items(), key=lambda x: x[1]):
                logger.info(f"    {cat:30s} {score:.6f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Analyze feature effectiveness")
    parser.add_argument("--importance", action="store_true", help="Show feature importance")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--correlations", action="store_true", help="Show feature-target correlations")
    parser.add_argument("--category-effectiveness", action="store_true", help="Analyze category effectiveness")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    
    args = parser.parse_args()
    
    analyzer = FeatureAnalyzer()
    
    if args.all or (not any([args.importance, args.ablation, args.correlations, args.category_effectiveness])):
        # Run all by default
        analyzer.analyze_feature_importance()
        analyzer.analyze_feature_correlations()
        analyzer.ablation_study()
        analyzer.feature_category_effectiveness()
    else:
        if args.importance:
            analyzer.analyze_feature_importance()
        if args.correlations:
            analyzer.analyze_feature_correlations()
        if args.ablation:
            analyzer.ablation_study()
        if args.category_effectiveness:
            analyzer.feature_category_effectiveness()


if __name__ == "__main__":
    main()

