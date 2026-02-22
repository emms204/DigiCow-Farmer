"""
Feature engineering pipeline for DigiCow challenge.

Extracts date-based, topic-based, prior-history, and aggregation features
from the raw DataFrames.  All transforms are stateless or fit on training
data first to prevent data leakage.

New features added (evaluation results):
- Confidence-aware aggregates: ✅ ENABLED BY DEFAULT
  * Massive improvement: LogLoss -0.17, weighted score +0.22
  * Bayesian-smoothed rates with SE and lower confidence bounds
  
- Recency/intensity features: ✅ ENABLED BY DEFAULT
  * Small improvement: LogLoss -0.007
  * Days since last training, training frequency, momentum
  
- Interaction features: ❌ DISABLED BY DEFAULT
  * Worsens performance: LogLoss +0.007, less stable
  * Can be enabled for testing but not recommended
  
- Quality features: ❌ DISABLED BY DEFAULT
  * Worsens performance: LogLoss +0.009
  * Topic diversity, novelty, repeat ratio
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from shared.constants import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    DATE_COL,
    FARMER_COL,
    GROUP_COL,
    ID_COL,
    RANDOM_SEED,
    TARGET_COLS,
    TOPIC_CATEGORIES,
    TOPICS_COL,
    TRAINER_COL,
)

logger = logging.getLogger(__name__)

# Minimal baseline: 9 features only (no aggregates, prior history, or extra engineering).
# Set env DIGICOW_MINIMAL_FEATURES=1 to force all plans to use this (e.g. run_all.py --minimal-features).
MINIMAL_FEATURE_GROUPS: dict[str, bool] = {
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


class FeatureEngineer:
    """Build a feature matrix from raw DataFrames.

    Follows a fit/transform pattern: aggregation statistics (means, counts)
    are computed on the *training* data during ``fit()`` and applied to any
    dataset during ``transform()``.  This prevents target leakage.

    Feature groups can be enabled/disabled for systematic testing.

    Parameters
    ----------
    prior : pd.DataFrame
        The historical Prior dataset (already normalised by DataLoader).

    Example
    -------
    >>> fe = FeatureEngineer(prior_df)
    >>> fe.fit(train_df)
    >>> fe.set_feature_groups({'interactions': True, 'recency_intensity': True})
    >>> X_train = fe.transform(train_df)
    >>> X_test  = fe.transform(test_df)
    """

    def __init__(self, prior: pd.DataFrame) -> None:
        self.prior = prior.copy()
        self.prior[DATE_COL] = pd.to_datetime(self.prior[DATE_COL])

        # Respect minimal-features override (e.g. run_all.py --minimal-features)
        if os.environ.get("DIGICOW_MINIMAL_FEATURES") == "1":
            self._feature_groups = dict(MINIMAL_FEATURE_GROUPS)
            logger.info("Using MINIMAL feature set (9 features, DIGICOW_MINIMAL_FEATURES=1)")
        else:
            # Feature group toggles (defaults based on evaluation results)
            self._feature_groups = {
            'direct_features': True,
            'date_features': True,
            'topic_features': True,
            'trainer_features': True,
            'prior_farmer_history': True,
            'aggregation_features': True,
            'confidence_aggregates': True,  # ✅ KEEP: Massive improvement (LogLoss -0.17, score +0.22)
            'interactions': False,  # ❌ SKIP: Worsens performance (LogLoss +0.007, less stable)
            'recency_intensity': True,  # ✅ KEEP: Small improvement (LogLoss -0.007)
            'quality_features': False,  # ❌ SKIP: Worsens performance (LogLoss +0.009)
            }

        # Lookup tables built during fit()
        self._categorical_maps: dict[str, dict[str, int]] = {}
        self._trainer_map: dict[str, int] = {}
        self._farmer_stats: Optional[pd.DataFrame] = None
        self._trainer_stats: Optional[pd.DataFrame] = None
        self._group_stats: Optional[pd.DataFrame] = None
        self._county_stats: Optional[pd.DataFrame] = None
        self._subcounty_stats: Optional[pd.DataFrame] = None
        self._ward_stats: Optional[pd.DataFrame] = None
        
        # New: confidence-aware stats
        self._trainer_stats_conf: Optional[pd.DataFrame] = None
        self._group_stats_conf: Optional[pd.DataFrame] = None
        self._county_stats_conf: Optional[pd.DataFrame] = None
        self._subcounty_stats_conf: Optional[pd.DataFrame] = None
        self._ward_stats_conf: Optional[pd.DataFrame] = None
        
        # New: farmer recency stats (computed per-row during transform)
        self._farmer_recency_cache: dict[str, pd.DataFrame] = {}

    # ── Public API ─────────────────────────────────────────────────────

    def set_feature_groups(self, groups: dict[str, bool]) -> None:
        """Enable/disable feature groups for testing.
        
        When DIGICOW_MINIMAL_FEATURES=1, this is a no-op and minimal groups stay active.
        
        Parameters
        ----------
        groups : dict[str, bool]
            Dictionary mapping feature group names to enabled/disabled.
            Valid groups:
            - 'direct_features': categorical/binary features
            - 'date_features': temporal features
            - 'topic_features': topic count and categories
            - 'trainer_features': trainer encoding
            - 'prior_farmer_history': farmer-level prior stats
            - 'aggregation_features': group-level adoption rates
            - 'confidence_aggregates': Bayesian-smoothed rates with SE (✅ enabled by default)
            - 'interactions': interaction features (❌ disabled by default - worsens performance)
            - 'recency_intensity': days since last training, momentum (✅ enabled by default)
            - 'quality_features': topic diversity, novelty (❌ disabled by default - worsens performance)
        """
        if os.environ.get("DIGICOW_MINIMAL_FEATURES") == "1":
            return  # keep minimal groups
        for group, enabled in groups.items():
            if group in self._feature_groups:
                self._feature_groups[group] = enabled
            else:
                logger.warning(f"Unknown feature group: {group}")

    def fit(self, train: pd.DataFrame) -> "FeatureEngineer":
        """Compute aggregation statistics from Prior + train labels.

        Parameters
        ----------
        train : pd.DataFrame
            The labelled training set.

        Returns
        -------
        self
        """
        logger.info("Fitting feature engineer …")
        ref = self.prior  # use Prior for aggregation stats (larger, no leak)
        self._fit_categorical_maps(train)
        
        # Core features (always compute - feature groups control usage in transform)
        self._farmer_stats = self._build_farmer_stats(ref)
        self._trainer_stats = self._build_group_agg(ref, TRAINER_COL, "trainer")
        self._group_stats = self._build_group_agg(ref, GROUP_COL, "group")
        self._county_stats = self._build_group_agg(ref, "county", "county")
        self._subcounty_stats = self._build_group_agg(ref, "subcounty", "subcounty")
        self._ward_stats = self._build_group_agg(ref, "ward", "ward")
        
        # New: confidence-aware aggregates (always compute for flexibility)
        self._trainer_stats_conf = self._build_confidence_agg(ref, TRAINER_COL, "trainer")
        self._group_stats_conf = self._build_confidence_agg(ref, GROUP_COL, "group")
        self._county_stats_conf = self._build_confidence_agg(ref, "county", "county")
        self._subcounty_stats_conf = self._build_confidence_agg(ref, "subcounty", "subcounty")
        self._ward_stats_conf = self._build_confidence_agg(ref, "ward", "ward")
        
        logger.info("  Feature engineer fitted.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate the full feature matrix for *df*.

        Complexity: O(n * t) where n = rows, t = max topics per row.
        All joins are hash-based O(1) lookups per row.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame (train or test), already normalised.

        Returns
        -------
        pd.DataFrame
            Feature matrix with numeric and encoded columns, indexed
            by the original row order.  Does **not** include target or
            ID columns.
        """
        features = pd.DataFrame(index=df.index)
        df = df.copy()
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

        # 1. Direct categorical / binary features
        if self._feature_groups.get('direct_features', True):
            features = self._add_direct_features(features, df)

        # 2. Date features
        if self._feature_groups.get('date_features', True):
            features = self._add_date_features(features, df)

        # 3. Topic features
        if self._feature_groups.get('topic_features', True):
            features = self._add_topic_features(features, df)

        # 4. Trainer features (from column itself)
        if self._feature_groups.get('trainer_features', True):
            features = self._add_trainer_direct(features, df)

        # 5. Prior-derived farmer history features
        if self._feature_groups.get('prior_farmer_history', True):
            features = self._add_farmer_history(features, df)

        # 6. Prior-derived group aggregation features
        if self._feature_groups.get('aggregation_features', True):
            features = self._add_aggregation_features(features, df)

        # 7. Confidence-aware aggregates (✅ Enabled by default - massive improvement)
        if self._feature_groups.get('confidence_aggregates', True):
            features = self._add_confidence_aggregates(features, df)

        # 8. Interaction features (❌ Disabled by default - worsens performance)
        if self._feature_groups.get('interactions', False):
            features = self._add_interaction_features(features, df)

        # 9. Recency/intensity features (✅ Enabled by default - small improvement)
        if self._feature_groups.get('recency_intensity', True):
            features = self._add_recency_intensity_features(features, df)

        # 10. Quality features (❌ Disabled by default - worsens performance)
        if self._feature_groups.get('quality_features', False):
            features = self._add_quality_features(features, df)

        # 11. Fill remaining NaN with 0 (farmers not found in Prior)
        features = features.fillna(0)

        logger.info("  Transformed %d rows → %d features", len(df), features.shape[1])
        return features

    def get_feature_names(self, df: pd.DataFrame) -> list[str]:
        """Return the ordered list of feature column names."""
        return self.transform(df.head(1)).columns.tolist()

    # ── Direct features ────────────────────────────────────────────────

    def _add_direct_features(
        self, features: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Encode categorical features as integers and copy binary features.

        Codes are built once in fit() and reused at transform time to keep
        category indices stable between train and test.
        """
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                mapping = self._categorical_maps.get(col, {})
                values = df[col].fillna("UNKNOWN").astype(str)
                features[col] = values.map(mapping).fillna(-1).astype(int)

        for col in BINARY_FEATURES:
            if col in df.columns:
                features[col] = df[col].astype(int)

        return features

    # ── Date features ──────────────────────────────────────────────────

    @staticmethod
    def _add_date_features(
        features: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract temporal signals from the training date.

        Features: month, day-of-week, day-of-year, quarter, week-of-year,
        is-weekend, is-month-start, is-month-end.
        All O(n).
        """
        dt = df[DATE_COL]
        features["month"] = dt.dt.month
        features["day_of_week"] = dt.dt.dayofweek
        features["day_of_year"] = dt.dt.dayofyear
        features["quarter"] = dt.dt.quarter
        features["week_of_year"] = dt.dt.isocalendar().week.astype(int)
        features["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        features["is_month_start"] = (dt.dt.day <= 7).astype(int)
        features["is_month_end"] = (dt.dt.day >= 25).astype(int)
        return features

    # ── Topic features ─────────────────────────────────────────────────

    @staticmethod
    def _add_topic_features(
        features: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Derive topic-count and category-membership features.

        For each row the flat topic list is scanned once — O(n * t) total
        where t is the average number of topics per farmer.
        """
        topics_series = df[TOPICS_COL]

        # Count of unique topics the farmer was trained on
        features["topic_count"] = topics_series.apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        
        # has_topic_trained_on (if not already in binary features)
        if "has_topic_trained_on" not in features.columns:
            features["has_topic_trained_on"] = topics_series.apply(
                lambda x: 1 if isinstance(x, list) and len(x) > 0 else 0
            )

        # Binary flags for broad topic categories
        for cat_name, keywords in TOPIC_CATEGORIES.items():
            features[f"topic_cat_{cat_name}"] = topics_series.apply(
                lambda topics, kw=keywords: (
                    int(
                        any(
                            keyword in topic.lower()
                            for topic in (topics if isinstance(topics, list) else [])
                            for keyword in kw
                        )
                    )
                )
            )

        return features

    # ── Trainer direct features ────────────────────────────────────────

    def _add_trainer_direct(
        self, features: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Encode trainer ID and detect multi-trainer sessions."""
        trainer_values = df[TRAINER_COL].fillna("UNKNOWN").astype(str)
        features["trainer_encoded"] = (
            trainer_values.map(self._trainer_map).fillna(-1).astype(int)
        )
        features["has_multiple_trainers"] = (
            df[TRAINER_COL].str.contains(",", na=False).astype(int)
        )
        return features

    def _fit_categorical_maps(self, train: pd.DataFrame) -> None:
        """Build stable category -> code mappings from train + prior data."""
        self._categorical_maps = {}

        for col in CATEGORICAL_FEATURES:
            series_list = []
            if col in train.columns:
                series_list.append(train[col])
            if col in self.prior.columns:
                series_list.append(self.prior[col])

            if not series_list:
                continue

            merged = pd.concat(series_list, ignore_index=True)
            categories = sorted(merged.fillna("UNKNOWN").astype(str).unique())
            self._categorical_maps[col] = {v: i for i, v in enumerate(categories)}

        trainer_series = []
        if TRAINER_COL in train.columns:
            trainer_series.append(train[TRAINER_COL])
        if TRAINER_COL in self.prior.columns:
            trainer_series.append(self.prior[TRAINER_COL])

        if trainer_series:
            merged = pd.concat(trainer_series, ignore_index=True)
            categories = sorted(merged.fillna("UNKNOWN").astype(str).unique())
            self._trainer_map = {v: i for i, v in enumerate(categories)}

    # ── Prior farmer history ───────────────────────────────────────────

    def _add_farmer_history(
        self, features: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Left-join pre-computed per-farmer statistics from Prior.

        Provides adoption rates, training counts, topic diversity, and
        recency for farmers who appear in the Prior dataset.
        """
        if self._farmer_stats is None:
            return features

        merge_cols = self._farmer_stats.columns.tolist()
        merged = df[[FARMER_COL]].merge(
            self._farmer_stats, on=FARMER_COL, how="left"
        )
        for col in merge_cols:
            if col != FARMER_COL:
                features[col] = merged[col].values

        # Flag: farmer has prior history
        features["has_prior_history"] = (
            df[FARMER_COL].isin(self._farmer_stats[FARMER_COL]).astype(int)
        )
        return features

    def _build_farmer_stats(self, ref: pd.DataFrame) -> pd.DataFrame:
        """Aggregate per-farmer statistics from the Prior dataset.

        Computed once during fit() — O(n_prior) overall.
        """
        grouped = ref.groupby(FARMER_COL)

        stats = pd.DataFrame({FARMER_COL: grouped.groups.keys()})
        stats = stats.set_index(FARMER_COL)

        # Adoption rates per target
        for col in TARGET_COLS:
            if col in ref.columns:
                stats[f"prior_{col}_rate"] = grouped[col].mean()

        # Training engagement metrics
        stats["prior_training_count"] = grouped.size()
        stats["prior_topic_diversity"] = grouped[TOPICS_COL].apply(
            lambda series: len(
                {
                    t
                    for topics in series
                    if isinstance(topics, list)
                    for t in topics
                }
            )
        )

        # Temporal recency: days between first and last training
        stats["prior_span_days"] = grouped[DATE_COL].apply(
            lambda s: (s.max() - s.min()).days
        )

        # Cooperative membership rate across visits
        if "belong_to_cooperative" in ref.columns:
            stats["prior_coop_rate"] = grouped["belong_to_cooperative"].mean()

        stats = stats.reset_index()
        return stats

    # ── Group aggregation features ─────────────────────────────────────

    def _add_aggregation_features(
        self, features: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Join pre-computed group-level adoption stats for trainer,
        group, county, subcounty, and ward."""
        lookup_map = {
            TRAINER_COL: (self._trainer_stats, "trainer"),
            GROUP_COL: (self._group_stats, "group"),
            "county": (self._county_stats, "county"),
            "subcounty": (self._subcounty_stats, "subcounty"),
            "ward": (self._ward_stats, "ward"),
        }
        for col_name, (stats_df, prefix) in lookup_map.items():
            if stats_df is None or col_name not in df.columns:
                continue
            merged = df[[col_name]].merge(stats_df, on=col_name, how="left")
            for c in stats_df.columns:
                if c != col_name:
                    features[c] = merged[c].values
        return features

    @staticmethod
    def _build_group_agg(
        ref: pd.DataFrame, group_col: str, prefix: str
    ) -> pd.DataFrame:
        """Build adoption-rate and count statistics grouped by *group_col*.

        O(n_ref) – single pass group-by.
        """
        if group_col not in ref.columns:
            return pd.DataFrame()

        grouped = ref.groupby(group_col)
        stats = pd.DataFrame({group_col: grouped.groups.keys()})
        stats = stats.set_index(group_col)

        for col in TARGET_COLS:
            if col in ref.columns:
                stats[f"{prefix}_{col}_rate"] = grouped[col].mean()

        stats[f"{prefix}_count"] = grouped.size()
        stats = stats.reset_index()
        return stats

    # ── New: Confidence-aware aggregates ──────────────────────────────

    def _add_confidence_aggregates(
        self, features: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add Bayesian-smoothed rates with standard errors and lower confidence bounds.
        
        These features account for uncertainty in group-level statistics,
        especially for small groups.
        """
        lookup_map = {
            TRAINER_COL: (self._trainer_stats_conf, "trainer"),
            GROUP_COL: (self._group_stats_conf, "group"),
            "county": (self._county_stats_conf, "county"),
            "subcounty": (self._subcounty_stats_conf, "subcounty"),
            "ward": (self._ward_stats_conf, "ward"),
        }
        
        for col_name, (stats_df, prefix) in lookup_map.items():
            if stats_df is None or col_name not in df.columns:
                continue
            merged = df[[col_name]].merge(stats_df, on=col_name, how="left")
            for c in stats_df.columns:
                if c != col_name:
                    features[c] = merged[c].values
        return features

    @staticmethod
    def _build_confidence_agg(
        ref: pd.DataFrame, group_col: str, prefix: str
    ) -> pd.DataFrame:
        """Build confidence-aware statistics with Bayesian smoothing.
        
        For each group, computes:
        - Bayesian-smoothed adoption rate (shrinks toward global mean)
        - Standard error of the rate
        - Lower 95% confidence bound
        """
        if group_col not in ref.columns:
            return pd.DataFrame()

        grouped = ref.groupby(group_col)
        stats = pd.DataFrame({group_col: grouped.groups.keys()})
        stats = stats.set_index(group_col)

        # Global prior (for Bayesian smoothing)
        global_rates = {}
        for col in TARGET_COLS:
            if col in ref.columns:
                global_rates[col] = ref[col].mean()

        for col in TARGET_COLS:
            if col not in ref.columns:
                continue
            
            # For each group, compute Bayesian-smoothed rate
            smoothed_rates = []
            ses = []
            lower_bounds = []
            
            for group_name, group_data in grouped:
                n = len(group_data)
                successes = group_data[col].sum()
                rate = successes / n if n > 0 else 0.0
                
                # Bayesian smoothing: shrink toward global mean
                # Using Beta prior with alpha=global_rate*10, beta=(1-global_rate)*10
                global_rate = global_rates[col]
                alpha_prior = global_rate * 10
                beta_prior = (1 - global_rate) * 10
                
                # Posterior parameters
                alpha_post = alpha_prior + successes
                beta_post = beta_prior + (n - successes)
                
                # Smoothed rate (posterior mean)
                smoothed_rate = alpha_post / (alpha_post + beta_post)
                smoothed_rates.append(smoothed_rate)
                
                # Standard error
                se = np.sqrt(smoothed_rate * (1 - smoothed_rate) / (alpha_post + beta_post))
                ses.append(se)
                
                # Lower 95% confidence bound
                lower_bound = max(0, smoothed_rate - 1.96 * se)
                lower_bounds.append(lower_bound)
            
            stats[f"{prefix}_{col}_rate_smoothed"] = smoothed_rates
            stats[f"{prefix}_{col}_rate_se"] = ses
            stats[f"{prefix}_{col}_rate_lower95"] = lower_bounds

        stats[f"{prefix}_count"] = grouped.size()
        stats = stats.reset_index()
        return stats

    # ── New: Interaction features ─────────────────────────────────────

    def _add_interaction_features(
        self, features: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add interaction features between high-signal features.
        
        Interactions:
        - has_topic_trained_on × group_rate (or smoothed rate)
        - topic_count × prior_adoption_rate
        - has_prior_history × days_since_last_training
        - trainer × topic_category (match score)
        """
        # has_topic_trained_on × group adoption rate (check both regular and smoothed)
        if "has_topic_trained_on" in features.columns:
            for target in TARGET_COLS:
                # Try regular rate first, then smoothed rate
                group_rate_col = f"group_{target}_rate"
                group_smoothed_col = f"group_{target}_rate_smoothed"
                
                if group_rate_col in features.columns:
                    features[f"has_topic_x_group_{target}_rate"] = (
                        features["has_topic_trained_on"] * features[group_rate_col]
                    )
                elif group_smoothed_col in features.columns:
                    features[f"has_topic_x_group_{target}_rate"] = (
                        features["has_topic_trained_on"] * features[group_smoothed_col]
                    )
        
        # topic_count × prior adoption rate
        if "topic_count" in features.columns:
            for target in TARGET_COLS:
                prior_rate_col = f"prior_{target}_rate"
                if prior_rate_col in features.columns:
                    features[f"topic_count_x_prior_{target}_rate"] = (
                        features["topic_count"] * features[prior_rate_col]
                    )
        
        # has_prior_history × days_since_last_training
        if "has_prior_history" in features.columns:
            days_since_col = "days_since_last_training"
            if days_since_col in features.columns:
                features["has_prior_x_days_since"] = (
                    features["has_prior_history"] * features[days_since_col]
                )
        
        # trainer × topic_category match (simplified: trainer effectiveness for dairy/poultry/etc)
        if "trainer_encoded" in features.columns:
            for cat_name in TOPIC_CATEGORIES.keys():
                topic_cat_col = f"topic_cat_{cat_name}"
                if topic_cat_col in features.columns:
                    features[f"trainer_x_topic_{cat_name}"] = (
                        features["trainer_encoded"] * features[topic_cat_col]
                    )
        
        return features

    # ── New: Recency/intensity features ───────────────────────────────

    def _add_recency_intensity_features(
        self, features: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add recency and intensity features from Prior data.
        
        Features:
        - days_since_last_training (farmer-specific)
        - trainings_last_30d, trainings_last_90d (farmer-specific)
        - recent_training_momentum (last 30d vs previous 30d)
        """
        # Sort prior by date for efficient lookups
        prior_sorted = self.prior.sort_values([FARMER_COL, DATE_COL])
        
        for idx, row in df.iterrows():
            farmer = row[FARMER_COL]
            training_day = row[DATE_COL]
            
            # Get farmer's prior history (strictly before training_day)
            farmer_prior = prior_sorted[
                (prior_sorted[FARMER_COL] == farmer) &
                (prior_sorted[DATE_COL] < training_day)
            ]
            
            if len(farmer_prior) == 0:
                # No prior history
                features.loc[idx, "days_since_last_training"] = 999
                features.loc[idx, "trainings_last_30d"] = 0
                features.loc[idx, "trainings_last_90d"] = 0
                features.loc[idx, "recent_training_momentum"] = 0
                continue
            
            # Days since last training
            last_training = farmer_prior[DATE_COL].max()
            days_since = (training_day - last_training).days
            features.loc[idx, "days_since_last_training"] = days_since
            
            # Trainings in last 30/90 days
            cutoff_30d = training_day - pd.Timedelta(days=30)
            cutoff_90d = training_day - pd.Timedelta(days=90)
            
            trainings_30d = len(farmer_prior[farmer_prior[DATE_COL] >= cutoff_30d])
            trainings_90d = len(farmer_prior[farmer_prior[DATE_COL] >= cutoff_90d])
            
            features.loc[idx, "trainings_last_30d"] = trainings_30d
            features.loc[idx, "trainings_last_90d"] = trainings_90d
            
            # Momentum: last 30d vs previous 30d
            cutoff_60d = training_day - pd.Timedelta(days=60)
            trainings_prev_30d = len(
                farmer_prior[
                    (farmer_prior[DATE_COL] >= cutoff_60d) &
                    (farmer_prior[DATE_COL] < cutoff_30d)
                ]
            )
            
            if trainings_prev_30d > 0:
                momentum = (trainings_30d - trainings_prev_30d) / trainings_prev_30d
            else:
                momentum = trainings_30d  # New activity
            features.loc[idx, "recent_training_momentum"] = momentum
        
        return features

    # ── New: Quality features ─────────────────────────────────────────

    def _add_quality_features(
        self, features: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add quality measures of training content.
        
        Features:
        - topic_diversity: unique topics / total topics (repetition ratio)
        - topic_novelty: fraction of current topics not seen in prior trainings
        - repeat_topic_ratio: fraction of topics that are repeats
        """
        # Sort prior by date for efficient lookups
        prior_sorted = self.prior.sort_values([FARMER_COL, DATE_COL])
        
        for idx, row in df.iterrows():
            farmer = row[FARMER_COL]
            training_day = row[DATE_COL]
            current_topics = row[TOPICS_COL]
            
            if not isinstance(current_topics, list) or len(current_topics) == 0:
                features.loc[idx, "topic_diversity"] = 0
                features.loc[idx, "topic_novelty"] = 0
                features.loc[idx, "repeat_topic_ratio"] = 0
                continue
            
            current_topics_set = set(t.lower() for t in current_topics)
            
            # Get farmer's prior topics (strictly before training_day)
            farmer_prior = prior_sorted[
                (prior_sorted[FARMER_COL] == farmer) &
                (prior_sorted[DATE_COL] < training_day)
            ]
            
            if len(farmer_prior) == 0:
                # No prior history - all topics are novel
                features.loc[idx, "topic_diversity"] = 1.0
                features.loc[idx, "topic_novelty"] = 1.0
                features.loc[idx, "repeat_topic_ratio"] = 0.0
                continue
            
            # Collect all prior topics
            prior_topics_set = set()
            for topics in farmer_prior[TOPICS_COL]:
                if isinstance(topics, list):
                    prior_topics_set.update(t.lower() for t in topics)
            
            # Topic diversity: unique / total (handles duplicates in current_topics)
            unique_current = len(current_topics_set)
            total_current = len(current_topics)
            diversity = unique_current / total_current if total_current > 0 else 0
            features.loc[idx, "topic_diversity"] = diversity
            
            # Topic novelty: fraction of current topics not in prior
            novel_topics = current_topics_set - prior_topics_set
            novelty = len(novel_topics) / len(current_topics_set) if len(current_topics_set) > 0 else 0
            features.loc[idx, "topic_novelty"] = novelty
            
            # Repeat topic ratio: fraction of current topics seen before
            repeat_topics = current_topics_set & prior_topics_set
            repeat_ratio = len(repeat_topics) / len(current_topics_set) if len(current_topics_set) > 0 else 0
            features.loc[idx, "repeat_topic_ratio"] = repeat_ratio
        
        return features
