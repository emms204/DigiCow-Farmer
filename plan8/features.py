"""
Gating and cohort features for Plan 8.

- has_prior_history: from prior as-of (farmer appears in prior before cutoff).
- Gating matrix: has_prior_history, has_topic_trained_on, topic_count, county/subcounty/ward/group encoded.
- Cohorts: (has_prior_history x has_topic_trained_on) => 4 groups.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from shared.constants import (
    DATE_COL,
    FARMER_COL,
    GROUP_COL,
    TOPICS_COL,
)

# Columns used for gating (must exist in df).
GATING_CATEGORICAL = ["county", "subcounty", "ward", GROUP_COL]


def compute_has_prior_history(df: pd.DataFrame, prior_asof: pd.DataFrame) -> np.ndarray:
    """Binary: 1 if farmer appears in prior_asof, 0 else. One value per row of df."""
    if prior_asof.empty or FARMER_COL not in prior_asof.columns:
        return np.zeros(len(df), dtype=np.float64)
    farmers_in_prior = set(prior_asof[FARMER_COL].dropna().unique())
    return df[FARMER_COL].isin(farmers_in_prior).astype(np.float64).values


def _safe_topic_count(series: pd.Series) -> np.ndarray:
    return series.apply(lambda x: len(x) if isinstance(x, list) else 0).values


def _safe_has_topic(series: pd.Series) -> np.ndarray:
    return series.apply(lambda x: 1.0 if isinstance(x, list) and len(x) > 0 else 0.0).values


def build_gating_features(
    df: pd.DataFrame,
    prior_asof: pd.DataFrame,
    fit_encoders: dict[str, dict[str, int]] | None = None,
) -> tuple[np.ndarray, dict[str, dict[str, int]]]:
    """Build gating feature matrix: has_prior_history, has_topic_trained_on, topic_count, encoded categoricals.

    Returns (X, encoders). X shape (n, n_features). encoders used for val transform if fit_encoders was None.
    """
    n = len(df)
    parts: list[np.ndarray] = []

    has_prior = compute_has_prior_history(df, prior_asof)
    parts.append(has_prior.reshape(-1, 1))

    if TOPICS_COL in df.columns:
        topics = df[TOPICS_COL]
        parts.append(_safe_has_topic(topics).reshape(-1, 1))
        parts.append(_safe_topic_count(topics).reshape(-1, 1))
    else:
        parts.append(np.zeros((n, 1)))
        parts.append(np.zeros((n, 1)))

    encoders = fit_encoders if fit_encoders is not None else {}
    for col in GATING_CATEGORICAL:
        if col not in df.columns:
            parts.append(np.zeros((n, 1)))
            continue
        values = df[col].fillna("__NA__").astype(str)
        if fit_encoders is None or col not in encoders:
            uniq = sorted(values.unique())
            encoders[col] = {v: i for i, v in enumerate(uniq)}
        enc = encoders[col]
        codes = values.map(enc).fillna(-1).astype(np.int32).values.reshape(-1, 1)
        parts.append(codes)

    X = np.hstack(parts).astype(np.float64)
    return X, encoders


def cohort_id(has_prior_history: np.ndarray, has_topic_trained_on: np.ndarray) -> np.ndarray:
    """Cohort index 0..3: (has_prior, has_topic) -> 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)."""
    return (has_prior_history.astype(int) * 2 + has_topic_trained_on.astype(int)).astype(int)
