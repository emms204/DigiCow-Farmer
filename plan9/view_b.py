"""
View B: Topic-text model (TF-IDF + SVD embeddings, then LGBM per target).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from shared.constants import TARGET_COLS, TOPICS_COL

from plan9.config import (
    PROB_CLIP,
    SEED,
    SVD_N_COMPONENTS,
    TFIDF_MAX_FEATURES,
    VIEW_B_LGBM_PARAMS,
)
from plan9.config import EARLY_STOPPING_ROUNDS, NUM_BOOST_ROUND


def _topics_to_text(series: pd.Series) -> pd.Series:
    """Convert topics_list (list of strings) to single space-joined string per row."""
    def join_topics(x):
        if isinstance(x, list):
            return " ".join(str(t).strip() for t in x if t)
        return ""
    return series.apply(join_topics)


def get_view_b_oof(
    train_df: pd.DataFrame,
    splits: list[dict[str, Any]],
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (p07, p90, p120, oof_mask) for View B = TF-IDF + SVD + LGBM per target."""
    import lightgbm as lgb

    n_rows = len(train_df)
    oof_p07 = np.zeros(n_rows)
    oof_p90 = np.zeros(n_rows)
    oof_p120 = np.zeros(n_rows)
    oof_mask = np.zeros(n_rows, dtype=bool)

    for split in splits:
        train_pos = split["train_pos"]
        val_pos = split["val_pos"]
        train_slice = train_df.iloc[train_pos]
        val_slice = train_df.iloc[val_pos]

        train_text = _topics_to_text(train_slice[TOPICS_COL])
        val_text = _topics_to_text(val_slice[TOPICS_COL])

        vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, token_pattern=r"(?u)\b\w+\b")
        X_tr_tfidf = vectorizer.fit_transform(train_text.fillna(""))
        X_va_tfidf = vectorizer.transform(val_text.fillna(""))

        n_comp = min(SVD_N_COMPONENTS, X_tr_tfidf.shape[1], X_tr_tfidf.shape[0], 100)
        if n_comp < 1:
            n_comp = 1
        svd = TruncatedSVD(n_components=n_comp, random_state=seed)
        X_tr = svd.fit_transform(X_tr_tfidf)
        X_va = svd.transform(X_va_tfidf)
        if X_tr.ndim == 1:
            X_tr = X_tr.reshape(-1, 1)
            X_va = X_va.reshape(-1, 1)

        for i, target in enumerate(TARGET_COLS):
            y_tr = train_slice[target].values
            y_va = val_slice[target].values
            train_set = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
            val_set = lgb.Dataset(X_va, label=y_va, free_raw_data=False)
            booster = lgb.train(
                params={**VIEW_B_LGBM_PARAMS, "random_state": seed},
                train_set=train_set,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[train_set, val_set],
                valid_names=["train", "valid"],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            pred = booster.predict(X_va, num_iteration=booster.best_iteration)
            pred = np.clip(pred, *PROB_CLIP)
            if i == 0:
                oof_p07[val_pos] = pred
            elif i == 1:
                oof_p90[val_pos] = pred
            else:
                oof_p120[val_pos] = pred
        oof_mask[val_pos] = True

    return oof_p07, oof_p90, oof_p120, oof_mask


def get_view_b_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    splits: list[dict[str, Any]],
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (test_p07, test_p90, test_p120) for View B: TF-IDF + SVD + LGBM, fold-averaged test preds."""
    import lightgbm as lgb

    n_test = len(test_df)
    test_p07 = np.zeros(n_test)
    test_p90 = np.zeros(n_test)
    test_p120 = np.zeros(n_test)

    for split in splits:
        train_pos = split["train_pos"]
        val_pos = split["val_pos"]
        train_slice = train_df.iloc[train_pos]
        val_slice = train_df.iloc[val_pos]

        train_text = _topics_to_text(train_slice[TOPICS_COL])
        val_text = _topics_to_text(val_slice[TOPICS_COL])
        test_text = _topics_to_text(test_df[TOPICS_COL])

        vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, token_pattern=r"(?u)\b\w+\b")
        X_tr_tfidf = vectorizer.fit_transform(train_text.fillna(""))
        X_va_tfidf = vectorizer.transform(val_text.fillna(""))
        X_te_tfidf = vectorizer.transform(test_text.fillna(""))

        n_comp = min(SVD_N_COMPONENTS, X_tr_tfidf.shape[1], X_tr_tfidf.shape[0], 100)
        if n_comp < 1:
            n_comp = 1
        svd = TruncatedSVD(n_components=n_comp, random_state=seed)
        X_tr = svd.fit_transform(X_tr_tfidf)
        X_va = svd.transform(X_va_tfidf)
        X_te = svd.transform(X_te_tfidf)
        if X_tr.ndim == 1:
            X_tr = X_tr.reshape(-1, 1)
            X_va = X_va.reshape(-1, 1)
            X_te = X_te.reshape(-1, 1)

        for i, target in enumerate(TARGET_COLS):
            y_tr = train_slice[target].values
            y_va = val_slice[target].values
            train_set = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
            val_set = lgb.Dataset(X_va, label=y_va, free_raw_data=False)
            booster = lgb.train(
                params={**VIEW_B_LGBM_PARAMS, "random_state": seed},
                train_set=train_set,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[train_set, val_set],
                valid_names=["train", "valid"],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            pred_te = booster.predict(X_te, num_iteration=booster.best_iteration)
            pred_te = np.clip(pred_te, *PROB_CLIP)
            if i == 0:
                test_p07 += pred_te
            elif i == 1:
                test_p90 += pred_te
            else:
                test_p120 += pred_te

    n_folds = len(splits)
    test_p07 = np.clip(test_p07 / n_folds, *PROB_CLIP)
    test_p90 = np.clip(test_p90 / n_folds, *PROB_CLIP)
    test_p120 = np.clip(test_p120 / n_folds, *PROB_CLIP)
    return test_p07, test_p90, test_p120
