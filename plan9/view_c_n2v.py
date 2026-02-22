"""
View C Node2Vec path: heterogeneous graph + Node2Vec embeddings + LGBM per target.
Used when VIEW_C_BACKEND == "n2v". Run once (or via precompute script), cache OOF and test preds.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import networkx as nx

from shared.constants import (
    FARMER_COL,
    TRAINER_COL,
    GROUP_COL,
    TARGET_COLS,
    TOPICS_COL,
)

from plan9.config import (
    VIEW_C_N2V_DIMENSIONS,
    VIEW_C_N2V_NUM_WALKS,
    VIEW_C_N2V_P,
    VIEW_C_N2V_Q,
    VIEW_C_N2V_WALK_LENGTH,
    VIEW_C_N2V_WORKERS,
    PROB_CLIP,
    SEED,
    VIEW_C_LGBM_PARAMS,
)
from plan9.config import EARLY_STOPPING_ROUNDS, NUM_BOOST_ROUND

logger = logging.getLogger(__name__)


def _safe_str(x: Any) -> str:
    if pd.isna(x):
        return "__NA__"
    s = str(x).strip()
    return s if s else "__NA__"


def _topics_list(row: pd.Series) -> list[str]:
    x = row[TOPICS_COL]
    if isinstance(x, list):
        return [_safe_str(t) for t in x if _safe_str(t) and _safe_str(t) != "__NA__"]
    return []


def build_graph_from_slices(train_slice: pd.DataFrame, val_slice: pd.DataFrame):  # -> nx.Graph
    """Build undirected graph: nodes = farmer, trainer, group, county, subcounty, ward, topic tokens. Edges from each row."""
    import networkx as nx

    G = nx.Graph()
    for _, row in pd.concat([train_slice, val_slice], ignore_index=True).iterrows():
        farmer = "farmer_" + _safe_str(row[FARMER_COL])
        trainer = "trainer_" + _safe_str(row[TRAINER_COL])
        group = "group_" + _safe_str(row.get(GROUP_COL, ""))
        county = "county_" + _safe_str(row.get("county", ""))
        subcounty = "subcounty_" + _safe_str(row.get("subcounty", ""))
        ward = "ward_" + _safe_str(row.get("ward", ""))
        G.add_node(farmer)
        G.add_edge(farmer, trainer)
        G.add_edge(farmer, group)
        G.add_edge(farmer, county)
        G.add_edge(farmer, subcounty)
        G.add_edge(farmer, ward)
        for t in _topics_list(row):
            if t and t != "__NA__":
                tok = "topic_" + t.replace(" ", "_")[:50]
                G.add_node(tok)
                G.add_edge(farmer, tok)
    return G


def get_farmer_embeddings_node2vec(
    G: Any,  # nx.Graph
    farmer_names: np.ndarray,
    dimensions: int,
    seed: int,
) -> np.ndarray:
    """Run Node2Vec and return (n, dimensions) array of farmer embeddings. Missing farmers get zeros."""
    try:
        from node2vec import Node2Vec
    except ImportError:
        logger.warning("node2vec not installed; View C n2v will use zero embeddings. pip install node2vec")
        return np.zeros((len(farmer_names), dimensions), dtype=np.float64)

    n2v = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=VIEW_C_N2V_WALK_LENGTH,
        num_walks=VIEW_C_N2V_NUM_WALKS,
        p=VIEW_C_N2V_P,
        q=VIEW_C_N2V_Q,
        workers=VIEW_C_N2V_WORKERS,
        quiet=True,
        seed=seed,
    )
    model = n2v.fit()
    embs = []
    for name in farmer_names:
        key = "farmer_" + _safe_str(name)
        if key in model.wv:
            embs.append(model.wv[key])
        else:
            embs.append(np.zeros(dimensions, dtype=np.float64))
    return np.array(embs, dtype=np.float64)


def run_view_c_node2v_oof(
    train_df: pd.DataFrame,
    splits: list[dict[str, Any]],
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run Node2Vec View C: graph per fold, farmer embeddings, LGBM per target. Returns (oof_p07, oof_p90, oof_p120, oof_mask)."""
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

        G = build_graph_from_slices(train_slice, val_slice)
        train_farmers = train_slice[FARMER_COL].values
        val_farmers = val_slice[FARMER_COL].values
        X_tr = get_farmer_embeddings_node2vec(G, train_farmers, VIEW_C_N2V_DIMENSIONS, seed)
        X_va = get_farmer_embeddings_node2vec(G, val_farmers, VIEW_C_N2V_DIMENSIONS, seed)
        if X_tr.ndim == 1:
            X_tr = X_tr.reshape(-1, 1)
            X_va = X_va.reshape(-1, 1)

        for i, target in enumerate(TARGET_COLS):
            y_tr = train_slice[target].values
            y_va = val_slice[target].values
            train_set = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
            val_set = lgb.Dataset(X_va, label=y_va, free_raw_data=False)
            booster = lgb.train(
                params={**VIEW_C_LGBM_PARAMS, "random_state": seed},
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


def run_view_c_node2v_oof_and_test(
    train_df: pd.DataFrame,
    splits: list[dict[str, Any]],
    test_df: pd.DataFrame,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Node2Vec View C: OOF on train + test predictions (averaged over folds).
    Returns (oof_p07, oof_p90, oof_p120, oof_mask, test_p07, test_p90, test_p120).
    """
    import lightgbm as lgb

    n_rows = len(train_df)
    n_test = len(test_df)
    oof_p07 = np.zeros(n_rows)
    oof_p90 = np.zeros(n_rows)
    oof_p120 = np.zeros(n_rows)
    oof_mask = np.zeros(n_rows, dtype=bool)
    test_p07 = np.zeros(n_test)
    test_p90 = np.zeros(n_test)
    test_p120 = np.zeros(n_test)

    for fold_idx, split in enumerate(splits):
        train_pos = split["train_pos"]
        val_pos = split["val_pos"]
        train_slice = train_df.iloc[train_pos]
        val_slice = train_df.iloc[val_pos]

        G = build_graph_from_slices(train_slice, val_slice)
        train_farmers = train_slice[FARMER_COL].values
        val_farmers = val_slice[FARMER_COL].values
        test_farmers = test_df[FARMER_COL].values

        X_tr = get_farmer_embeddings_node2vec(G, train_farmers, VIEW_C_N2V_DIMENSIONS, seed)
        X_va = get_farmer_embeddings_node2vec(G, val_farmers, VIEW_C_N2V_DIMENSIONS, seed)
        X_te = get_farmer_embeddings_node2vec(G, test_farmers, VIEW_C_N2V_DIMENSIONS, seed)
        if X_tr.ndim == 1:
            X_tr = X_tr.reshape(-1, 1)
            X_va = X_va.reshape(-1, 1)
            X_te = X_te.reshape(-1, 1)

        t07, t90, t120 = np.zeros(n_test), np.zeros(n_test), np.zeros(n_test)
        for i, target in enumerate(TARGET_COLS):
            y_tr = train_slice[target].values
            y_va = val_slice[target].values
            train_set = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
            val_set = lgb.Dataset(X_va, label=y_va, free_raw_data=False)
            booster = lgb.train(
                params={**VIEW_C_LGBM_PARAMS, "random_state": seed},
                train_set=train_set,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[train_set, val_set],
                valid_names=["train", "valid"],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            pred_val = booster.predict(X_va, num_iteration=booster.best_iteration)
            pred_val = np.clip(pred_val, *PROB_CLIP)
            pred_te = booster.predict(X_te, num_iteration=booster.best_iteration)
            pred_te = np.clip(pred_te, *PROB_CLIP)
            if i == 0:
                oof_p07[val_pos] = pred_val
                t07 = pred_te
            elif i == 1:
                oof_p90[val_pos] = pred_val
                t90 = pred_te
            else:
                oof_p120[val_pos] = pred_val
                t120 = pred_te
        oof_mask[val_pos] = True
        test_p07 += t07
        test_p90 += t90
        test_p120 += t120

    n_folds = len(splits)
    test_p07 /= n_folds
    test_p90 /= n_folds
    test_p120 /= n_folds
    test_p07 = np.clip(test_p07, *PROB_CLIP)
    test_p90 = np.clip(test_p90, *PROB_CLIP)
    test_p120 = np.clip(test_p120, *PROB_CLIP)
    return oof_p07, oof_p90, oof_p120, oof_mask, test_p07, test_p90, test_p120
