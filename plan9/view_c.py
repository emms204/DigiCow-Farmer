"""
View C: Either Node2Vec (cached artifact view_c_n2v_v1) or PPMI-SVD + interaction TE.
- n2v: load/save OOF from plan9/artifacts/view_c_n2v_{version}/; run Node2Vec only when precomputing.
- ppmi: PPMI-SVD farmer embeddings + interaction TE (trainer×topic, group×topic, ward×topic) + LGBM.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.decomposition import TruncatedSVD

from shared.constants import (
    FARMER_COL,
    TRAINER_COL,
    GROUP_COL,
    TARGET_COLS,
    TOPICS_COL,
)

from plan9.config import (
    VIEW_C_BACKEND,
    VIEW_C_N2V_ARTIFACT_VERSION,
    VIEW_C_PPMI_DIM,
    VIEW_C_PPMI_SHIFT,
    VIEW_C_TE_MIN_SUPPORT,
    VIEW_C_TE_SHRINK,
    PROB_CLIP,
    SEED,
    VIEW_C_LGBM_PARAMS,
)
from plan9.config import EARLY_STOPPING_ROUNDS, NUM_BOOST_ROUND

logger = logging.getLogger(__name__)


def _get_view_c_n2v_artifact_dir() -> Path:
    """Artifact dir for View C Node2Vec cache: plan9/artifacts/view_c_n2v_{version}/"""
    base = Path(__file__).resolve().parent
    return base / "artifacts" / f"view_c_n2v_{VIEW_C_N2V_ARTIFACT_VERSION}"


def _load_view_c_n2v_oof(artifact_dir: Path, train_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Load cached OOF from oof.npz. Returns (oof_p07, oof_p90, oof_p120, oof_mask) or None if missing/invalid."""
    oof_path = artifact_dir / "oof.npz"
    if not oof_path.exists():
        return None
    try:
        data = np.load(oof_path, allow_pickle=False)
        cached_len = int(data["train_len"])
        if cached_len != train_len:
            logger.warning("View C n2v cache train_len %s != %s, ignoring cache", cached_len, train_len)
            return None
        oof_p07 = data["oof_p07"]
        oof_p90 = data["oof_p90"]
        oof_p120 = data["oof_p120"]
        oof_mask = data["oof_mask"]
        return oof_p07, oof_p90, oof_p120, oof_mask
    except Exception as e:
        logger.warning("View C n2v cache load failed: %s", e)
        return None


def _save_view_c_n2v_oof(
    artifact_dir: Path,
    oof_p07: np.ndarray,
    oof_p90: np.ndarray,
    oof_p120: np.ndarray,
    oof_mask: np.ndarray,
    train_len: int,
) -> None:
    """Save OOF to artifact_dir/oof.npz."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        artifact_dir / "oof.npz",
        oof_p07=oof_p07,
        oof_p90=oof_p90,
        oof_p120=oof_p120,
        oof_mask=oof_mask,
        train_len=np.int64(train_len),
    )
    logger.info("View C n2v OOF cached to %s", artifact_dir)


def _save_view_c_n2v_test(
    artifact_dir: Path,
    test_p07: np.ndarray,
    test_p90: np.ndarray,
    test_p120: np.ndarray,
    test_len: int,
) -> None:
    """Save test predictions to artifact_dir/test.npz."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        artifact_dir / "test.npz",
        test_p07=test_p07,
        test_p90=test_p90,
        test_p120=test_p120,
        test_len=np.int64(test_len),
    )
    logger.info("View C n2v test preds cached to %s", artifact_dir)


def load_view_c_n2v_test(artifact_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load cached View C Node2Vec test predictions. Returns (test_p07, test_p90, test_p120) or None if missing/invalid."""
    if artifact_dir is None:
        artifact_dir = _get_view_c_n2v_artifact_dir()
    test_path = artifact_dir / "test.npz"
    if not test_path.exists():
        return None
    try:
        data = np.load(test_path, allow_pickle=False)
        return data["test_p07"], data["test_p90"], data["test_p120"]
    except Exception as e:
        logger.warning("View C n2v test cache load failed: %s", e)
        return None


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


def _topic_tokens(row: pd.Series) -> list[str]:
    """Normalized topic tokens for this row."""
    out = []
    for t in _topics_list(row):
        tok = "topic_" + t.replace(" ", "_")[:50]
        out.append(tok)
    return out


def _entity_keys(row: pd.Series) -> dict[str, str]:
    return {
        "trainer": "trainer_" + _safe_str(row[TRAINER_COL]),
        "group": "group_" + _safe_str(row.get(GROUP_COL, "")),
        "ward": "ward_" + _safe_str(row.get("ward", "")),
    }


# ---- Farmer-entity co-occurrence → log/TF-IDF → PPMI → SVD ----

def build_farmer_entity_cooccurrence(
    train_slice: pd.DataFrame, val_slice: pd.DataFrame
) -> tuple[lil_matrix, list[str], list[str]]:
    """Build sparse farmer x entity count matrix from train+val. Returns (M, farmer_list, entity_list)."""
    df = pd.concat([train_slice, val_slice], ignore_index=True)
    farmer_to_idx: dict[str, int] = {}
    entity_to_idx: dict[str, int] = {}
    # Collect farmers and entities, and (farmer, entity) counts
    rows_data: dict[int, dict[int, float]] = {}  # farmer_idx -> { entity_idx: count }

    for _, row in df.iterrows():
        fkey = _safe_str(row[FARMER_COL])
        if fkey not in farmer_to_idx:
            farmer_to_idx[fkey] = len(farmer_to_idx)
        fi = farmer_to_idx[fkey]
        if fi not in rows_data:
            rows_data[fi] = {}
        # Entities: trainer, group, county, subcounty, ward, topic tokens
        trainer = "trainer_" + _safe_str(row[TRAINER_COL])
        group = "group_" + _safe_str(row.get(GROUP_COL, ""))
        county = "county_" + _safe_str(row.get("county", ""))
        subcounty = "subcounty_" + _safe_str(row.get("subcounty", ""))
        ward = "ward_" + _safe_str(row.get("ward", ""))
        for ent in [trainer, group, county, subcounty, ward] + _topic_tokens(row):
            if ent not in entity_to_idx:
                entity_to_idx[ent] = len(entity_to_idx)
            ei = entity_to_idx[ent]
            rows_data[fi][ei] = rows_data[fi].get(ei, 0) + 1.0

    farmers = sorted(farmer_to_idx.keys(), key=lambda x: farmer_to_idx[x])
    entities = sorted(entity_to_idx.keys(), key=lambda x: entity_to_idx[x])
    n_f, n_e = len(farmers), len(entities)
    M = lil_matrix((n_f, n_e), dtype=np.float64)
    for fi, counts in rows_data.items():
        for ei, c in counts.items():
            M[fi, ei] = c
    return M, farmers, entities


def cooccurrence_to_ppmi(
    M: lil_matrix, shift: float = 1.0
) -> csr_matrix:
    """Convert count matrix to PPMI (positive PMI). shift used in numerator/denominator to avoid log(0)."""
    M = M.tocsr().astype(np.float64)
    total = M.sum()
    if total <= 0:
        return M
    n_f, n_e = M.shape
    row_sum = np.array(M.sum(axis=1)).ravel()
    col_sum = np.array(M.sum(axis=0)).ravel()
    # Shifted PMI: (M_ij + shift) * total_eff / ((row_i + shift*n_e) * (col_j + shift*n_f))
    row_sum = row_sum + shift * n_e
    col_sum = col_sum + shift * n_f
    row_sum[row_sum <= 0] = 1.0
    col_sum[col_sum <= 0] = 1.0
    rows, cols = M.nonzero()
    data = M.data
    ppmi_data = np.zeros_like(data)
    total_eff = total + shift * len(data)
    for k in range(len(data)):
        i, j = rows[k], cols[k]
        num = (data[k] + shift) * total_eff
        denom = row_sum[i] * col_sum[j]
        if denom > 0:
            pmi = np.log(num / denom + 1e-12)
            val = max(0.0, float(pmi))
            ppmi_data[k] = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
    return csr_matrix((ppmi_data, (rows, cols)), shape=M.shape)


def farmer_embeddings_ppmi_svd(
    train_slice: pd.DataFrame,
    val_slice: pd.DataFrame,
    dimensions: int,
    seed: int,
    ppmi_shift: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Build farmer-entity co-occurrence, weight (log(1+count)), convert to PPMI, TruncatedSVD.
    Return (X_tr_emb, X_va_emb, farmer_to_idx) where X_tr/va are (n_rows, dim) by mapping each row to its farmer.
    farmer_to_idx is over the fold's farmers; missing farmers get zero embedding.
    """
    M, farmers, _ = build_farmer_entity_cooccurrence(train_slice, val_slice)
    # Optional: log weighting before PPMI. Common: use raw counts for PPMI, or log(1+c). We'll do PPMI on counts with shift.
    P = cooccurrence_to_ppmi(M, shift=ppmi_shift)
    n_f, n_e = P.shape
    n_components = min(dimensions, n_f, n_e)
    if n_components < 1:
        farmer_to_idx = {f: i for i, f in enumerate(farmers)}
        n_train, n_val = len(train_slice), len(val_slice)
        dim = max(1, dimensions)
        return (
            np.zeros((n_train, dim), dtype=np.float64),
            np.zeros((n_val, dim), dtype=np.float64),
            farmer_to_idx,
        )
    svd = TruncatedSVD(n_components=n_components, random_state=seed, n_iter=10)
    U = svd.fit_transform(P)  # (n_farmers, n_components)
    if n_components < dimensions:
        U = np.hstack([U, np.zeros((U.shape[0], dimensions - n_components), dtype=np.float64)])
    farmer_to_idx = {f: i for i, f in enumerate(farmers)}
    # Map train/val rows to farmer embedding
    train_farmers = train_slice[FARMER_COL].map(lambda x: _safe_str(x)).values
    val_farmers = val_slice[FARMER_COL].map(lambda x: _safe_str(x)).values
    X_tr = np.array([U[farmer_to_idx[f]] if f in farmer_to_idx else np.zeros(U.shape[1]) for f in train_farmers], dtype=np.float64)
    X_va = np.array([U[farmer_to_idx[f]] if f in farmer_to_idx else np.zeros(U.shape[1]) for f in val_farmers], dtype=np.float64)
    return X_tr, X_va, farmer_to_idx


# ---- Interaction TE: trainer×topic, group×topic, ward×topic (min-support + Bayesian shrinkage) ----

def compute_interaction_te(
    train_slice: pd.DataFrame,
    min_support: int,
    shrink: float,
) -> tuple[dict, dict, dict, dict]:
    """
    From train_slice only. For (trainer, topic), (group, topic), (ward, topic) compute
    count and smoothed target rate with Bayesian shrinkage. Returns four:
    te_trainer, te_group, te_ward, global_means. Each te[*][(key, topic)] = {"count", "rate_<col>"}.
    Below min_support we use global_mean at lookup.
    """
    global_sum = {c: 0.0 for c in TARGET_COLS}
    n_global = 0
    for _, row in train_slice.iterrows():
        for c in TARGET_COLS:
            global_sum[c] += row[c]
        n_global += 1
    global_means = {c: global_sum[c] / n_global if n_global else 0.5 for c in TARGET_COLS}

    # Build (entity_key, topic) -> count, sum per target
    te_trainer: dict[tuple[str, str], dict] = {}
    te_group: dict[tuple[str, str], dict] = {}
    te_ward: dict[tuple[str, str], dict] = {}

    for _, row in train_slice.iterrows():
        keys = _entity_keys(row)
        topics = _topic_tokens(row)
        for t in topics:
            for (te_dict, key) in [(te_trainer, keys["trainer"]), (te_group, keys["group"]), (te_ward, keys["ward"])]:
                pair = (key, t)
                if pair not in te_dict:
                    te_dict[pair] = {"count": 0, **{"sum_" + c: 0.0 for c in TARGET_COLS}}
                te_dict[pair]["count"] += 1
                for c in TARGET_COLS:
                    te_dict[pair]["sum_" + c] += row[c]

    def smooth_te(te_dict: dict) -> dict:
        out = {}
        for pair, v in te_dict.items():
            count = v["count"]
            out[pair] = {
                "count": count,
                **{
                    "rate_" + c: (v["sum_" + c] + shrink * global_means[c]) / (count + shrink)
                    for c in TARGET_COLS
                },
            }
        return out

    return smooth_te(te_trainer), smooth_te(te_group), smooth_te(te_ward), global_means


def build_interaction_te_features(
    df: pd.DataFrame,
    te_trainer: dict,
    te_group: dict,
    te_ward: dict,
    global_means: dict[str, float],
    min_support: int,
) -> np.ndarray:
    """
    Per row: for trainer×topic, group×topic, ward×topic compute mean over this row's topics
    of smoothed rate (if count >= min_support else global_mean). One feature per (interaction_type, target).
    So 3 types * 3 targets = 9 features. We'll use mean over topics; if no topics, use global_mean.
    """
    n_rows = len(df)
    n_feat = 3 * len(TARGET_COLS)  # trainer_topic_mean_p07, ..., ward_topic_mean_p07, ...
    X = np.zeros((n_rows, n_feat), dtype=np.float64)
    te_list = [te_trainer, te_group, te_ward]

    for i in range(n_rows):
        row = df.iloc[i]
        keys = _entity_keys(row)
        topic_list = _topic_tokens(row)
        key_list = [keys["trainer"], keys["group"], keys["ward"]]
        off = 0
        for te_dict, key in zip(te_list, key_list):
            for col in TARGET_COLS:
                if not topic_list:
                    X[i, off] = global_means[col]
                else:
                    vals = []
                    for t in topic_list:
                        pair = (key, t)
                        st = te_dict.get(pair)
                        if st is not None and st["count"] >= min_support:
                            vals.append(st["rate_" + col])
                        else:
                            vals.append(global_means[col])
                    X[i, off] = np.mean(vals)
                off += 1

    return X


# ---- Combine and OOF ----

def get_view_c_oof(
    train_df: pd.DataFrame,
    splits: list[dict[str, Any]],
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (p07, p90, p120, oof_mask). If backend n2v: load from cache if present, else run Node2Vec and cache. Else: PPMI-SVD + interaction TE."""
    n_rows = len(train_df)

    if VIEW_C_BACKEND == "n2v" and VIEW_C_N2V_ARTIFACT_VERSION:
        artifact_dir = _get_view_c_n2v_artifact_dir()
        cached = _load_view_c_n2v_oof(artifact_dir, n_rows)
        if cached is not None:
            logger.info("View C (graph): using cached n2v OOF from %s", artifact_dir)
            return cached
        from plan9.view_c_n2v import run_view_c_node2v_oof
        logger.info("View C (graph): cache miss, running Node2Vec (slow) ...")
        oof_p07, oof_p90, oof_p120, oof_mask = run_view_c_node2v_oof(train_df, splits, seed=seed)
        _save_view_c_n2v_oof(artifact_dir, oof_p07, oof_p90, oof_p120, oof_mask, n_rows)
        return oof_p07, oof_p90, oof_p120, oof_mask

    # PPMI-SVD + interaction TE path
    import lightgbm as lgb

    oof_p07 = np.zeros(n_rows)
    oof_p90 = np.zeros(n_rows)
    oof_p120 = np.zeros(n_rows)
    oof_mask = np.zeros(n_rows, dtype=bool)

    for split in splits:
        train_pos = split["train_pos"]
        val_pos = split["val_pos"]
        train_slice = train_df.iloc[train_pos]
        val_slice = train_df.iloc[val_pos]

        # PPMI-SVD farmer embeddings
        X_tr_emb, X_va_emb, _ = farmer_embeddings_ppmi_svd(
            train_slice, val_slice, VIEW_C_PPMI_DIM, seed, VIEW_C_PPMI_SHIFT
        )

        # Interaction TE (from train only)
        te_trainer, te_group, te_ward, global_means = compute_interaction_te(
            train_slice, VIEW_C_TE_MIN_SUPPORT, VIEW_C_TE_SHRINK
        )
        X_tr_te = build_interaction_te_features(
            train_slice, te_trainer, te_group, te_ward, global_means, VIEW_C_TE_MIN_SUPPORT
        )
        X_va_te = build_interaction_te_features(
            val_slice, te_trainer, te_group, te_ward, global_means, VIEW_C_TE_MIN_SUPPORT
        )

        X_tr = np.hstack([X_tr_emb, X_tr_te])
        X_va = np.hstack([X_va_emb, X_va_te])

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
