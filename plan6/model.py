"""
Plan 6 — Hazard-style multi-task neural network.

This model learns three interval hazards:
    h1: event in [0, 7]
    h2: event in (7, 90]
    h3: event in (90, 120]

Cumulative targets are reconstructed as:
    P(<=7)   = h1
    P(<=90)  = 1 - (1-h1)(1-h2)
    P(<=120) = 1 - (1-h1)(1-h2)(1-h3)

This guarantees monotonic probabilities:
    P7 <= P90 <= P120
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader as TorchDataLoader
    from torch.utils.data import TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from plan4.config import (
    AUC_PARAMS,
    EARLY_STOPPING_ROUNDS as P4_ESR,
    NUM_BOOST_ROUND as P4_NBR,
)
from plan6.config import (
    BATCH_SIZE,
    CALIBRATION,
    CV_STRATEGY,
    CV_TIME_CUTOFF,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    FEATURE_GROUPS,
    HIDDEN_DIMS,
    INTERVAL_LOSS_WEIGHTS,
    LEARNING_RATE,
    MAX_POS_WEIGHT,
    MIN_DELTA,
    N_CV_FOLDS,
    SEED,
    WEIGHT_DECAY,
)
from shared.calibration import Calibrator, CalibrationMethod, get_default_calibration_method
from shared.constants import ID_COL, PROB_CLIP_MAX, PROB_CLIP_MIN, TARGET_COLS
from shared.data_loader import DataLoader
from shared.evaluation import calculate_weighted_score
from shared.feature_engineering import FeatureEngineer
from shared.submission import SubmissionGenerator
from shared.validation import CVResult, Validator

logger = logging.getLogger(__name__)


class _HazardNet(nn.Module):  # type: ignore[misc]
    """Simple MLP that outputs three interval hazard logits."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = HIDDEN_DIMS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, hidden),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden),
                    nn.Dropout(dropout),
                ]
            )
            prev = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        logits = self.head(h)  # (n, 3)
        hazard = torch.sigmoid(logits)

        h1 = hazard[:, 0]
        h2 = hazard[:, 1]
        h3 = hazard[:, 2]

        p7 = h1
        p90 = 1.0 - (1.0 - h1) * (1.0 - h2)
        p120 = 1.0 - (1.0 - h1) * (1.0 - h2) * (1.0 - h3)

        cumulative = torch.stack([p7, p90, p120], dim=1)
        return logits, cumulative


class HazardNeuralModel:
    """Plan 6 runner."""

    def __init__(
        self,
        n_folds: int = N_CV_FOLDS,
        cv_strategy: str = CV_STRATEGY,
        cv_time_cutoff: str = CV_TIME_CUTOFF,
        calibration: str = CALIBRATION,
    ) -> None:
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for Plan 6. Install torch to run this plan."
            )

        self.n_folds = n_folds
        self.cv_strategy = cv_strategy
        self.cv_time_cutoff = cv_time_cutoff
        self.calibration = calibration.strip().lower()
        self.plan_name = "plan6_hazard"

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._seed_all(SEED)

    def run(self, submission_filename: str | None = None) -> Path:
        """Execute Plan 6 end-to-end."""
        logger.info("=" * 60)
        logger.info("Running %s", self.plan_name)
        logger.info("Device: %s", self._device)
        logger.info("=" * 60)

        loader = DataLoader()
        train_df, test_df, prior_df, sample_sub = loader.load_all()

        fe = FeatureEngineer(prior_df)
        fe.set_feature_groups(FEATURE_GROUPS)
        fe.fit(train_df)
        X_train = fe.transform(train_df)
        X_test = fe.transform(test_df)
        y_train = train_df[TARGET_COLS].copy()

        validator = Validator(n_splits=self.n_folds)
        cv_result = CVResult()

        splits = list(
            validator.cv_splits(
                train_df,
                y_train[TARGET_COLS[0]],
                strategy=self.cv_strategy,
                cutoff=self.cv_time_cutoff,
            )
        )
        n_folds_eff = len(splits)
        logger.info("Using %d CV fold(s)", n_folds_eff)

        oof_preds = np.zeros((len(train_df), 3), dtype=float)
        oof_mask = np.zeros(len(train_df), dtype=bool)
        test_preds_sum = np.zeros((len(test_df), 3), dtype=float)

        for fold_idx, (tr_idx, va_idx) in enumerate(splits):
            logger.info(
                "Fold %d | train=%d val=%d", fold_idx, len(tr_idx), len(va_idx)
            )
            X_tr = X_train.iloc[tr_idx]
            X_va = X_train.iloc[va_idx]
            y_tr = y_train.iloc[tr_idx]
            y_va = y_train.iloc[va_idx]

            net, scaler = self._train_single_fold(X_tr, y_tr, X_va, y_va)

            val_pred = self._predict_cumulative(net, scaler, X_va)
            test_pred = self._predict_cumulative(net, scaler, X_test)

            oof_preds[va_idx] = val_pred
            oof_mask[va_idx] = True
            test_preds_sum += test_pred / n_folds_eff

            for i, target in enumerate(TARGET_COLS):
                fold_result = validator.evaluate(
                    y_va[target].values,
                    val_pred[:, i],
                    target_name=target,
                    fold=fold_idx,
                )
                cv_result.add(fold_result)

            fold_weighted = self._compute_weighted_score(
                y_va[TARGET_COLS].values, val_pred
            )
            logger.info("  Fold %d weighted score: %.6f", fold_idx, fold_weighted)

        test_preds = np.clip(test_preds_sum, PROB_CLIP_MIN, PROB_CLIP_MAX)

        cal_method = get_default_calibration_method()
        if cal_method != CalibrationMethod.NONE:
            logger.info("Applying %s calibration on OOF predictions", cal_method.name)
            for i, target in enumerate(TARGET_COLS):
                cal = Calibrator(cal_method)
                cal.fit(y_train[target].values[oof_mask], oof_preds[oof_mask, i])
                test_preds[:, i] = cal.transform(test_preds[:, i])

        # Keep hierarchy after calibration.
        p7, p90, p120 = Calibrator.enforce_hierarchy(
            test_preds[:, 0], test_preds[:, 1], test_preds[:, 2]
        )
        test_preds[:, 0] = p7
        test_preds[:, 1] = p90
        test_preds[:, 2] = p120

        # OOF summary on covered validation rows.
        oof_weighted = self._compute_weighted_score(
            y_train[TARGET_COLS].values[oof_mask], oof_preds[oof_mask]
        )
        logger.info("OOF weighted score: %.6f", oof_weighted)
        logger.info("\n%s CV Summary:\n%s", self.plan_name, cv_result.summary().to_string())

        preds_map = {target: test_preds[:, i] for i, target in enumerate(TARGET_COLS)}

        # AUC columns: LightGBM trained for AUC on same features/splits
        logger.info("Training LightGBM AUC models for AUC columns …")
        preds_auc_map: dict[str, np.ndarray] = {}
        for i, target in enumerate(TARGET_COLS):
            y = y_train[target]
            test_auc_sum = np.zeros(len(X_test))
            for fold_idx, (tr_idx, va_idx) in enumerate(splits):
                train_ds = lgb.Dataset(
                    X_train.iloc[tr_idx], label=y.iloc[tr_idx], free_raw_data=False
                )
                val_ds = lgb.Dataset(
                    X_train.iloc[va_idx], label=y.iloc[va_idx], free_raw_data=False
                )
                booster = lgb.train(
                    AUC_PARAMS,
                    train_ds,
                    num_boost_round=P4_NBR,
                    valid_sets=[train_ds, val_ds],
                    valid_names=["train", "valid"],
                    callbacks=[lgb.early_stopping(P4_ESR, verbose=False)],
                )
                test_auc_sum += (
                    booster.predict(X_test, num_iteration=booster.best_iteration)
                    / n_folds_eff
                )
            preds_auc_map[target] = np.clip(test_auc_sum, PROB_CLIP_MIN, PROB_CLIP_MAX)
        p7_a, p90_a, p120_a = Calibrator.enforce_hierarchy(
            preds_auc_map[TARGET_COLS[0]],
            preds_auc_map[TARGET_COLS[1]],
            preds_auc_map[TARGET_COLS[2]],
        )
        preds_auc_map[TARGET_COLS[0]] = p7_a
        preds_auc_map[TARGET_COLS[1]] = p90_a
        preds_auc_map[TARGET_COLS[2]] = p120_a

        filename = submission_filename or f"{self.plan_name}_submission.csv"
        gen = SubmissionGenerator(sample_sub)
        path = gen.generate(
            test_ids=test_df[ID_COL],
            predictions=preds_map,
            predictions_auc=preds_auc_map,
            filename=filename,
        )

        logger.info("✓ %s complete → %s", self.plan_name, path)
        return path

    # ── Training / inference helpers ─────────────────────────────────

    def _train_single_fold(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
    ) -> tuple[_HazardNet, StandardScaler]:
        scaler = StandardScaler()
        X_tr_np = scaler.fit_transform(X_train).astype(np.float32)
        X_va_np = scaler.transform(X_val).astype(np.float32)

        y_tr_interval = self._to_interval_targets(y_train[TARGET_COLS].values).astype(
            np.float32
        )
        y_va_cum = y_val[TARGET_COLS].values.astype(np.float32)

        ds = TensorDataset(
            torch.from_numpy(X_tr_np), torch.from_numpy(y_tr_interval)
        )
        dl = TorchDataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

        net = _HazardNet(input_dim=X_tr_np.shape[1]).to(self._device)
        optim = torch.optim.AdamW(
            net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        pos_weight = self._compute_pos_weight(y_tr_interval)
        criterion = nn.BCEWithLogitsLoss(
            reduction="none", pos_weight=pos_weight.to(self._device)
        )
        interval_weights = torch.tensor(
            INTERVAL_LOSS_WEIGHTS, dtype=torch.float32, device=self._device
        ).view(1, -1)

        best_score = -np.inf
        best_state: Optional[dict] = None
        stale_epochs = 0

        for epoch in range(EPOCHS):
            net.train()
            losses = []
            for xb, yb in dl:
                xb = xb.to(self._device)
                yb = yb.to(self._device)

                optim.zero_grad()
                logits, _ = net(xb)
                loss_matrix = criterion(logits, yb)
                loss = (loss_matrix * interval_weights).mean()
                loss.backward()
                optim.step()
                losses.append(float(loss.detach().cpu().item()))

            # Validation monitor uses cumulative targets and competition score.
            val_pred = self._predict_cumulative_from_array(net, X_va_np)
            val_score = self._compute_weighted_score(y_va_cum, val_pred)

            if val_score > best_score + MIN_DELTA:
                best_score = val_score
                best_state = {
                    k: v.detach().cpu().clone() for k, v in net.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1

            if epoch % 10 == 0 or epoch == EPOCHS - 1:
                logger.debug(
                    "    epoch=%d train_loss=%.6f val_weighted=%.6f",
                    epoch,
                    float(np.mean(losses)) if losses else 0.0,
                    val_score,
                )

            if stale_epochs >= EARLY_STOPPING_PATIENCE:
                break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return net, scaler

    def _predict_cumulative(
        self, net: _HazardNet, scaler: StandardScaler, X: pd.DataFrame
    ) -> np.ndarray:
        X_np = scaler.transform(X).astype(np.float32)
        return self._predict_cumulative_from_array(net, X_np)

    def _predict_cumulative_from_array(
        self, net: _HazardNet, X_np: np.ndarray
    ) -> np.ndarray:
        net.eval()
        out: list[np.ndarray] = []
        bs = 2048
        with torch.no_grad():
            for start in range(0, len(X_np), bs):
                xb = torch.from_numpy(X_np[start : start + bs]).to(self._device)
                _, cumulative = net(xb)
                out.append(cumulative.detach().cpu().numpy())
        pred = np.vstack(out)
        return np.clip(pred, PROB_CLIP_MIN, PROB_CLIP_MAX)

    # ── Metric / target helpers ───────────────────────────────────────

    @staticmethod
    def _to_interval_targets(y_cumulative: np.ndarray) -> np.ndarray:
        """Convert cumulative labels [<=7, <=90, <=120] to interval labels."""
        y7 = y_cumulative[:, 0].astype(int)
        y90 = y_cumulative[:, 1].astype(int)
        y120 = y_cumulative[:, 2].astype(int)

        z1 = y7
        z2 = ((y90 == 1) & (y7 == 0)).astype(int)
        z3 = ((y120 == 1) & (y90 == 0)).astype(int)
        return np.column_stack([z1, z2, z3])

    def _compute_weighted_score(
        self, y_true_cum: np.ndarray, y_pred_cum: np.ndarray
    ) -> float:
        ll = []
        auc = []
        for i in range(3):
            y_true = y_true_cum[:, i]
            y_pred = np.clip(y_pred_cum[:, i], PROB_CLIP_MIN, PROB_CLIP_MAX)
            ll.append(log_loss(y_true, y_pred))
            if len(np.unique(y_true)) < 2:
                auc.append(0.5)
            else:
                auc.append(roc_auc_score(y_true, y_pred))

        return calculate_weighted_score(
            auc[0], ll[0], auc[1], ll[1], auc[2], ll[2]
        )

    @staticmethod
    def _compute_pos_weight(y_interval: np.ndarray) -> torch.Tensor:
        """Compute capped positive-class weights for the three interval heads."""
        weights = []
        for i in range(y_interval.shape[1]):
            pos = float(np.sum(y_interval[:, i] == 1))
            neg = float(np.sum(y_interval[:, i] == 0))
            if pos <= 0:
                w = 1.0
            else:
                w = min(MAX_POS_WEIGHT, max(1.0, neg / pos))
            weights.append(w)
        return torch.tensor(weights, dtype=torch.float32)

    @staticmethod
    def _seed_all(seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
