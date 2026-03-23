"""
Train Random Forest (baseline) and XGBoost (primary) on the NBA game-outcome
feature matrix with 10-fold stratified cross-validation.

Persists to ``artifacts/``:
  - model_rf.joblib
  - model_xgb.joblib
  - feature_columns.json
  - metrics.json   (per-fold and mean+-std for accuracy, F1, AUC)

Usage
-----
  python -m ml.train                          # standalone
  python manage.py train_models               # via Django management command
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from .preprocess import main as preprocess_main
from .features.build import build_feature_matrix

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
N_FOLDS = 10


# ── Hyperparameters ─────────────────────────────────────────────────────────

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}

XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "eval_metric": "logloss",
    "n_jobs": -1,
}


# ── Evaluation helpers ──────────────────────────────────────────────────────

def _evaluate_fold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
    }


def _summarise(fold_metrics: list[dict]) -> dict:
    out: dict = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        out[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "per_fold": vals,
        }
    return out


# ── Main training routine ──────────────────────────────────────────────────

def train(
    games_path: Path | None = None,
    player_game_path: Path | None = None,
    stint_path: Path | None = None,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> dict:
    """
    Full training pipeline.

    Returns a dict with ``rf`` and ``xgb`` summaries.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Preprocess (mirrors report's preprocess.main — always writes games_clean.csv)
    print("[1/4] Preprocessing games …")
    games_df = preprocess_main(raw_path=games_path)

    # Optional player / stint data for chemistry
    player_game_df, stint_df = None, None
    try:
        from .data_loader import load_player_game_stats, load_lineups

        if player_game_path:
            player_game_df = pd.read_csv(player_game_path, low_memory=False)
        else:
            player_game_df = load_player_game_stats()
    except Exception:
        player_game_df = None

    try:
        if stint_path:
            stint_df = pd.read_csv(stint_path, low_memory=False)
        else:
            stint_df = load_lineups()
    except Exception:
        stint_df = None

    # 2. Build features
    print("[2/4] Building feature matrix …")
    full_df, feature_cols = build_feature_matrix(
        games_df, player_game_df=player_game_df, stint_df=stint_df
    )

    X = full_df[feature_cols].values
    y = full_df["home_win"].values

    print(f"       Samples: {len(y)}  |  Features: {len(feature_cols)}")
    print(f"       Class balance: {y.mean():.3f} home-win rate")

    # Persist feature column order (needed at inference time)
    with open(artifacts_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # 3. Cross-validation
    print(f"[3/4] {N_FOLDS}-fold stratified CV …")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    rf_folds: list[dict] = []
    xgb_folds: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Random Forest
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_tr, y_tr)
        rf_pred = rf.predict(X_val)
        rf_prob = rf.predict_proba(X_val)[:, 1]
        rf_folds.append(_evaluate_fold(y_val, rf_pred, rf_prob))

        # XGBoost
        xgb = XGBClassifier(**XGB_PARAMS)
        xgb.fit(X_tr, y_tr, verbose=False)
        xgb_pred = xgb.predict(X_val)
        xgb_prob = xgb.predict_proba(X_val)[:, 1]
        xgb_folds.append(_evaluate_fold(y_val, xgb_pred, xgb_prob))

        print(
            f"  Fold {fold:>2}/{N_FOLDS}  "
            f"RF acc={rf_folds[-1]['accuracy']:.4f}  "
            f"XGB acc={xgb_folds[-1]['accuracy']:.4f}"
        )

    rf_summary = _summarise(rf_folds)
    xgb_summary = _summarise(xgb_folds)

    print("\n── Random Forest ──")
    for k in ("accuracy", "f1", "auc"):
        print(f"  {k}: {rf_summary[k]['mean']:.4f} ± {rf_summary[k]['std']:.4f}")
    print("── XGBoost ──")
    for k in ("accuracy", "f1", "auc"):
        print(f"  {k}: {xgb_summary[k]['mean']:.4f} ± {xgb_summary[k]['std']:.4f}")

    # 4. Retrain on full data and persist
    print("\n[4/4] Retraining on full dataset and saving artifacts …")
    rf_full = RandomForestClassifier(**RF_PARAMS)
    rf_full.fit(X, y)
    joblib.dump(rf_full, artifacts_dir / "model_rf.joblib")

    xgb_full = XGBClassifier(**XGB_PARAMS)
    xgb_full.fit(X, y, verbose=False)
    joblib.dump(xgb_full, artifacts_dir / "model_xgb.joblib")

    metrics = {
        "random_forest": rf_summary,
        "xgboost": xgb_summary,
        "n_folds": N_FOLDS,
        "n_samples": int(len(y)),
        "n_features": len(feature_cols),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Done. Artifacts saved to:", artifacts_dir)
    return metrics


if __name__ == "__main__":
    train()
