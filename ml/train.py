"""
Train Random Forest (baseline) and XGBoost (primary) on the NBA game-outcome
feature matrix with 10-fold stratified cross-validation.

Additionally trains a stacking ensemble that combines RF + XGBoost
out-of-fold predictions via logistic regression for improved accuracy.

Persists to ``artifacts/``:
  - model_rf.joblib
  - model_xgb.joblib
  - model_stack.joblib   (logistic regression meta-learner)
  - feature_columns.json
  - metrics.json

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from .preprocess import main as preprocess_main
from .data_loader import load_team_statistics, load_player_statistics
from .features.build import build_feature_matrix

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
N_FOLDS = 10


# ── Hyperparameters ─────────────────────────────────────────────────────────

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "min_samples_split": 20,
    "min_samples_leaf": 12,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
    "class_weight": "balanced_subsample",
}

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.02,
    "subsample": 0.55,
    "colsample_bytree": 0.4,
    "colsample_bylevel": 0.4,
    "reg_alpha": 3.0,
    "reg_lambda": 10.0,
    "min_child_weight": 20,
    "gamma": 1.5,
    "random_state": 42,
    "eval_metric": "logloss",
    "n_jobs": -1,
    "early_stopping_rounds": 50,
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
    team_stats_path: Path | None = None,
    player_stats_path: Path | None = None,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> dict:
    """Full training pipeline. Returns a dict with metric summaries."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Preprocess games
    print("[1/5] Preprocessing games …")
    games_df = preprocess_main(raw_path=games_path)

    # 2. Load supplementary data
    print("[2/5] Loading team statistics and player statistics …")
    try:
        if team_stats_path:
            team_stats_df = pd.read_csv(team_stats_path, low_memory=False)
        else:
            team_stats_df = load_team_statistics()
        print(f"       TeamStatistics: {len(team_stats_df)} rows")
    except Exception as exc:
        print(f"       TeamStatistics not available: {exc}")
        team_stats_df = None

    try:
        if player_stats_path:
            player_stats_df = pd.read_csv(player_stats_path, low_memory=False)
        else:
            player_stats_df = load_player_statistics()
        print(f"       PlayerStatistics: {len(player_stats_df)} rows")
    except Exception as exc:
        print(f"       PlayerStatistics not available: {exc}")
        player_stats_df = None

    # 3. Build features
    print("[3/5] Building feature matrix …")
    full_df, feature_cols = build_feature_matrix(
        games_df,
        team_stats_df=team_stats_df,
        player_stats_df=player_stats_df,
    )

    X = full_df[feature_cols].values
    y = full_df["home_win"].values

    print(f"       Samples: {len(y)}  |  Features: {len(feature_cols)}")
    print(f"       Class balance: {y.mean():.3f} home-win rate")

    with open(artifacts_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # 4. Cross-validation with out-of-fold predictions for stacking
    print(f"[4/5] {N_FOLDS}-fold stratified CV …")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    rf_folds: list[dict] = []
    xgb_folds: list[dict] = []
    stack_folds: list[dict] = []

    oof_rf = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Random Forest
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_tr, y_tr)
        rf_prob = rf.predict_proba(X_val)[:, 1]
        rf_pred = rf.predict(X_val)
        rf_folds.append(_evaluate_fold(y_val, rf_pred, rf_prob))
        oof_rf[val_idx] = rf_prob

        # XGBoost with early stopping
        xgb = XGBClassifier(**XGB_PARAMS)
        xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        xgb_prob = xgb.predict_proba(X_val)[:, 1]
        xgb_pred = xgb.predict(X_val)
        xgb_folds.append(_evaluate_fold(y_val, xgb_pred, xgb_prob))
        oof_xgb[val_idx] = xgb_prob

        # Stacking: blend RF + XGB probabilities
        stack_prob = 0.4 * rf_prob + 0.6 * xgb_prob
        stack_pred = (stack_prob >= 0.5).astype(int)
        stack_folds.append(_evaluate_fold(y_val, stack_pred, stack_prob))

        print(
            f"  Fold {fold:>2}/{N_FOLDS}  "
            f"RF={rf_folds[-1]['accuracy']:.4f}  "
            f"XGB={xgb_folds[-1]['accuracy']:.4f}  "
            f"Stack={stack_folds[-1]['accuracy']:.4f}"
        )

    rf_summary = _summarise(rf_folds)
    xgb_summary = _summarise(xgb_folds)
    stack_summary = _summarise(stack_folds)

    print("\n── Random Forest ──")
    for k in ("accuracy", "f1", "auc"):
        print(f"  {k}: {rf_summary[k]['mean']:.4f} ± {rf_summary[k]['std']:.4f}")
    print("── XGBoost ──")
    for k in ("accuracy", "f1", "auc"):
        print(f"  {k}: {xgb_summary[k]['mean']:.4f} ± {xgb_summary[k]['std']:.4f}")
    print("── Stacking Ensemble (0.4*RF + 0.6*XGB) ──")
    for k in ("accuracy", "f1", "auc"):
        print(f"  {k}: {stack_summary[k]['mean']:.4f} ± {stack_summary[k]['std']:.4f}")

    # 5. Train stacking meta-learner on out-of-fold predictions
    print("\n[5/5] Training stacking meta-learner …")
    stack_X = np.column_stack([oof_rf, oof_xgb])
    meta = LogisticRegression(C=1.0, random_state=42)
    meta.fit(stack_X, y)
    meta_prob = meta.predict_proba(stack_X)[:, 1]
    meta_pred = meta.predict(stack_X)
    print(
        f"  Meta-learner train acc: {accuracy_score(y, meta_pred):.4f}  "
        f"AUC: {roc_auc_score(y, meta_prob):.4f}"
    )

    # Retrain base models on full data and persist
    print("\nRetraining base models on full dataset …")
    rf_full = RandomForestClassifier(**RF_PARAMS)
    rf_full.fit(X, y)
    joblib.dump(rf_full, artifacts_dir / "model_rf.joblib")

    xgb_final_params = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}
    xgb_full = XGBClassifier(**xgb_final_params)
    xgb_full.fit(X, y, verbose=False)
    joblib.dump(xgb_full, artifacts_dir / "model_xgb.joblib")

    joblib.dump(meta, artifacts_dir / "model_stack.joblib")

    metrics = {
        "random_forest": rf_summary,
        "xgboost": xgb_summary,
        "stacking_ensemble": stack_summary,
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
