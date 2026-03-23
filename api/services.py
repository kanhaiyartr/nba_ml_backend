"""
Singleton service that loads trained model artifacts once and provides
prediction + explanation helpers consumed by DRF views.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import joblib
import numpy as np

from django.conf import settings

_lock = threading.Lock()
_state: dict = {}


class ModelNotReady(Exception):
    pass


def _artifacts_dir() -> Path:
    return Path(settings.MODEL_ARTIFACTS_DIR)


def _ensure_loaded() -> dict:
    """Load models + metadata once (thread-safe)."""
    if _state:
        return _state

    with _lock:
        if _state:
            return _state

        adir = _artifacts_dir()
        fc_path = adir / "feature_columns.json"
        rf_path = adir / "model_rf.joblib"
        xgb_path = adir / "model_xgb.joblib"
        metrics_path = adir / "metrics.json"

        if not fc_path.exists():
            raise ModelNotReady(
                "Model artifacts not found. Run `python manage.py train_models` first."
            )

        with open(fc_path) as f:
            _state["feature_columns"] = json.load(f)

        if rf_path.exists():
            _state["rf"] = joblib.load(rf_path)
        if xgb_path.exists():
            _state["xgb"] = joblib.load(xgb_path)

        if metrics_path.exists():
            with open(metrics_path) as f:
                _state["metrics"] = json.load(f)
        else:
            _state["metrics"] = {}

    return _state


def get_feature_columns() -> list[str]:
    return _ensure_loaded()["feature_columns"]


def get_metrics() -> dict:
    return _ensure_loaded()["metrics"]


def predict(feature_values: list | np.ndarray, model_name: str = "xgboost") -> dict:
    """
    Run inference on a single feature row.

    Returns dict with ``win_probability``, ``predicted_label``, and optional
    ``shap_summary`` (top features).
    """
    state = _ensure_loaded()

    key = "xgb" if model_name == "xgboost" else "rf"
    model = state.get(key)
    if model is None:
        raise ModelNotReady(f"Model '{model_name}' not found in artifacts.")

    row = np.asarray(feature_values, dtype=np.float64).reshape(1, -1)

    prob = float(model.predict_proba(row)[0, 1])
    label = int(prob >= 0.5)

    # SHAP explanation (import lazily to keep startup fast)
    from ml.explain import explain_single

    explanation = explain_single(row.flatten(), model_name=model_name, top_k=8)

    return {
        "model": model_name,
        "home_win_probability": round(prob, 4),
        "predicted_winner": "home" if label == 1 else "away",
        "shap_summary": {
            "base_value": explanation["base_value"],
            "top_positive": explanation["top_positive"],
            "top_negative": explanation["top_negative"],
        },
    }
