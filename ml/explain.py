"""
SHAP-based model explanations.

Uses ``TreeExplainer`` (works for both Random Forest and XGBoost) to
decompose a single prediction into per-feature contributions.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import shap

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def _load_model(model_name: str = "xgboost"):
    filename = "model_xgb.joblib" if model_name == "xgboost" else "model_rf.joblib"
    return joblib.load(ARTIFACTS_DIR / filename)


def _load_feature_columns() -> list[str]:
    with open(ARTIFACTS_DIR / "feature_columns.json") as f:
        return json.load(f)


def explain_single(
    feature_row: np.ndarray | list,
    model_name: str = "xgboost",
    top_k: int = 10,
) -> dict:
    """
    Explain a single prediction row.

    Parameters
    ----------
    feature_row : 1-D array of feature values (same order as feature_columns.json)
    model_name  : ``"xgboost"`` or ``"random_forest"``
    top_k       : number of top positive + top negative contributors to return

    Returns
    -------
    dict with keys:
      - ``base_value``   : expected model output (log-odds or probability baseline)
      - ``shap_values``  : full dict of {feature: shap_value}
      - ``top_positive`` : list of (feature, value) pushing toward home win
      - ``top_negative`` : list of (feature, value) pushing toward home loss
    """
    model = _load_model(model_name)
    feature_cols = _load_feature_columns()

    row = np.asarray(feature_row).reshape(1, -1)

    explainer = shap.TreeExplainer(model)
    shap_out = explainer(row)

    sv = shap_out.values[0]
    # For binary classification shap may return (n_features, 2); take class-1
    if sv.ndim > 1:
        sv = sv[:, 1]

    base = shap_out.base_values
    if hasattr(base, "__len__"):
        base = base[0]
    if hasattr(base, "__len__"):
        base = base[1]  # class-1 base value
    base = float(base)

    feature_shap = {col: float(v) for col, v in zip(feature_cols, sv)}

    sorted_items = sorted(feature_shap.items(), key=lambda x: x[1])
    top_negative = [(k, v) for k, v in sorted_items[:top_k]]
    top_positive = [(k, v) for k, v in sorted_items[-top_k:]][::-1]

    return {
        "base_value": base,
        "shap_values": feature_shap,
        "top_positive": top_positive,
        "top_negative": top_negative,
    }
