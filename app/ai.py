# app/ai.py
# AI advisor: logistic regression + simple explanation.

from __future__ import annotations

import os
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import joblib

from app.config import MODEL_PATH

# Cache the loaded model so we don't reload it for every request.
_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Make sure model.joblib is included in the deployed repo."
            )
        _MODEL = joblib.load(MODEL_PATH)
    return _MODEL


def _prettify_feature_name(raw_name: str) -> str:
    name = raw_name

    # Remove transformer prefixes like "num__" or "cat__"
    if "__" in name:
        name = name.split("__", 1)[1]

    # If it's one-hot encoded, sklearn often uses "col_value"
    if "_" in name:
        parts = name.split("_", 1)
        col, rest = parts[0], parts[1]
        if "cat__" in raw_name:
            return f"{col.replace('_', ' ')}: {rest.replace('_', ' ')}"

    return name.replace("_", " ")


def _get_pipeline_parts(model) -> Tuple[Optional[Any], Optional[Any]]:
    preprocess = None
    clf = None

    if hasattr(model, "named_steps"):
        preprocess = model.named_steps.get("preprocess")
        clf = model.named_steps.get("clf") or model.named_steps.get("classifier")
    else:
        clf = model

    return preprocess, clf


def _compute_contributions(model, X_dict: Dict[str, Any]) -> List[Tuple[str, float]]:
    """
    Approximate contributions for logistic regression: x_i * coef_i.
    Returns [] if not possible.

    NOTE: For a Pipeline + ColumnTransformer, we must pass a pandas DataFrame (2D).
    """
    try:
        preprocess, clf = _get_pipeline_parts(model)
        if clf is None or not hasattr(clf, "coef_"):
            return []

        if preprocess is None:
            # Without a preprocess step we can't reliably align coef_ with raw dict features.
            return []

        import pandas as pd  # local import
        X_df = pd.DataFrame([X_dict])

        X_trans = preprocess.transform(X_df)

        if hasattr(preprocess, "get_feature_names_out"):
            feat_names = preprocess.get_feature_names_out()
        else:
            feat_names = np.array([f"f{i}" for i in range(X_trans.shape[1])])

        coefs = clf.coef_.ravel()

        # X_trans may be sparse
        if hasattr(X_trans, "toarray"):
            x_vals = X_trans.toarray().ravel()
        else:
            x_vals = np.array(X_trans).ravel()

        contributions = x_vals * coefs
        pairs = list(zip(list(feat_names), contributions.tolist()))

        # remove near-zero
        pairs = [(f, c) for (f, c) in pairs if abs(c) > 1e-9]
        pairs.sort(key=lambda fc: abs(fc[1]), reverse=True)
        return pairs

    except Exception:
        return []


def _build_explanation(contribs: List[Tuple[str, float]], recommendation: str, max_items: int = 3) -> List[str]:
    reasons: List[str] = []
    if not contribs:
        return reasons

    if recommendation == "Approve":
        filtered = [(f, c) for (f, c) in contribs if c > 0]
        label = "supported approval"
    else:
        filtered = [(f, c) for (f, c) in contribs if c < 0]
        label = "increased risk"

    # If not enough in the desired direction, use top absolute impacts
    if len(filtered) < max_items:
        filtered = contribs

    for f, c in filtered[:max_items]:
        nice = _prettify_feature_name(str(f))

        if recommendation == "Approve":
            reasons.append(f"{nice} {label}" if c >= 0 else f"{nice} slightly reduced approval support")
        else:
            reasons.append(f"{nice} {label}" if c <= 0 else f"{nice} slightly reduced risk")

    return reasons[:max_items]


def get_ai_advice(features: Dict[str, Any], approval_threshold: float = 0.55) -> Dict[str, Any]:
    """
    Returns:
      {
        "recommendation": "Approve" | "Reject",
        "confidence": 0..1,
        "prob_approve": 0..1,
        "explanation": [str, str, str]
      }

    NOTE: For a Pipeline + ColumnTransformer, we must pass a pandas DataFrame (2D).
    """
    model = _load_model()

    import pandas as pd  # local import
    X_df = pd.DataFrame([features])

    probas = model.predict_proba(X_df)[0]
    classes = getattr(model, "classes_", None)

    if classes is None and hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf") or model.named_steps.get("classifier")
        classes = getattr(clf, "classes_", None)

    if classes is not None:
        idx_approve = int(np.where(np.array(classes) == 1)[0][0])
        prob_approve = float(probas[idx_approve])
    else:
        # fallback: assume second column is class 1
        prob_approve = float(probas[1])

    recommendation = "Approve" if prob_approve >= approval_threshold else "Reject"
    confidence = prob_approve if recommendation == "Approve" else (1.0 - prob_approve)

    contribs = _compute_contributions(model, features)
    explanation = _build_explanation(contribs, recommendation=recommendation, max_items=3)

    return {
        "recommendation": recommendation,
        "confidence": round(confidence, 3),
        "prob_approve": round(prob_approve, 3),
        "explanation": explanation,
    }
