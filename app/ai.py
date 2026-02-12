# app/ai.py
# This file handles the "AI advisor" part of my system.
# I use a logistic regression model (saved by model_train.py) to predict
# probability of approval. Then I convert that into:
#   - recommendation: Approve / Reject
#   - confidence: a number between 0 and 1
#   - explanation: a few short reasons (based on feature contributions)
#
# The explanation is not perfect like a full SHAP analysis, but it is
# interpretable enough for my thesis prototype and user study.

from __future__ import annotations

import os
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import joblib


# Where I expect the trained model to be saved.
# (model_train.py should create this file.)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

# Cache the loaded model so we don't reload it for every request.
_MODEL = None


# -----------------------
# Helper functions
# -----------------------

def _load_model():
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Run model_train.py first to train and save the model."
            )
        _MODEL = joblib.load(MODEL_PATH)
    return _MODEL


def _prettify_feature_name(raw_name: str) -> str:
    """
    Turn sklearn-ish feature names into something more human readable.

    Examples:
      - 'num__annual_income' -> 'annual income'
      - 'cat__employment_status_Permanent' -> 'employment status: Permanent'
    """
    name = raw_name

    # Remove transformer prefixes like "num__" or "cat__"
    if "__" in name:
        name = name.split("__", 1)[1]

    # If it's a one-hot encoded category, sklearn often uses "col_value"
    # We'll convert that to "col: value"
    if "_" in name:
        parts = name.split("_", 1)
        # Heuristic: if first part looks like a column name and second like a category
        col, rest = parts[0], parts[1]
        # If the original raw name had "cat__", it's likely categorical
        if "cat__" in raw_name:
            return f"{col.replace('_', ' ')}: {rest.replace('_', ' ')}"

    return name.replace("_", " ")


def _get_pipeline_parts(model) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Try to extract (preprocess, classifier) from a sklearn Pipeline.

    I keep this defensive because different pipelines can be saved.
    """
    preprocess = None
    clf = None

    # Typical: Pipeline(steps=[('preprocess', ...), ('clf', LogisticRegression(...))])
    if hasattr(model, "named_steps"):
        preprocess = model.named_steps.get("preprocess")
        clf = model.named_steps.get("clf") or model.named_steps.get("classifier")
    else:
        # If it's not a pipeline, maybe it's directly a classifier
        clf = model

    return preprocess, clf


def _compute_contributions(model, X_dict: Dict[str, Any]) -> List[Tuple[str, float]]:
    """
    Approximate feature contributions for a logistic regression model:
      contribution_i = x_i * coef_i

    This only works cleanly if:
      - model is a Pipeline with a preprocess step that outputs a feature matrix
      - final estimator has .coef_

    If anything fails, we return an empty list (still OK for the UI).
    """
    try:
        preprocess, clf = _get_pipeline_parts(model)
        if clf is None or not hasattr(clf, "coef_"):
            return []

        # Put a single row into a "mini dataframe-ish" format using dict -> list
        # (sklearn can accept list-of-dicts for some setups, but DataFrame is safer.)
        import pandas as pd  # local import to keep file lightweight
        X_df = pd.DataFrame([X_dict])

        # Transform into model feature space
        if preprocess is not None:
            X_trans = preprocess.transform(X_df)
            # Get feature names after preprocessing (including one-hot)
            if hasattr(preprocess, "get_feature_names_out"):
                feat_names = preprocess.get_feature_names_out()
            else:
                feat_names = np.array([f"f{i}" for i in range(X_trans.shape[1])])
        else:
            # No preprocess step: use raw values
            X_trans = X_df.values
            feat_names = np.array(list(X_df.columns))

        # coef_ shape is (1, n_features) for binary logistic regression
        coefs = clf.coef_.ravel()

        # X_trans may be sparse
        if hasattr(X_trans, "toarray"):
            x_vals = X_trans.toarray().ravel()
        else:
            x_vals = np.array(X_trans).ravel()

        contributions = x_vals * coefs

        # Pair each feature with its contribution
        pairs = list(zip(feat_names.tolist(), contributions.tolist()))

        # Remove zero-ish contributions (common for one-hot categories that are off)
        pairs = [(f, c) for (f, c) in pairs if abs(c) > 1e-9]

        # Sort by absolute impact (biggest influences first)
        pairs.sort(key=lambda fc: abs(fc[1]), reverse=True)
        return pairs

    except Exception:
        # For a student prototype: if explanation breaks, I prefer the app to still run.
        return []


def _build_explanation(contribs: List[Tuple[str, float]], recommendation: str, max_items: int = 3) -> List[str]:
    """
    Convert contributions into short, human-readable reasons.

    If recommendation is Approve:
      we list the top *positive* contributions as "helping approval"
    If recommendation is Reject:
      we list the top *negative* contributions as "increasing risk"

    If we don't have enough, we fall back to "top absolute" influences.
    """
    reasons: List[str] = []

    if not contribs:
        return reasons

    if recommendation == "Approve":
        filtered = [(f, c) for (f, c) in contribs if c > 0]
        label = "supported approval"
    else:
        filtered = [(f, c) for (f, c) in contribs if c < 0]
        label = "increased risk"

    # If we don't get enough in the direction we want, use absolute top impacts.
    if len(filtered) < max_items:
        filtered = contribs

    for f, c in filtered[:max_items]:
        nice = _prettify_feature_name(f)

        # Very simple text templates (clear enough for a user study)
        if recommendation == "Approve":
            if c >= 0:
                reasons.append(f"{nice} {label}")
            else:
                reasons.append(f"{nice} slightly {label} less")
        else:
            if c <= 0:
                reasons.append(f"{nice} {label}")
            else:
                reasons.append(f"{nice} slightly reduced risk")

    return reasons[:max_items]


# -----------------------
# Main function used by Flask
# -----------------------

def get_ai_advice(features: Dict[str, Any], approval_threshold: float = 0.55) -> Dict[str, Any]:
    """
    Returns AI "advisor" output for the UI.

    Output format:
      {
        "recommendation": "Approve" | "Reject",
        "confidence": 0.00..1.00,
        "prob_approve": 0.00..1.00,
        "explanation": [ "...", "...", "..." ]
      }
    """
    model = _load_model()

    # Predict probability of approval (class 1).
    # I assume the model was trained with target 0/1 where 1 means approved.
    import pandas as pd  # local import
    X_df = pd.DataFrame([features])

    # Some models return probs in order of classes (clf.classes_)
    probas = model.predict_proba(X_df)[0]
    classes = getattr(model, "classes_", None)

    if classes is None and hasattr(model, "named_steps"):
        # In a pipeline, classes_ is usually on the final estimator
        clf = model.named_steps.get("clf") or model.named_steps.get("classifier")
        classes = getattr(clf, "classes_", None)

    if classes is not None:
        # Find index for class "1"
        idx_approve = int(np.where(np.array(classes) == 1)[0][0])
        prob_approve = float(probas[idx_approve])
    else:
        # Fallback: assume second column is class 1
        prob_approve = float(probas[1])

    # Recommendation logic
    recommendation = "Approve" if prob_approve >= approval_threshold else "Reject"

    # Confidence: how sure is the AI about its recommended class?
    # If recommending Approve -> confidence = prob_approve
    # If recommending Reject  -> confidence = 1 - prob_approve
    confidence = prob_approve if recommendation == "Approve" else (1.0 - prob_approve)

    # Explanations (approx. feature contributions)
    contribs = _compute_contributions(model, features)
    explanation = _build_explanation(contribs, recommendation=recommendation, max_items=3)

    return {
        "recommendation": recommendation,
        "confidence": round(confidence, 3),
        "prob_approve": round(prob_approve, 3),
        "explanation": explanation,
    }