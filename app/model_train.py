# app/model_train.py
# This script trains the logistic regression model that my AI advisor uses.
#
# Why logistic regression?
# - It is a real ML model (looks "advanced enough" for a master thesis prototype)
# - It outputs probabilities, which I use as "confidence"
# - It is still interpretable (feature weights)
#
# How to run (in VS Code terminal):
#   python app/model_train.py
#
# Output:
#   app/model.joblib

from __future__ import annotations

import os
from typing import List, Tuple

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from app.config import (
    DATA_PATH,
    CASES_FOR_STUDY_PATH,
    MODEL_PATH,
    TARGET_COL,
)

# These are columns I do NOT want to include in the model.
# I exclude gender/marital_status so my thesis doesn't become a fairness paper.
EXCLUDE_COLS = [
    "applicant_id",
    "case_id",
    "gender",
    "marital_status",
]


def _load_training_data() -> pd.DataFrame:
    """
    Load dataset for model training.

    I prefer to train on the full dataset (loanapproval.csv) for stability.
    But if only the study dataset exists, I still allow training from that.
    """
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    if os.path.exists(CASES_FOR_STUDY_PATH):
        return pd.read_csv(CASES_FOR_STUDY_PATH)
    raise FileNotFoundError("Could not find a dataset to train on.")


def _split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical columns automatically.
    Then we can build a preprocessing pipeline.
    """
    feature_cols = [c for c in df.columns if c != TARGET_COL and c not in EXCLUDE_COLS]

    numeric_cols = [c for c in feature_cols if df[c].dtype.kind in "biufc"]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    return numeric_cols, categorical_cols


def main() -> None:
    df = _load_training_data().copy()

    # Basic cleaning to avoid training issues.
    # (My dataset is synthetic/clean already, but I still do this as a safety net.)
    df = df.drop_duplicates()

    # Ensure target is int 0/1
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Handle missing values simply (transparent approach)
    for col in df.columns:
        if col == TARGET_COL:
            continue
        if df[col].dtype.kind in "biufc":
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    # Build feature matrix X and target y
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Drop excluded columns if they exist
    for col in EXCLUDE_COLS:
        if col in X.columns:
            X = X.drop(columns=[col])

    numeric_cols, categorical_cols = _split_columns(df)

    # But numeric_cols / categorical_cols must match X after dropping columns
    numeric_cols = [c for c in numeric_cols if c in X.columns]
    categorical_cols = [c for c in categorical_cols if c in X.columns]

    print("Training columns:")
    print("  Numeric:", numeric_cols)
    print("  Categorical:", categorical_cols)

    # Preprocessing:
    # - scale numeric features
    # - one-hot encode categorical features
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # Logistic regression classifier
    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("clf", clf),
        ]
    )

    # Split for reporting model performance (not the main thesis focus, but good to show)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y if len(y.unique()) > 1 else None,
    )

    model.fit(X_train, y_train)

    # Evaluate quickly
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\nModel evaluation (just to sanity check):")
    print(f"Accuracy: {acc:.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nDone.")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()