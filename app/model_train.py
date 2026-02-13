# app/model_train.py
# Trains the logistic regression model used by the AI advisor.

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

# Lists columns that should not be used as model features
EXCLUDE_COLS = [
    "applicant_id",
    "case_id",
    "gender",
    "marital_status",
]


def _load_training_data() -> pd.DataFrame:
    # Loads training data from the main dataset, or falls back to the study dataset
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    if os.path.exists(CASES_FOR_STUDY_PATH):
        return pd.read_csv(CASES_FOR_STUDY_PATH)
    raise FileNotFoundError("Could not find a dataset to train on.")


def _split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    # Splits feature columns into numeric and categorical lists
    feature_cols = [c for c in df.columns if c != TARGET_COL and c not in EXCLUDE_COLS]

    numeric_cols = [c for c in feature_cols if df[c].dtype.kind in "biufc"]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    return numeric_cols, categorical_cols


def main() -> None:
    df = _load_training_data().copy()

    # Removes duplicate rows
    df = df.drop_duplicates()

    # Converts target to int 0/1
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Fills missing values in a simple and transparent way
    for col in df.columns:
        if col == TARGET_COL:
            continue
        if df[col].dtype.kind in "biufc":
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    # Builds feature matrix X and target vector y
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Removes excluded columns if present
    for col in EXCLUDE_COLS:
        if col in X.columns:
            X = X.drop(columns=[col])

    numeric_cols, categorical_cols = _split_columns(df)

    # Keeps only columns that still exist in X
    numeric_cols = [c for c in numeric_cols if c in X.columns]
    categorical_cols = [c for c in categorical_cols if c in X.columns]

    print("Training columns:")
    print("  Numeric:", numeric_cols)
    print("  Categorical:", categorical_cols)

    # Builds preprocessing steps for numeric scaling and categorical encoding
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # Defines the logistic regression classifier
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

    # Splits data for a quick performance check
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y if len(y.unique()) > 1 else None,
    )

    model.fit(X_train, y_train)

    # Evaluates model on the test split
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\nModel evaluation (just to sanity check):")
    print(f"Accuracy: {acc:.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    # Saves trained model to disk
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nDone.")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
