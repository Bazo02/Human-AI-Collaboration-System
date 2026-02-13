# app/data_prep.py
# Cleans the dataset and prepares a smaller set of cases for the user study.

from __future__ import annotations

import os
from typing import Tuple

import pandas as pd

from app.config import (
    DATA_PATH,
    CASES_FOR_STUDY_PATH,
    TARGET_COL,
    DROP_COLS_FOR_UI,
)

# Sets the size and mix of the study pool
STUDY_SET_SIZE = 120     # total cases in the study pool
BORDERLINE_SHARE = 0.30  # share of borderline-ish (harder) cases


def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Does basic cleaning:
    - keeps the target as 0/1 int
    - removes duplicate rows
    - fills missing values (simple)
    - adds case_id if missing
    """
    df = df.copy()

    # Removes exact duplicates
    df = df.drop_duplicates()

    # Converts target to int 0/1
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Adds case_id for UI/logging
    if "case_id" not in df.columns:
        df.insert(0, "case_id", range(1, len(df) + 1))

    # Fills missing values: numeric -> median, categorical -> "Unknown"
    for col in df.columns:
        if col == TARGET_COL:
            continue

        if df[col].dtype.kind in "biufc":
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        else:
            if df[col].isna().any():
                df[col] = df[col].fillna("Unknown")

    return df


def _heuristic_risk_score(df: pd.DataFrame) -> pd.Series:
    """
    Creates a simple risk score used only to pick study cases.
    Higher score means higher risk (more likely reject).
    """
    # Returns a column if present, otherwise returns zeros
    def col_or_zero(name: str) -> pd.Series:
        return df[name] if name in df.columns else pd.Series([0] * len(df))

    # Pulls common fields (missing columns become zeros)
    credit_score = col_or_zero("credit_score")
    income = col_or_zero("annual_income")
    loan_amount = col_or_zero("loan_amount")
    existing_loans = col_or_zero("existing_loans_count")

    # Transforms fields into rough risk components
    credit_risk = (700 - credit_score) / 200.0
    income_risk = (500000 - income) / 500000.0
    loan_risk = loan_amount / 500000.0
    existing_risk = existing_loans / 5.0

    # Clamps values so extremes do not dominate
    credit_risk = credit_risk.clip(-2, 2)
    income_risk = income_risk.clip(-2, 2)
    loan_risk = loan_risk.clip(0, 3)
    existing_risk = existing_risk.clip(0, 3)

    score = 0.50 * credit_risk + 0.25 * income_risk + 0.20 * loan_risk + 0.05 * existing_risk
    return score


def _select_study_cases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects cases for the user study pool.
    Keeps a mix of approved/rejected and includes some borderline cases.
    """
    df = df.copy()

    # Adds a difficulty-ish score used for selection
    df["risk_score"] = _heuristic_risk_score(df)

    # Splits by target label
    df_approve = df[df[TARGET_COL] == 1].copy()
    df_reject = df[df[TARGET_COL] == 0].copy()

    # Sets target counts (handles imbalance by taking what is available)
    half = STUDY_SET_SIZE // 2
    n_approve = min(half, len(df_approve))
    n_reject = min(STUDY_SET_SIZE - n_approve, len(df_reject))

    # Sets how many borderline cases to include
    n_borderline_total = int(STUDY_SET_SIZE * BORDERLINE_SHARE)
    n_borderline_each = n_borderline_total // 2

    # Picks approved borderline and easy cases
    approve_borderline = df_approve.sort_values("risk_score", ascending=False).head(n_borderline_each)
    approve_easy = df_approve.sort_values("risk_score", ascending=True).head(max(0, n_approve - len(approve_borderline)))

    # Picks rejected borderline and easy cases
    reject_borderline = df_reject.sort_values("risk_score", ascending=True).head(n_borderline_each)
    reject_easy = df_reject.sort_values("risk_score", ascending=False).head(max(0, n_reject - len(reject_borderline)))

    selected = pd.concat([approve_borderline, approve_easy, reject_borderline, reject_easy], axis=0)

    # Fills remaining slots if the dataset is imbalanced
    if len(selected) < STUDY_SET_SIZE:
        missing = STUDY_SET_SIZE - len(selected)
        remaining = df.drop(index=selected.index, errors="ignore")
        if missing > 0 and len(remaining) > 0:
            fill = remaining.sample(n=min(missing, len(remaining)), random_state=42)
            selected = pd.concat([selected, fill], axis=0)

    # Shuffles the final pool for variety
    selected = selected.sample(frac=1.0, random_state=123).reset_index(drop=True)

    # Drops the helper column
    selected = selected.drop(columns=["risk_score"], errors="ignore")

    return selected


def _drop_sensitive_and_unused_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns that are not needed in the UI.
    Keeps the target label for later evaluation.
    """
    df = df.copy()

    # Drops configured columns when they exist
    for col in DROP_COLS_FOR_UI:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def main() -> None:
    # Checks that the input dataset exists
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find dataset at: {DATA_PATH}")

    # Loads the raw dataset
    df = pd.read_csv(DATA_PATH)

    # Cleans data and removes unused fields
    df = _basic_clean(df)
    df = _drop_sensitive_and_unused_cols(df)

    # Selects a stable pool of study cases
    study_df = _select_study_cases(df)

    # Saves the output file
    os.makedirs(os.path.dirname(CASES_FOR_STUDY_PATH), exist_ok=True)
    study_df.to_csv(CASES_FOR_STUDY_PATH, index=False)

    print("Done.")
    print(f"Saved study cases to: {CASES_FOR_STUDY_PATH}")
    print(f"Rows: {len(study_df)}")
    print(f"Approve rate: {study_df[TARGET_COL].mean():.3f}")


if __name__ == "__main__":
    main()
