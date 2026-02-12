# app/data_prep.py
# This script cleans the dataset and prepares a smaller set of cases
# that are suitable for the user study.
#
# Why I do this:
# - I want the user study to use a stable set of cases (reproducible)
# - I want to remove sensitive / unnecessary columns (gender etc.)
# - I want a balanced mix of approve/reject cases, plus some "borderline" cases
#
# How to run (in VS Code terminal):
#   python app/data_prep.py
#
# Output:
#   data/cases_for_study.csv

from __future__ import annotations

import os
from typing import Tuple

import pandas as pd

from config import (
    DATA_PATH,
    CASES_FOR_STUDY_PATH,
    TARGET_COL,
    DROP_COLS_FOR_UI,
)

# If you want to tune the size of the study set, change these.
STUDY_SET_SIZE = 120     # total cases for the user study pool
BORDERLINE_SHARE = 0.30  # fraction of cases that are borderline-ish


def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - ensure target is 0/1 int
    - remove duplicate rows
    - handle missing values (simple approach)
    - create a case_id if missing

    For a thesis prototype I keep imputation simple and transparent.
    """
    df = df.copy()

    # Drop exact duplicates (rare but safe)
    df = df.drop_duplicates()

    # Make sure target is int 0/1
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Create a case_id for UI/logging if not present
    if "case_id" not in df.columns:
        df.insert(0, "case_id", range(1, len(df) + 1))

    # Simple missing value handling:
    # numeric -> median, categorical -> "Unknown"
    for col in df.columns:
        if col == TARGET_COL:
            continue

        if df[col].dtype.kind in "biufc":  # numeric types
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        else:
            if df[col].isna().any():
                df[col] = df[col].fillna("Unknown")

    return df


def _heuristic_risk_score(df: pd.DataFrame) -> pd.Series:
    """
    A simple risk score used ONLY for selecting cases for the user study.
    This is NOT the "AI model". It's just a way to find borderline cases.

    Higher score => higher risk => more likely reject

    NOTE: I keep this heuristic based on common-sense fields.
    If your dataset columns differ, update the names here.
    """
    # Defensive: if a column isn't present, we treat it as neutral (0).
    def col_or_zero(name: str) -> pd.Series:
        return df[name] if name in df.columns else pd.Series([0] * len(df))

    # Normalize-ish pieces (roughly)
    credit_score = col_or_zero("credit_score")
    income = col_or_zero("annual_income")
    loan_amount = col_or_zero("loan_amount")
    existing_loans = col_or_zero("existing_loans_count")

    # I assume higher credit score is better, so risk goes down with credit score.
    # These formulas are not "bank accurate" - just enough to sort cases.
    credit_risk = (700 - credit_score) / 200.0  # ~ negative if credit_score > 700
    income_risk = (500000 - income) / 500000.0  # lower income => higher risk
    loan_risk = loan_amount / 500000.0          # bigger loan => higher risk
    existing_risk = existing_loans / 5.0

    # Clamp-ish (to avoid extreme values dominating)
    credit_risk = credit_risk.clip(-2, 2)
    income_risk = income_risk.clip(-2, 2)
    loan_risk = loan_risk.clip(0, 3)
    existing_risk = existing_risk.clip(0, 3)

    score = 0.50 * credit_risk + 0.25 * income_risk + 0.20 * loan_risk + 0.05 * existing_risk
    return score


def _select_study_cases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select cases for the user study pool.

    Strategy:
    - Keep a balanced number of approved/rejected.
    - Include some borderline-ish cases (harder decisions).
    """
    df = df.copy()

    # Create a heuristic score for "difficulty"
    df["risk_score"] = _heuristic_risk_score(df)

    # Balance: split by target
    df_approve = df[df[TARGET_COL] == 1].copy()
    df_reject = df[df[TARGET_COL] == 0].copy()

    # If dataset is imbalanced, we will just take as many as we can.
    half = STUDY_SET_SIZE // 2
    n_approve = min(half, len(df_approve))
    n_reject = min(STUDY_SET_SIZE - n_approve, len(df_reject))

    # Borderline selection:
    # For approved cases, borderline means higher risk_score (still approved)
    # For rejected cases, borderline means lower risk_score (still rejected)
    n_borderline_total = int(STUDY_SET_SIZE * BORDERLINE_SHARE)
    n_borderline_each = n_borderline_total // 2

    # Approved borderline: top risk scores (harder approvals)
    approve_borderline = df_approve.sort_values("risk_score", ascending=False).head(n_borderline_each)
    # Approved easy: low risk scores
    approve_easy = df_approve.sort_values("risk_score", ascending=True).head(max(0, n_approve - len(approve_borderline)))

    # Rejected borderline: low risk scores (harder rejections)
    reject_borderline = df_reject.sort_values("risk_score", ascending=True).head(n_borderline_each)
    # Rejected easy: high risk scores
    reject_easy = df_reject.sort_values("risk_score", ascending=False).head(max(0, n_reject - len(reject_borderline)))

    selected = pd.concat([approve_borderline, approve_easy, reject_borderline, reject_easy], axis=0)

    # If we didn't reach STUDY_SET_SIZE (because of imbalance), fill randomly.
    if len(selected) < STUDY_SET_SIZE:
        missing = STUDY_SET_SIZE - len(selected)
        remaining = df.drop(index=selected.index, errors="ignore")
        if missing > 0 and len(remaining) > 0:
            fill = remaining.sample(n=min(missing, len(remaining)), random_state=42)
            selected = pd.concat([selected, fill], axis=0)

    # Shuffle for variety
    selected = selected.sample(frac=1.0, random_state=123).reset_index(drop=True)

    # Clean up helper column
    selected = selected.drop(columns=["risk_score"], errors="ignore")

    return selected


def _drop_sensitive_and_unused_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that are not needed in the UI (and also not needed for my thesis).
    I keep the target label because I need it later for accuracy evaluation.
    """
    df = df.copy()
    # We only drop these if they exist.
    for col in DROP_COLS_FOR_UI:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def main() -> None:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find dataset at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    df = _basic_clean(df)
    df = _drop_sensitive_and_unused_cols(df)

    # Select a stable study pool
    study_df = _select_study_cases(df)

    # Save output
    os.makedirs(os.path.dirname(CASES_FOR_STUDY_PATH), exist_ok=True)
    study_df.to_csv(CASES_FOR_STUDY_PATH, index=False)

    print("Done.")
    print(f"Saved study cases to: {CASES_FOR_STUDY_PATH}")
    print(f"Rows: {len(study_df)}")
    print(f"Approve rate: {study_df[TARGET_COL].mean():.3f}")


if __name__ == "__main__":
    main()