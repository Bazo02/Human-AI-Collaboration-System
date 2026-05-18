# app/data_prep.py: cleans the raw dataset and selects a balanced set of cases for the user study.

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

STUDY_SET_SIZE = 120     # total number of cases in the study pool
BORDERLINE_SHARE = 0.30  # share of borderline (harder) cases to include

# Removes duplicates, fills missing values, and adds a case_id column if missing
def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    if "case_id" not in df.columns:
        df.insert(0, "case_id", range(1, len(df) + 1))

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

# Computes a simple heuristic risk score used only for selecting study cases (not shown to participants)
def _heuristic_risk_score(df: pd.DataFrame) -> pd.Series:
    def col_or_zero(name: str) -> pd.Series:
        return df[name] if name in df.columns else pd.Series([0] * len(df))

    credit_score = col_or_zero("credit_score")
    income = col_or_zero("annual_income")
    loan_amount = col_or_zero("loan_amount")
    existing_loans = col_or_zero("existing_loans_count")

    credit_risk = (700 - credit_score) / 200.0
    income_risk = (500000 - income) / 500000.0
    loan_risk = loan_amount / 500000.0
    existing_risk = existing_loans / 5.0

    credit_risk = credit_risk.clip(-2, 2)
    income_risk = income_risk.clip(-2, 2)
    loan_risk = loan_risk.clip(0, 3)
    existing_risk = existing_risk.clip(0, 3)

    return 0.50 * credit_risk + 0.25 * income_risk + 0.20 * loan_risk + 0.05 * existing_risk

# Selects a balanced mix of approved/rejected and borderline cases for the study pool
def _select_study_cases(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["risk_score"] = _heuristic_risk_score(df)

    df_approve = df[df[TARGET_COL] == 1].copy()
    df_reject = df[df[TARGET_COL] == 0].copy()

    half = STUDY_SET_SIZE // 2
    n_approve = min(half, len(df_approve))
    n_reject = min(STUDY_SET_SIZE - n_approve, len(df_reject))

    n_borderline_total = int(STUDY_SET_SIZE * BORDERLINE_SHARE)
    n_borderline_each = n_borderline_total // 2

    approve_borderline = df_approve.sort_values("risk_score", ascending=False).head(n_borderline_each)
    approve_easy = df_approve.sort_values("risk_score", ascending=True).head(max(0, n_approve - len(approve_borderline)))

    reject_borderline = df_reject.sort_values("risk_score", ascending=True).head(n_borderline_each)
    reject_easy = df_reject.sort_values("risk_score", ascending=False).head(max(0, n_reject - len(reject_borderline)))

    selected = pd.concat([approve_borderline, approve_easy, reject_borderline, reject_easy], axis=0)

    if len(selected) < STUDY_SET_SIZE:
        missing = STUDY_SET_SIZE - len(selected)
        remaining = df.drop(index=selected.index, errors="ignore")
        if missing > 0 and len(remaining) > 0:
            fill = remaining.sample(n=min(missing, len(remaining)), random_state=42)
            selected = pd.concat([selected, fill], axis=0)

    selected = selected.sample(frac=1.0, random_state=123).reset_index(drop=True)
    selected = selected.drop(columns=["risk_score"], errors="ignore")

    return selected

# Drops columns that are not needed in the UI (e.g. sensitive or irrelevant fields)
def _drop_sensitive_and_unused_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in DROP_COLS_FOR_UI:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df

# Entry point: loads, cleans, selects cases, and saves the study dataset to disk
def main() -> None:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find dataset at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = _basic_clean(df)
    df = _drop_sensitive_and_unused_cols(df)
    study_df = _select_study_cases(df)

    os.makedirs(os.path.dirname(CASES_FOR_STUDY_PATH), exist_ok=True)
    study_df.to_csv(CASES_FOR_STUDY_PATH, index=False)

    print("Done.")
    print(f"Saved study cases to: {CASES_FOR_STUDY_PATH}")
    print(f"Rows: {len(study_df)}")
    print(f"Approve rate: {study_df[TARGET_COL].mean():.3f}")


if __name__ == "__main__":
    main()
