# app/config.py: central configuration file for file paths, study settings, and Flask options.

from __future__ import annotations

import os
import secrets

# Uses an environment variable for the secret key, or generates a random one if not set
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(16))


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

DATA_PATH = os.path.join(DATA_DIR, "loanapproval.csv")
CASES_FOR_STUDY_PATH = os.path.join(DATA_DIR, "cases_for_study.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "app", "model.joblib")
PARTICIPANT_SUMMARY_PATH = os.path.join(OUTPUTS_DIR, "participant_summary.csv")

# Name of the column the model predicts (1 = approved, 0 = rejected)
TARGET_COL = "loan_approved"

# Columns excluded from the UI 
DROP_COLS_FOR_UI = [
    "applicant_id",
    "gender",
    "marital_status",
]

CONDITION_NAMES = {
    "baseline": "Non-assisted (baseline)",
    "ai": "AI-assisted",
}

# Number of loan cases shown per block (baseline and AI are separate blocks)
CASES_PER_PARTICIPANT = 12
TOTAL_CASES_PER_PARTICIPANT = CASES_PER_PARTICIPANT * 2

# If the model's approval probability is at or above this threshold, it recommends "Approve"
APPROVAL_THRESHOLD = 0.65

EVENTS_LOG_PATH = os.path.join(OUTPUTS_DIR, "events.csv")
DECISIONS_LOG_PATH = os.path.join(OUTPUTS_DIR, "decisions.csv")
SURVEYS_LOG_PATH = os.path.join(OUTPUTS_DIR, "surveys.csv")
SQLITE_DB_PATH = os.path.join(OUTPUTS_DIR, "study.db")

# Admin password for accessing the admin interface
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin")
