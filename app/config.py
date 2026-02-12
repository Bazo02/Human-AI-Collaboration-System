# app/config.py
# Basic configuration for my thesis prototype.

from __future__ import annotations

import os
import secrets

# -----------------------
# Flask settings
# -----------------------

SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(16))

# -----------------------
# Paths
# -----------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

DATA_PATH = os.path.join(DATA_DIR, "loanapproval.csv")
CASES_FOR_STUDY_PATH = os.path.join(DATA_DIR, "cases_for_study.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "app", "model.joblib")

# -----------------------
# Dataset / feature settings
# -----------------------

TARGET_COL = "loan_approved"

DROP_COLS_FOR_UI = [
    "applicant_id",
    "gender",
    "marital_status",
]

CONDITION_NAMES = {
    "baseline": "Non-assisted (baseline)",
    "ai": "AI-assisted",
}

# -----------------------
# Experiment settings
# -----------------------

# Task cases config :
# - Each participant completes 2 blocks of 20 cases
# - Block 1: baseline (no AI)
# - Block 2: ai (AI recommendation + explanation)
CASES_PER_PARTICIPANT = 10  # cases per block (kept name for compatibility)
TOTAL_CASES_PER_PARTICIPANT = CASES_PER_PARTICIPANT * 2  # 40 total

# Decision threshold:
# If P(approve) >= threshold -> "Approve", else "Reject"
APPROVAL_THRESHOLD = 0.65

# -----------------------
# Logging filenames
# -----------------------

EVENTS_LOG_PATH = os.path.join(OUTPUTS_DIR, "events.csv")
DECISIONS_LOG_PATH = os.path.join(OUTPUTS_DIR, "decisions.csv")
SURVEYS_LOG_PATH = os.path.join(OUTPUTS_DIR, "surveys.csv")
SQLITE_DB_PATH = os.path.join(OUTPUTS_DIR, "study.db")

# -----------------------
# Admin
# -----------------------

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin")
