# app/analysis.py
# Reads study data from SQLite and generates summary + plots for the admin results page.
# Update:
# - Uses non-GUI matplotlib backend (Agg) for Flask safety
# - Parses free-text participant comments from surveys.answers_json
# - Returns comments so they can be shown on the results page

from __future__ import annotations

import os
import json
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

# IMPORTANT: Force a non-GUI backend BEFORE importing pyplot.
# This prevents Tkinter/thread issues when running under Flask.
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from db import get_conn


RESULTS_DIRNAME = "results"  # folder under /static


# ----------------------------
# Helpers for survey scoring
# ----------------------------

def _to_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _compute_sus_from_answers(answers: Dict[str, Any]) -> Optional[float]:
    """
    SUS scoring:
      odd items: (score - 1)
      even items: (5 - score)
      sum * 2.5 => 0..100
    """
    scores: List[int] = []
    for i in range(1, 11):
        key = f"sus_q{i}"
        v = _to_int(answers.get(key))
        if v is None:
            return None
        if i % 2 == 1:
            scores.append(v - 1)
        else:
            scores.append(5 - v)
    return float(sum(scores) * 2.5)


def _compute_trust_from_answers(answers: Dict[str, Any]) -> Optional[float]:
    """
    Simple trust score: average of trust_q1..trust_q3 (1..5)
    """
    vals = []
    for k in ["trust_q1", "trust_q2", "trust_q3"]:
        v = _to_int(answers.get(k))
        if v is None:
            return None
        vals.append(v)
    return float(sum(vals) / len(vals))


def _extract_comment(answers: Dict[str, Any]) -> str:
    """
    Try to find the participant's written feedback (textarea).
    This is robust in case the textarea "name" changes.
    """
    candidate_keys = [
        "comment", "comments", "feedback", "message",
        "free_text", "additional_feedback", "open_feedback",
        "participant_comment", "notes"
    ]

    for k in candidate_keys:
        if k in answers:
            txt = str(answers.get(k, "")).strip()
            if txt:
                return txt

    # Fallback: pick any non-empty non-numeric string that isn't SUS/trust
    for k, v in answers.items():
        if v is None:
            continue
        txt = str(v).strip()
        if not txt:
            continue
        if txt.isdigit():
            continue
        lowk = str(k).lower()
        if lowk.startswith(("sus_", "trust_")):
            continue
        # Avoid storing things like "baseline"/"ai" if that ended up in the form
        if txt.lower() in ("baseline", "ai"):
            continue
        return txt

    return ""


# ----------------------------
# DB read helpers
# ----------------------------

def _read_table_as_df(table: str) -> pd.DataFrame:
    conn = get_conn()
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    finally:
        conn.close()
    return df


def _parse_surveys_df(raw_surveys: pd.DataFrame) -> pd.DataFrame:
    """
    surveys table has answers_json; parse and compute:
      sus_score, trust_score, comment
    """
    if raw_surveys.empty:
        return raw_surveys

    rows = []
    for _, r in raw_surveys.iterrows():
        answers_json = r.get("answers_json", "{}")
        try:
            answers = json.loads(answers_json) if isinstance(answers_json, str) else {}
        except Exception:
            answers = {}

        rows.append(
            {
                "participant_id": r.get("participant_id"),
                "condition": r.get("condition"),
                "sus_score": _compute_sus_from_answers(answers),
                "trust_score": _compute_trust_from_answers(answers),
                "comment": _extract_comment(answers),
            }
        )

    return pd.DataFrame(rows)


# ----------------------------
# Plot helper
# ----------------------------

def _make_bar_plot(
    labels: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    out_path: str,
) -> None:
    plt.figure()
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# ----------------------------
# Main function used by Flask
# ----------------------------

def generate_results(static_root: str) -> Dict[str, Any]:
    """
    Called by the admin results route.
    Creates plots in /static/results and returns:
      summary dict + plot paths
    """
    # Read DB tables
    decisions = _read_table_as_df("decisions")
    events = _read_table_as_df("events")
    surveys_raw = _read_table_as_df("surveys")
    surveys = _parse_surveys_df(surveys_raw)

    # Ensure output folder exists
    results_dir = os.path.join(static_root, RESULTS_DIRNAME)
    os.makedirs(results_dir, exist_ok=True)

    # No data?
    if decisions.empty and surveys_raw.empty:
        return {
            "has_data": False,
            "message": "No results found yet. Complete at least one full run so data is saved in outputs/study.db.",
            "summary": {},
            "plots": {},
        }

    # Clean numeric columns
    if not decisions.empty:
        for col in ["correct", "time_ms", "ai_followed", "ai_seen", "explanation_opened", "ground_truth"]:
            if col in decisions.columns:
                decisions[col] = pd.to_numeric(decisions[col], errors="coerce")

    # Metrics by condition
    acc_by_cond: Dict[str, float] = {}
    time_by_cond: Dict[str, float] = {}
    follow_by_cond: Dict[str, float] = {}

    if not decisions.empty and "condition" in decisions.columns:
        grouped = decisions.groupby("condition")
        for cond, g in grouped:
            cond = str(cond)
            if "correct" in g.columns and g["correct"].notna().any():
                acc_by_cond[cond] = float(g["correct"].mean())
            if "time_ms" in g.columns and g["time_ms"].notna().any():
                time_by_cond[cond] = float(g["time_ms"].mean() / 1000.0)  # seconds
            if cond == "ai" and "ai_followed" in g.columns and g["ai_followed"].notna().any():
                follow_by_cond[cond] = float(g["ai_followed"].mean())

    sus_by_cond: Dict[str, float] = {}
    trust_by_cond: Dict[str, float] = {}

    if isinstance(surveys, pd.DataFrame) and (not surveys.empty) and "condition" in surveys.columns:
        grouped_s = surveys.groupby("condition")
        for cond, g in grouped_s:
            cond = str(cond)
            if "sus_score" in g.columns and g["sus_score"].notna().any():
                sus_by_cond[cond] = float(g["sus_score"].mean())
            if "trust_score" in g.columns and g["trust_score"].notna().any():
                trust_by_cond[cond] = float(g["trust_score"].mean())

    # Collect free-text comments so you can show them on the results page
    comments: List[Dict[str, str]] = []
    if isinstance(surveys, pd.DataFrame) and (not surveys.empty) and "comment" in surveys.columns:
        for _, row in surveys.iterrows():
            c = str(row.get("comment", "")).strip()
            if c:
                comments.append(
                    {
                        "participant_id": str(row.get("participant_id", "")).strip(),
                        "condition": str(row.get("condition", "")).strip(),
                        "comment": c,
                    }
                )

    # Plot ordering
    cond_order = ["baseline", "ai"]

    def ordered_values(d: Dict[str, float]) -> Tuple[List[str], List[float]]:
        labels = [c for c in cond_order if c in d]
        vals = [d[c] for c in labels]
        return labels, vals

    plots: Dict[str, str] = {}

    # Accuracy plot
    if acc_by_cond:
        labels, vals = ordered_values(acc_by_cond)
        out = os.path.join(results_dir, "accuracy.png")
        _make_bar_plot(labels, vals, "Accuracy by condition", "Accuracy (0–1)", out)
        plots["accuracy"] = f"/static/{RESULTS_DIRNAME}/accuracy.png"

    # Time plot
    if time_by_cond:
        labels, vals = ordered_values(time_by_cond)
        out = os.path.join(results_dir, "time.png")
        _make_bar_plot(labels, vals, "Average decision time by condition", "Seconds", out)
        plots["time"] = f"/static/{RESULTS_DIRNAME}/time.png"

    # Trust plot
    if trust_by_cond:
        labels, vals = ordered_values(trust_by_cond)
        out = os.path.join(results_dir, "trust.png")
        _make_bar_plot(labels, vals, "Trust score by condition", "Average (1–5)", out)
        plots["trust"] = f"/static/{RESULTS_DIRNAME}/trust.png"

    # SUS plot
    if sus_by_cond:
        labels, vals = ordered_values(sus_by_cond)
        out = os.path.join(results_dir, "sus.png")
        _make_bar_plot(labels, vals, "SUS score by condition", "SUS (0–100)", out)
        plots["sus"] = f"/static/{RESULTS_DIRNAME}/sus.png"

    # AI-followed plot
    if follow_by_cond:
        labels, vals = ordered_values(follow_by_cond)
        out = os.path.join(results_dir, "ai_followed.png")
        _make_bar_plot(labels, vals, "AI-followed rate (AI condition)", "Rate (0–1)", out)
        plots["ai_followed"] = f"/static/{RESULTS_DIRNAME}/ai_followed.png"

    summary = {
        "accuracy_by_condition": acc_by_cond,
        "time_seconds_by_condition": time_by_cond,
        "trust_by_condition": trust_by_cond,
        "sus_by_condition": sus_by_cond,
        "ai_followed_rate": follow_by_cond,
        "comments": comments,
        "n_decisions": int(len(decisions)),
        "n_surveys": int(len(surveys_raw)),
        "n_events": int(len(events)),
    }

    return {
        "has_data": True,
        "message": "",
        "summary": summary,
        "plots": plots,
    }
