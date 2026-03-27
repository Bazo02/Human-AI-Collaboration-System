# app/analysis.py
# Reads study data from SQLite and creates summary + plots for the admin results page.

from __future__ import annotations

import os
import json
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from app.db import get_conn
from app.config import PARTICIPANT_SUMMARY_PATH

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None


RESULTS_DIRNAME = "results"



# Survey scoring helpers:


def _to_int(x) -> Optional[int]:
    # Tries to convert a value to int and returns None on failure
    try:
        return int(x)
    except Exception:
        return None


def _compute_sus_from_answers(answers: Dict[str, Any]) -> Optional[float]:
    # Computes SUS (0..100) from sus_q1..sus_q10
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
    # Computes trust score as the average of trust_q1..trust_q3 (1..5)
    vals = []
    for k in ["trust_q1", "trust_q2", "trust_q3"]:
        v = _to_int(answers.get(k))
        if v is None:
            return None
        vals.append(v)
    return float(sum(vals) / len(vals))


def _extract_comment(answers: Dict[str, Any]) -> str:
    # Extracts a free-text comment from the survey answers
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

    # Falls back to any non-empty, non-numeric string that is not SUS/trust
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
        if txt.lower() in ("baseline", "ai"):
            continue
        return txt

    return ""



# Database helpers


def _read_table_as_df(table: str) -> pd.DataFrame:
    # Reads a table from SQLite into a DataFrame and returns an empty DF if missing
    conn = get_conn()
    try:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def _parse_surveys_df(raw_surveys: pd.DataFrame) -> pd.DataFrame:
    # Parses surveys.answers_json and computes sus_score, trust_score and comment
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


def _paired_stats(df: pd.DataFrame, baseline_col: str, ai_col: str) -> Dict[str, Any]:
    if df.empty or baseline_col not in df.columns or ai_col not in df.columns:
        return {}

    pair_df = df[["participant_id", baseline_col, ai_col]].dropna().copy()
    if pair_df.empty:
        return {}

    baseline = pair_df[baseline_col].astype(float).to_numpy()
    ai = pair_df[ai_col].astype(float).to_numpy()
    diff = ai - baseline

    result: Dict[str, Any] = {
        "n": int(len(pair_df)),
        "baseline_mean": float(np.mean(baseline)),
        "baseline_sd": float(np.std(baseline, ddof=1)) if len(baseline) > 1 else 0.0,
        "ai_mean": float(np.mean(ai)),
        "ai_sd": float(np.std(ai, ddof=1)) if len(ai) > 1 else 0.0,
        "mean_difference": float(np.mean(diff)),
        "sd_difference": float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0,
    }

    if len(diff) > 1 and np.std(diff, ddof=1) > 0:
        result["cohens_dz"] = float(np.mean(diff) / np.std(diff, ddof=1))
    else:
        result["cohens_dz"] = 0.0

    if scipy_stats is not None and len(diff) > 1:
        t_stat, p_value = scipy_stats.ttest_rel(ai, baseline, nan_policy="omit")
        sem = scipy_stats.sem(diff, nan_policy="omit")
        ci_low, ci_high = scipy_stats.t.interval(
            confidence=0.95,
            df=len(diff) - 1,
            loc=np.mean(diff),
            scale=sem,
        )
        result["t_statistic"] = float(t_stat)
        result["p_value"] = float(p_value)
        result["ci95_low"] = float(ci_low)
        result["ci95_high"] = float(ci_high)
    else:
        result["t_statistic"] = None
        result["p_value"] = None
        result["ci95_low"] = None
        result["ci95_high"] = None

    return result


def _participant_level_summary(
    participants: pd.DataFrame,
    decisions: pd.DataFrame,
    surveys: pd.DataFrame,
) -> pd.DataFrame:
    if decisions.empty and surveys.empty and participants.empty:
        return pd.DataFrame()

    participant_df = participants.copy()
    if participant_df.empty:
        participant_ids = set()

        if not decisions.empty and "participant_id" in decisions.columns:
            participant_ids.update(str(x) for x in decisions["participant_id"].dropna().tolist())

        if not surveys.empty and "participant_id" in surveys.columns:
            participant_ids.update(str(x) for x in surveys["participant_id"].dropna().tolist())

        participant_df = pd.DataFrame({"participant_id": sorted(participant_ids)})

    if "completed" not in participant_df.columns:
        participant_df["completed"] = 0

    decision_summary = pd.DataFrame()
    if not decisions.empty:
        agg_map: Dict[str, Tuple[str, str]] = {}

        if "correct" in decisions.columns:
            agg_map["accuracy"] = ("correct", "mean")
        if "time_ms" in decisions.columns:
            agg_map["avg_time_seconds"] = ("time_ms", lambda s: float(s.mean() / 1000.0))
        if "ai_followed" in decisions.columns:
            agg_map["ai_followed_rate"] = ("ai_followed", "mean")
        if "ai_seen" in decisions.columns:
            agg_map["ai_seen_rate"] = ("ai_seen", "mean")
        if "explanation_opened" in decisions.columns:
            agg_map["explanation_open_rate"] = ("explanation_opened", "mean")
        if "ai_confidence" in decisions.columns:
            agg_map["avg_ai_confidence"] = ("ai_confidence", "mean")
        if "ai_prob_approve" in decisions.columns:
            agg_map["avg_ai_prob_approve"] = ("ai_prob_approve", "mean")

        if agg_map:
            grouped = decisions.groupby(["participant_id", "condition"]).agg(**agg_map).reset_index()
            pivoted = grouped.pivot(index="participant_id", columns="condition")
            pivoted.columns = [f"{cond}_{metric}" for metric, cond in pivoted.columns]
            decision_summary = pivoted.reset_index()

    survey_summary = pd.DataFrame()
    if not surveys.empty:
        survey_summary = surveys.groupby("participant_id", as_index=False).agg(
            trust_score=("trust_score", "mean"),
            sus_score=("sus_score", "mean"),
            comment=("comment", lambda s: " | ".join([str(x).strip() for x in s if str(x).strip()])),
        )

    merged = participant_df.copy()

    if not decision_summary.empty:
        merged = merged.merge(decision_summary, on="participant_id", how="left")

    if not survey_summary.empty:
        merged = merged.merge(survey_summary, on="participant_id", how="left")

    return merged.sort_values("participant_id").reset_index(drop=True)



# Plot helper


def _make_bar_plot(
    labels: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    out_path: str,
) -> None:
    # Creates a simple bar plot and saves it to disk
    plt.figure()
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _make_count_plot(
    labels: List[str],
    values: List[int],
    title: str,
    ylabel: str,
    out_path: str,
) -> None:
    # Creates a simple count plot and saves it to disk
    plt.figure()
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()



# Main function (used by Flask)


def generate_results(static_root: str) -> Dict[str, Any]:
    # Reads study tables from the database
    participants = _read_table_as_df("participants")
    decisions = _read_table_as_df("decisions")
    events = _read_table_as_df("events")
    surveys_raw = _read_table_as_df("surveys")
    surveys = _parse_surveys_df(surveys_raw)

    # Ensures the output folder exists
    results_dir = os.path.join(static_root, RESULTS_DIRNAME)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.dirname(PARTICIPANT_SUMMARY_PATH), exist_ok=True)

    # Returns early if there is no data yet
    if decisions.empty and surveys_raw.empty and participants.empty:
        return {
            "has_data": False,
            "message": "No results found yet. Complete at least one full run so data is saved in outputs/study.db.",
            "summary": {},
            "plots": {},
        }

    # Converts selected decision columns to numeric when present
    if not decisions.empty:
        for col in [
            "correct",
            "time_ms",
            "ai_followed",
            "ai_seen",
            "explanation_opened",
            "ground_truth",
            "ai_confidence",
            "ai_prob_approve",
        ]:
            if col in decisions.columns:
                decisions[col] = pd.to_numeric(decisions[col], errors="coerce")

    if not participants.empty and "completed" in participants.columns:
        participants["completed"] = pd.to_numeric(participants["completed"], errors="coerce").fillna(0).astype(int)

    participant_summary = _participant_level_summary(participants, decisions, surveys)

    if not participant_summary.empty:
        participant_summary.to_csv(PARTICIPANT_SUMMARY_PATH, index=False)

    acc_by_cond: Dict[str, float] = {}
    time_by_cond: Dict[str, float] = {}
    follow_by_cond: Dict[str, float] = {}
    trust_by_cond: Dict[str, float] = {}
    sus_by_cond: Dict[str, float] = {}
    ai_confidence_by_cond: Dict[str, float] = {}
    ai_prob_approve_by_cond: Dict[str, float] = {}
    explanation_open_by_cond: Dict[str, float] = {}
    ai_seen_by_cond: Dict[str, float] = {}

    if not participant_summary.empty:
        if "baseline_accuracy" in participant_summary.columns and participant_summary["baseline_accuracy"].notna().any():
            acc_by_cond["baseline"] = float(participant_summary["baseline_accuracy"].mean())
        if "ai_accuracy" in participant_summary.columns and participant_summary["ai_accuracy"].notna().any():
            acc_by_cond["ai"] = float(participant_summary["ai_accuracy"].mean())

        if "baseline_avg_time_seconds" in participant_summary.columns and participant_summary["baseline_avg_time_seconds"].notna().any():
            time_by_cond["baseline"] = float(participant_summary["baseline_avg_time_seconds"].mean())
        if "ai_avg_time_seconds" in participant_summary.columns and participant_summary["ai_avg_time_seconds"].notna().any():
            time_by_cond["ai"] = float(participant_summary["ai_avg_time_seconds"].mean())

        if "ai_ai_followed_rate" in participant_summary.columns and participant_summary["ai_ai_followed_rate"].notna().any():
            follow_by_cond["ai"] = float(participant_summary["ai_ai_followed_rate"].mean())

        if "trust_score" in participant_summary.columns and participant_summary["trust_score"].notna().any():
            trust_by_cond["ai"] = float(participant_summary["trust_score"].mean())

        if "sus_score" in participant_summary.columns and participant_summary["sus_score"].notna().any():
            sus_by_cond["ai"] = float(participant_summary["sus_score"].mean())

        if "ai_avg_ai_confidence" in participant_summary.columns and participant_summary["ai_avg_ai_confidence"].notna().any():
            ai_confidence_by_cond["ai"] = float(participant_summary["ai_avg_ai_confidence"].mean())

        if "ai_avg_ai_prob_approve" in participant_summary.columns and participant_summary["ai_avg_ai_prob_approve"].notna().any():
            ai_prob_approve_by_cond["ai"] = float(participant_summary["ai_avg_ai_prob_approve"].mean())

        if "ai_explanation_open_rate" in participant_summary.columns and participant_summary["ai_explanation_open_rate"].notna().any():
            explanation_open_by_cond["ai"] = float(participant_summary["ai_explanation_open_rate"].mean())
        if "baseline_explanation_open_rate" in participant_summary.columns and participant_summary["baseline_explanation_open_rate"].notna().any():
            explanation_open_by_cond["baseline"] = float(participant_summary["baseline_explanation_open_rate"].mean())

        if "ai_ai_seen_rate" in participant_summary.columns and participant_summary["ai_ai_seen_rate"].notna().any():
            ai_seen_by_cond["ai"] = float(participant_summary["ai_ai_seen_rate"].mean())
        if "baseline_ai_seen_rate" in participant_summary.columns and participant_summary["baseline_ai_seen_rate"].notna().any():
            ai_seen_by_cond["baseline"] = float(participant_summary["baseline_ai_seen_rate"].mean())

    # Collects free-text comments for display on the results page
    comments: List[Dict[str, str]] = []
    if isinstance(participant_summary, pd.DataFrame) and (not participant_summary.empty) and "comment" in participant_summary.columns:
        for _, row in participant_summary.iterrows():
            c = str(row.get("comment", "")).strip()
            if c:
                comments.append(
                    {
                        "participant_id": str(row.get("participant_id", "")).strip(),
                        "condition": "ai",
                        "comment": c,
                    }
                )

    # Defines how conditions should be ordered in plots
    cond_order = ["baseline", "ai"]

    def ordered_values(d: Dict[str, float]) -> Tuple[List[str], List[float]]:
        # Returns labels/values in a consistent order
        labels = [c for c in cond_order if c in d]
        vals = [d[c] for c in labels]
        return labels, vals

    plots: Dict[str, str] = {}

    # Creates accuracy plot
    if acc_by_cond:
        labels, vals = ordered_values(acc_by_cond)
        out = os.path.join(results_dir, "accuracy.png")
        _make_bar_plot(labels, vals, "Accuracy by condition", "Accuracy (0–1)", out)
        plots["accuracy"] = f"/static/{RESULTS_DIRNAME}/accuracy.png"

    # Creates time plot
    if time_by_cond:
        labels, vals = ordered_values(time_by_cond)
        out = os.path.join(results_dir, "time.png")
        _make_bar_plot(labels, vals, "Average decision time by condition", "Seconds", out)
        plots["time"] = f"/static/{RESULTS_DIRNAME}/time.png"

    # Creates trust plot
    if trust_by_cond:
        labels, vals = ordered_values(trust_by_cond)
        out = os.path.join(results_dir, "trust.png")
        _make_bar_plot(labels, vals, "Trust score by condition", "Average (1–5)", out)
        plots["trust"] = f"/static/{RESULTS_DIRNAME}/trust.png"

    # Creates SUS plot
    if sus_by_cond:
        labels, vals = ordered_values(sus_by_cond)
        out = os.path.join(results_dir, "sus.png")
        _make_bar_plot(labels, vals, "SUS score by condition", "SUS (0–100)", out)
        plots["sus"] = f"/static/{RESULTS_DIRNAME}/sus.png"

    # Creates AI-followed plot
    if follow_by_cond:
        labels, vals = ordered_values(follow_by_cond)
        out = os.path.join(results_dir, "ai_followed.png")
        _make_bar_plot(labels, vals, "AI-followed rate (AI condition)", "Rate (0–1)", out)
        plots["ai_followed"] = f"/static/{RESULTS_DIRNAME}/ai_followed.png"

    # Creates AI confidence plot
    if ai_confidence_by_cond:
        labels, vals = ordered_values(ai_confidence_by_cond)
        out = os.path.join(results_dir, "ai_confidence.png")
        _make_bar_plot(labels, vals, "Average AI confidence", "Confidence (0–1)", out)
        plots["ai_confidence"] = f"/static/{RESULTS_DIRNAME}/ai_confidence.png"

    # Creates AI probability plot
    if ai_prob_approve_by_cond:
        labels, vals = ordered_values(ai_prob_approve_by_cond)
        out = os.path.join(results_dir, "ai_prob_approve.png")
        _make_bar_plot(labels, vals, "Average AI approval probability", "Probability (0–1)", out)
        plots["ai_prob_approve"] = f"/static/{RESULTS_DIRNAME}/ai_prob_approve.png"

    # Creates explanation-open plot
    if explanation_open_by_cond:
        labels, vals = ordered_values(explanation_open_by_cond)
        out = os.path.join(results_dir, "explanation_open_rate.png")
        _make_bar_plot(labels, vals, "Explanation open rate by condition", "Rate (0–1)", out)
        plots["explanation_open_rate"] = f"/static/{RESULTS_DIRNAME}/explanation_open_rate.png"

    # Creates AI-seen plot
    if ai_seen_by_cond:
        labels, vals = ordered_values(ai_seen_by_cond)
        out = os.path.join(results_dir, "ai_seen_rate.png")
        _make_bar_plot(labels, vals, "AI seen rate by condition", "Rate (0–1)", out)
        plots["ai_seen_rate"] = f"/static/{RESULTS_DIRNAME}/ai_seen_rate.png"

    # Creates age-group distribution plot
    if not participants.empty and "age_group" in participants.columns:
        counts = participants["age_group"].fillna("").astype(str).str.strip()
        counts = counts[counts != ""].value_counts()
        if not counts.empty:
            out = os.path.join(results_dir, "age_group_distribution.png")
            _make_count_plot(counts.index.tolist(), counts.astype(int).tolist(), "Participant age group distribution", "Count", out)
            plots["age_group_distribution"] = f"/static/{RESULTS_DIRNAME}/age_group_distribution.png"

    # Creates background distribution plot
    if not participants.empty and "background" in participants.columns:
        counts = participants["background"].fillna("").astype(str).str.strip()
        counts = counts[counts != ""].value_counts()
        if not counts.empty:
            out = os.path.join(results_dir, "background_distribution.png")
            _make_count_plot(counts.index.tolist(), counts.astype(int).tolist(), "Participant background distribution", "Count", out)
            plots["background_distribution"] = f"/static/{RESULTS_DIRNAME}/background_distribution.png"

    # Creates AI familiarity distribution plot
    if not participants.empty and "ai_familiarity" in participants.columns:
        counts = participants["ai_familiarity"].fillna("").astype(str).str.strip()
        counts = counts[counts != ""].value_counts()
        if not counts.empty:
            out = os.path.join(results_dir, "ai_familiarity_distribution.png")
            _make_count_plot(counts.index.tolist(), counts.astype(int).tolist(), "AI familiarity distribution", "Count", out)
            plots["ai_familiarity_distribution"] = f"/static/{RESULTS_DIRNAME}/ai_familiarity_distribution.png"

    # Creates finance familiarity distribution plot
    if not participants.empty and "finance_familiarity" in participants.columns:
        counts = participants["finance_familiarity"].fillna("").astype(str).str.strip()
        counts = counts[counts != ""].value_counts()
        if not counts.empty:
            out = os.path.join(results_dir, "finance_familiarity_distribution.png")
            _make_count_plot(counts.index.tolist(), counts.astype(int).tolist(), "Finance familiarity distribution", "Count", out)
            plots["finance_familiarity_distribution"] = f"/static/{RESULTS_DIRNAME}/finance_familiarity_distribution.png"

    paired_tests = {}
    if not participant_summary.empty:
        paired_tests["accuracy"] = _paired_stats(participant_summary, "baseline_accuracy", "ai_accuracy")
        paired_tests["decision_time_seconds"] = _paired_stats(
            participant_summary,
            "baseline_avg_time_seconds",
            "ai_avg_time_seconds",
        )

    participant_summary_preview: List[Dict[str, Any]] = []
    if not participant_summary.empty:
        preview_df = participant_summary.copy()
        for col in preview_df.columns:
            if pd.api.types.is_float_dtype(preview_df[col]):
                preview_df[col] = preview_df[col].round(3)
        participant_summary_preview = preview_df.to_dict(orient="records")

    summary = {
        "accuracy_by_condition": acc_by_cond,
        "time_seconds_by_condition": time_by_cond,
        "trust_by_condition": trust_by_cond,
        "sus_by_condition": sus_by_cond,
        "ai_followed_rate": follow_by_cond,
        "ai_confidence_by_condition": ai_confidence_by_cond,
        "ai_prob_approve_by_condition": ai_prob_approve_by_cond,
        "explanation_open_rate_by_condition": explanation_open_by_cond,
        "ai_seen_rate_by_condition": ai_seen_by_cond,
        "paired_tests": paired_tests,
        "comments": comments,
        "participant_summary_preview": participant_summary_preview,
        "n_participants": int(len(participants)) if not participants.empty else 0,
        "n_completed_participants": int(participants["completed"].sum()) if (not participants.empty and "completed" in participants.columns) else 0,
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