# app/analysis.py: analyzes study data from the database and generates graphs and statistics.

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

# Helper function for converting values to integers
def _to_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None

# Computes the SUS score from the answers to the 10 SUS questions
def _compute_sus_from_answers(answers: Dict[str, Any]) -> Optional[float]:
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

# Computes the average of the three trust questions to get an overall trust score
def _compute_trust_from_answers(answers: Dict[str, Any]) -> Optional[float]:
    vals = []
    for k in ["trust_q1", "trust_q2", "trust_q3"]:
        v = _to_int(answers.get(k))
        if v is None:
            return None
        vals.append(v)
    return float(sum(vals) / len(vals))

# Extracts the free-text comment from the survey answers
def _extract_comment(answers: Dict[str, Any]) -> str:
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
    for k, v in answers.items():
        if v is None:
            continue
        txt = str(v).strip()
        if not txt or txt.isdigit():
            continue
        if str(k).lower().startswith(("sus_", "trust_")):
            continue
        if txt.lower() in ("baseline", "ai"):
            continue
        return txt
    return ""

# Reads a table from the database into a pandas DataFrame
def _read_table_as_df(table: str) -> pd.DataFrame:
    conn = get_conn()
    try:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()

# Parses the raw surveys DataFrame, extracting trust scores, SUS scores, and comments into a cleaner format
def _parse_surveys_df(raw_surveys: pd.DataFrame) -> pd.DataFrame:
    if raw_surveys.empty:
        return raw_surveys
    rows = []
    for _, r in raw_surveys.iterrows():
        answers_json = r.get("answers_json", "{}")
        try:
            answers = json.loads(answers_json) if isinstance(answers_json, str) else {}
        except Exception:
            answers = {}
        rows.append({
            "participant_id": r.get("participant_id"),
            "condition": r.get("condition"),
            "sus_score": _compute_sus_from_answers(answers),
            "trust_score": _compute_trust_from_answers(answers),
            "comment": _extract_comment(answers),
        })
    return pd.DataFrame(rows)

# Calculates paired statistics (t-test) between baseline and AI conditions for a given metric
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

    # calculates Cohen's dz for paired samples (mean of differences divided by SD of differences)
    if len(diff) > 1 and np.std(diff, ddof=1) > 0:
        result["cohens_dz"] = float(np.mean(diff) / np.std(diff, ddof=1))
    else:
        result["cohens_dz"] = 0.0

    # runs a paired t-test with 95% confidence interval for the mean difference
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

# Divides participants into groups based on whether their accuracy improved or worsened with AI, and runs Welch's t-test between the two groups
def _accuracy_improvement_groups(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}
    needed = ["participant_id", "baseline_accuracy", "ai_accuracy"]
    if not all(c in df.columns for c in needed):
        return {}

    pair_df = df[needed].dropna().copy()
    if pair_df.empty:
        return {}

    pair_df["diff"] = pair_df["ai_accuracy"] - pair_df["baseline_accuracy"]
    improved = pair_df[pair_df["diff"] > 0]
    worsened = pair_df[pair_df["diff"] < 0]
    unchanged = pair_df[pair_df["diff"] == 0]

    result: Dict[str, Any] = {
        "n_improved": int(len(improved)),
        "n_worsened": int(len(worsened)),
        "n_unchanged": int(len(unchanged)),
        "improved_baseline_mean": float(improved["baseline_accuracy"].mean()) if not improved.empty else None,
        "improved_baseline_sd": float(improved["baseline_accuracy"].std(ddof=1)) if len(improved) > 1 else None,
        "worsened_baseline_mean": float(worsened["baseline_accuracy"].mean()) if not worsened.empty else None,
        "worsened_baseline_sd": float(worsened["baseline_accuracy"].std(ddof=1)) if len(worsened) > 1 else None,
    }

    if scipy_stats is not None and len(improved) > 1 and len(worsened) > 1:
        t_stat, p_value = scipy_stats.ttest_ind(
            improved["baseline_accuracy"].to_numpy(),
            worsened["baseline_accuracy"].to_numpy(),
            equal_var=False,
        )
        n1 = len(improved)
        n2 = len(worsened)
        s1 = float(improved["baseline_accuracy"].std(ddof=1))
        s2 = float(worsened["baseline_accuracy"].std(ddof=1))
        welch_df = (s1**2/n1 + s2**2/n2)**2 / (
            (s1**2/n1)**2 / (n1 - 1) + (s2**2/n2)**2 / (n2 - 1)
        )
        result["group_ttest_t"] = float(t_stat)
        result["group_ttest_p"] = float(p_value)
        result["group_ttest_df"] = float(round(welch_df, 2))
    else:
        result["group_ttest_t"] = None
        result["group_ttest_p"] = None
        result["group_ttest_df"] = None

    # checks if worsened participants had high AI follow rates
    if "ai_ai_followed_rate" in df.columns:
        worsened_ids = worsened["participant_id"].tolist()
        worsened_follow = df[df["participant_id"].isin(worsened_ids)]["ai_ai_followed_rate"].dropna()
        if not worsened_follow.empty:
            result["worsened_high_follow_count"] = int((worsened_follow >= 0.90).sum())
            result["worsened_follow_mean"] = float(worsened_follow.mean())

    return result

# Computes Spearman correlation between trust score and AI-followed rate, and reports the minimum trust value
def _spearman_trust_vs_ai_followed(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}
    needed = ["trust_score", "ai_ai_followed_rate"]
    if not all(c in df.columns for c in needed):
        return {}

    pair_df = df[needed].dropna().copy()
    if len(pair_df) < 3:
        return {}

    result: Dict[str, Any] = {}

    if scipy_stats is not None:
        rs, p = scipy_stats.spearmanr(
            pair_df["trust_score"].to_numpy(),
            pair_df["ai_ai_followed_rate"].to_numpy(),
        )
        result["rs"] = float(rs)
        result["p_value"] = float(p)
        result["n"] = int(len(pair_df))

    min_trust_val = float(pair_df["trust_score"].min())
    min_trust_rows = pair_df[pair_df["trust_score"] == min_trust_val]
    result["min_trust"] = min_trust_val
    result["min_trust_ai_followed_values"] = sorted(
        [float(v) for v in min_trust_rows["ai_ai_followed_rate"].tolist()]
    )
    result["min_trust_n"] = int(len(min_trust_rows))

    return result

# Returns descriptive statistics for the AI-followed rate across participants
def _ai_followed_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty or "ai_ai_followed_rate" not in df.columns:
        return {}
    vals = df["ai_ai_followed_rate"].dropna().to_numpy()
    if len(vals) == 0:
        return {}
    return {
        "mean": float(np.mean(vals)),
        "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "median": float(np.median(vals)),
    }

# Returns mean and SD of AI confidence scores across all AI-condition decisions
def _ai_confidence_stats(decisions: pd.DataFrame) -> Dict[str, Any]:
    if decisions.empty or "ai_confidence" not in decisions.columns:
        return {}
    ai_decisions = decisions[decisions["condition"] == "ai"]["ai_confidence"].dropna()
    if ai_decisions.empty:
        return {}
    return {
        "mean": float(ai_decisions.mean()),
        "sd": float(ai_decisions.std(ddof=1)) if len(ai_decisions) > 1 else 0.0,
    }

# Returns mean and SD of AI approval probability across all AI-condition decisions
def _ai_prob_approve_stats(decisions: pd.DataFrame) -> Dict[str, Any]:
    if decisions.empty or "ai_prob_approve" not in decisions.columns:
        return {}
    ai_decisions = decisions[decisions["condition"] == "ai"]["ai_prob_approve"].dropna()
    if ai_decisions.empty:
        return {}
    return {
        "mean": float(ai_decisions.mean()),
        "sd": float(ai_decisions.std(ddof=1)) if len(ai_decisions) > 1 else 0.0,
    }

# Builds a summary DataFrame with one row per participant, merging decisions, surveys, and participant info
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

# Creates a simple bar chart and saves it to disk
def _make_bar_plot(labels, values, title, ylabel, out_path):
    plt.figure()
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

# Creates a bar chart with rotated x-axis labels, used for count/distribution data
def _make_count_plot(labels, values, title, ylabel, out_path):
    plt.figure()
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

# Creates a scatter plot with an optional linear trend line
def _make_scatter_plot(x, y, title, xlabel, ylabel, out_path, trend_line=False):
    plt.figure()
    plt.scatter(x, y, alpha=0.6)
    if trend_line and len(x) > 1:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x), max(x), 100)
        plt.plot(x_line, p(x_line), "k--", alpha=0.5, linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

# Creates a scatter plot comparing baseline vs AI accuracy per participant, with a diagonal no-change reference line
def _make_scatter_diagonal_plot(baseline, ai, title, out_path):
    plt.figure()
    plt.scatter(baseline, ai, alpha=0.6)
    lims = [min(min(baseline), min(ai)) - 0.05, max(max(baseline), max(ai)) + 0.05]
    plt.plot(lims, lims, "k--", alpha=0.4, linewidth=1)
    plt.xlabel("Baseline accuracy")
    plt.ylabel("AI-assisted accuracy")
    plt.title(title)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

# Main function: loads all data, runs analyses, generates plots, and returns a summary dict
def generate_results(static_root: str) -> Dict[str, Any]:
    participants = _read_table_as_df("participants")
    decisions = _read_table_as_df("decisions")
    events = _read_table_as_df("events")
    surveys_raw = _read_table_as_df("surveys")
    surveys = _parse_surveys_df(surveys_raw)

    results_dir = os.path.join(static_root, RESULTS_DIRNAME)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.dirname(PARTICIPANT_SUMMARY_PATH), exist_ok=True)

    if decisions.empty and surveys_raw.empty and participants.empty:
        return {
            "has_data": False,
            "message": "No results found yet. Complete at least one full run so data is saved in outputs/study.db.",
            "summary": {},
            "plots": {},
        }

    if not decisions.empty:
        for col in ["correct", "time_ms", "ai_followed", "ai_seen", "explanation_opened",
                    "ground_truth", "ai_confidence", "ai_prob_approve"]:
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
        for col, d in [
            ("baseline_accuracy", acc_by_cond), ("ai_accuracy", acc_by_cond),
            ("baseline_avg_time_seconds", time_by_cond), ("ai_avg_time_seconds", time_by_cond),
            ("ai_ai_followed_rate", follow_by_cond),
            ("trust_score", trust_by_cond), ("sus_score", sus_by_cond),
            ("ai_avg_ai_confidence", ai_confidence_by_cond),
            ("ai_avg_ai_prob_approve", ai_prob_approve_by_cond),
            ("ai_explanation_open_rate", explanation_open_by_cond),
            ("baseline_explanation_open_rate", explanation_open_by_cond),
            ("ai_ai_seen_rate", ai_seen_by_cond),
            ("baseline_ai_seen_rate", ai_seen_by_cond),
        ]:
            if col in participant_summary.columns and participant_summary[col].notna().any():
                if col == "trust_score": d["ai"] = float(participant_summary[col].mean())
                elif col == "sus_score": d["ai"] = float(participant_summary[col].mean())
                elif col == "ai_ai_followed_rate": d["ai"] = float(participant_summary[col].mean())
                elif col == "ai_avg_ai_confidence": d["ai"] = float(participant_summary[col].mean())
                elif col == "ai_avg_ai_prob_approve": d["ai"] = float(participant_summary[col].mean())
                elif col == "ai_explanation_open_rate": d["ai"] = float(participant_summary[col].mean())
                elif col == "baseline_explanation_open_rate": d["baseline"] = float(participant_summary[col].mean())
                elif col == "ai_ai_seen_rate": d["ai"] = float(participant_summary[col].mean())
                elif col == "baseline_ai_seen_rate": d["baseline"] = float(participant_summary[col].mean())
                elif col == "baseline_accuracy": d["baseline"] = float(participant_summary[col].mean())
                elif col == "ai_accuracy": d["ai"] = float(participant_summary[col].mean())
                elif col == "baseline_avg_time_seconds": d["baseline"] = float(participant_summary[col].mean())
                elif col == "ai_avg_time_seconds": d["ai"] = float(participant_summary[col].mean())

    accuracy_groups = _accuracy_improvement_groups(participant_summary)
    spearman_trust_followed = _spearman_trust_vs_ai_followed(participant_summary)
    ai_followed_dist = _ai_followed_distribution(participant_summary)
    ai_confidence_detail = _ai_confidence_stats(decisions)
    ai_prob_detail = _ai_prob_approve_stats(decisions)

    comments: List[Dict[str, str]] = []
    if not participant_summary.empty and "comment" in participant_summary.columns:
        for _, row in participant_summary.iterrows():
            c = str(row.get("comment", "")).strip()
            if c:
                comments.append({
                    "participant_id": str(row.get("participant_id", "")).strip(),
                    "condition": "ai",
                    "comment": c,
                })

    cond_order = ["baseline", "ai"]

    def ordered_values(d):
        labels = [c for c in cond_order if c in d]
        return labels, [d[c] for c in labels]

    plots: Dict[str, str] = {}

    if acc_by_cond:
        labels, vals = ordered_values(acc_by_cond)
        out = os.path.join(results_dir, "accuracy.png")
        _make_bar_plot(labels, vals, "Mean decision accuracy by condition", "Accuracy (0–1)", out)
        plots["accuracy"] = f"/static/{RESULTS_DIRNAME}/accuracy.png"

    if not participant_summary.empty:
        if "baseline_accuracy" in participant_summary.columns and "ai_accuracy" in participant_summary.columns:
            pair_df = participant_summary[["baseline_accuracy", "ai_accuracy"]].dropna()
            if len(pair_df) >= 2:
                out = os.path.join(results_dir, "accuracy_scatter.png")
                _make_scatter_diagonal_plot(
                    pair_df["baseline_accuracy"].tolist(),
                    pair_df["ai_accuracy"].tolist(),
                    "Participant-level accuracy: baseline vs AI-assisted",
                    out,
                )
                plots["accuracy_scatter"] = f"/static/{RESULTS_DIRNAME}/accuracy_scatter.png"

    if time_by_cond:
        labels, vals = ordered_values(time_by_cond)
        out = os.path.join(results_dir, "time.png")
        _make_bar_plot(labels, vals, "Mean decision time by condition", "Seconds", out)
        plots["time"] = f"/static/{RESULTS_DIRNAME}/time.png"

    if follow_by_cond:
        labels, vals = ordered_values(follow_by_cond)
        out = os.path.join(results_dir, "ai_followed.png")
        _make_bar_plot(labels, vals, "Mean AI-followed rate (AI condition)", "Rate (0–1)", out)
        plots["ai_followed"] = f"/static/{RESULTS_DIRNAME}/ai_followed.png"

    if trust_by_cond:
        labels, vals = ordered_values(trust_by_cond)
        out = os.path.join(results_dir, "trust.png")
        _make_bar_plot(labels, vals, "Mean trust score (AI condition)", "Score (1–5)", out)
        plots["trust"] = f"/static/{RESULTS_DIRNAME}/trust.png"

    if not participant_summary.empty:
        needed = ["trust_score", "ai_ai_followed_rate"]
        if all(c in participant_summary.columns for c in needed):
            scatter_df = participant_summary[needed].dropna()
            if len(scatter_df) >= 3:
                out = os.path.join(results_dir, "trust_vs_ai_followed.png")
                _make_scatter_plot(
                    scatter_df["trust_score"].tolist(),
                    scatter_df["ai_ai_followed_rate"].tolist(),
                    "Trust score vs AI-followed rate",
                    "Trust score (1–5)",
                    "AI-followed rate (0–1)",
                    out,
                    trend_line=True,
                )
                plots["trust_vs_ai_followed"] = f"/static/{RESULTS_DIRNAME}/trust_vs_ai_followed.png"

    if sus_by_cond:
        labels, vals = ordered_values(sus_by_cond)
        out = os.path.join(results_dir, "sus.png")
        _make_bar_plot(labels, vals, "Mean SUS score (AI condition)", "SUS (0–100)", out)
        plots["sus"] = f"/static/{RESULTS_DIRNAME}/sus.png"

    if ai_confidence_by_cond:
        labels, vals = ordered_values(ai_confidence_by_cond)
        out = os.path.join(results_dir, "ai_confidence.png")
        _make_bar_plot(labels, vals, "Mean AI confidence score", "Confidence (0–1)", out)
        plots["ai_confidence"] = f"/static/{RESULTS_DIRNAME}/ai_confidence.png"

    if ai_prob_approve_by_cond:
        labels, vals = ordered_values(ai_prob_approve_by_cond)
        out = os.path.join(results_dir, "ai_prob_approve.png")
        _make_bar_plot(labels, vals, "Mean AI approval probability", "Probability (0–1)", out)
        plots["ai_prob_approve"] = f"/static/{RESULTS_DIRNAME}/ai_prob_approve.png"

    if explanation_open_by_cond:
        labels, vals = ordered_values(explanation_open_by_cond)
        out = os.path.join(results_dir, "explanation_open_rate.png")
        _make_bar_plot(labels, vals, "Mean explanation open rate by condition", "Rate (0–1)", out)
        plots["explanation_open_rate"] = f"/static/{RESULTS_DIRNAME}/explanation_open_rate.png"

    for col, title, fname in [
        ("age_group", "Age group distribution", "age_group_distribution"),
        ("background", "Background distribution", "background_distribution"),
        ("ai_familiarity", "AI familiarity distribution", "ai_familiarity_distribution"),
        ("finance_familiarity", "Finance familiarity distribution", "finance_familiarity_distribution"),
    ]:
        if not participants.empty and col in participants.columns:
            counts = participants[col].fillna("").astype(str).str.strip()
            counts = counts[counts != ""].value_counts()
            if not counts.empty:
                out = os.path.join(results_dir, f"{fname}.png")
                _make_count_plot(counts.index.tolist(), counts.astype(int).tolist(), title, "Count", out)
                plots[fname] = f"/static/{RESULTS_DIRNAME}/{fname}.png"

    paired_tests = {}
    if not participant_summary.empty:
        paired_tests["accuracy"] = _paired_stats(participant_summary, "baseline_accuracy", "ai_accuracy")
        paired_tests["decision_time_seconds"] = _paired_stats(
            participant_summary, "baseline_avg_time_seconds", "ai_avg_time_seconds"
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
        "accuracy_improvement_groups": accuracy_groups,
        "spearman_trust_vs_ai_followed": spearman_trust_followed,
        "ai_followed_distribution": ai_followed_dist,
        "ai_confidence_detail": ai_confidence_detail,
        "ai_prob_approve_detail": ai_prob_detail,
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
