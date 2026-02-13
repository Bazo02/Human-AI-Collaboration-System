# app/main.py
# UPDATED (within-subjects design):
# - Every participant completes 20 cases WITHOUT AI, then 20 cases WITH AI
# - Decision rows are logged with condition=baseline or condition=ai

from __future__ import annotations

import os
import uuid
import time
from typing import Dict, Any, List, Optional

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, jsonify

from ai import get_ai_advice
from logger import log_event, log_decision, log_survey
from analysis import generate_results

from config import (
    SECRET_KEY,
    DATA_PATH,
    CASES_FOR_STUDY_PATH,
    TARGET_COL,
    DROP_COLS_FOR_UI,
    CASES_PER_PARTICIPANT,          # 20 per block
    TOTAL_CASES_PER_PARTICIPANT,    # 40 total
    APPROVAL_THRESHOLD,
    ADMIN_PASSWORD,
)

from db import (
    db_count_rows,
    db_list_participants,
    db_delete_participant,
    db_clear_table,
    db_clear_all,
)

app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.secret_key = SECRET_KEY


FIELD_DESCRIPTIONS: Dict[str, str] = {
    "age": "The applicant’s age in years.",
    "annual_income": "Yearly income (before tax). Higher income may indicate better repayment capacity.",
    "loan_amount": "The amount of money the applicant wants to borrow.",
    "credit_score": "A summary score of credit history. Higher scores generally indicate lower financial risk.",
    "employment_status": "The applicant’s employment situation.",
    "existing_loans_count": "Number of existing loans the applicant already has.",
    "num_dependents": "Number of financially dependent persons.",
}


def _load_cases() -> pd.DataFrame:
    if os.path.exists(CASES_FOR_STUDY_PATH):
        df = pd.read_csv(CASES_FOR_STUDY_PATH)
    else:
        df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    if "case_id" not in df.columns:
        df = df.copy()
        df["case_id"] = range(1, len(df) + 1)

    return df


CASES_DF = _load_cases()


def _pick_cases_for_participant() -> Dict[str, List[Dict[str, Any]]]:
    """
    Sample 40 cases once. First 20 = baseline, next 20 = ai.
    """
    seed = session.get("seed")
    if seed is None:
        seed = int(time.time())
        session["seed"] = seed

    n_needed = TOTAL_CASES_PER_PARTICIPANT
    if len(CASES_DF) < n_needed:
        # (Shouldn't happen if you have 100+, but keeps it robust)
        df_sample = CASES_DF.sample(n=n_needed, random_state=seed, replace=True).reset_index(drop=True)
    else:
        df_sample = CASES_DF.sample(n=n_needed, random_state=seed).reset_index(drop=True)

    cases = df_sample.to_dict(orient="records")

    return {
        "baseline": cases[:CASES_PER_PARTICIPANT],
        "ai": cases[CASES_PER_PARTICIPANT:CASES_PER_PARTICIPANT * 2],
    }


def _ui_case_view(case_row: Dict[str, Any]) -> Dict[str, Any]:
    view = dict(case_row)
    view.pop(TARGET_COL, None)
    view.pop("case_id", None)
    for col in DROP_COLS_FOR_UI:
        view.pop(col, None)
    return view


def _features_for_model(case_row: Dict[str, Any]) -> Dict[str, Any]:
    feats = dict(case_row)
    feats.pop(TARGET_COL, None)
    feats.pop("case_id", None)
    return feats


def _require_admin():
    if not session.get("is_admin"):
        return redirect(url_for("admin_login"))
    return None


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    participant_id = request.form.get("participant_id", "").strip()
    if not participant_id:
        participant_id = f"p_{uuid.uuid4().hex[:8]}"

    session.clear()
    session["participant_id"] = participant_id
    session["block"] = "baseline"   # always start baseline
    session["case_index"] = 0
    session["cases_by_block"] = _pick_cases_for_participant()
    session["started_at"] = time.time()

    log_event(participant_id, "baseline", case_id=None, event="session_start", payload={})
    return redirect(url_for("task"))


@app.route("/transition", methods=["GET"])
def transition():
    participant_id = session.get("participant_id")
    if not participant_id:
        return redirect(url_for("index"))

    # Only show transition if we are now in AI block
    if session.get("block") != "ai":
        return redirect(url_for("task"))

    return render_template("transition.html")


@app.route("/task", methods=["GET"])
def task():
    participant_id = session.get("participant_id")
    block = session.get("block", "baseline")
    cases_by_block = session.get("cases_by_block", {})
    cases = cases_by_block.get(block, [])
    idx = session.get("case_index", 0)

    if not participant_id or not cases:
        return redirect(url_for("index"))

    # Finished current block?
    if idx >= len(cases):
        if block == "baseline":
            session["block"] = "ai"
            session["case_index"] = 0
            log_event(participant_id, "baseline", case_id=None, event="baseline_block_complete", payload={})
            log_event(participant_id, "ai", case_id=None, event="ai_block_start", payload={})
            return redirect(url_for("transition"))
        return redirect(url_for("survey"))

    case_row = cases[idx]
    case_id = case_row.get("case_id")
    case_for_ui = _ui_case_view(case_row)

    ai_payload: Optional[Dict[str, Any]] = None
    if block == "ai":
        ai_payload = get_ai_advice(
            features=_features_for_model(case_row),
            approval_threshold=APPROVAL_THRESHOLD
        )

    log_event(participant_id, block, case_id=case_id, event="case_shown", payload={"index": idx, "block": block})

    return render_template(
        "task.html",
        participant_id=participant_id,
        condition=block,
        case_id=case_id,
        case=case_for_ui,
        ai=ai_payload,
        case_number=idx + 1,
        total_cases=len(cases),  # 20
        field_descriptions=FIELD_DESCRIPTIONS,
    )


@app.route("/submit_decision", methods=["POST"])
def submit_decision():
    participant_id = session.get("participant_id")
    block = session.get("block", "baseline")
    cases_by_block = session.get("cases_by_block", {})
    cases = cases_by_block.get(block, [])
    idx = session.get("case_index", 0)

    if not participant_id or not cases:
        return jsonify({"ok": False, "error": "No active session"}), 400
    if idx >= len(cases):
        return jsonify({"ok": False, "error": "No more cases"}), 400

    payload = request.get_json(force=True) or {}
    case_id_from_client = payload.get("case_id")
    decision = payload.get("decision")
    time_ms = payload.get("time_ms")

    if decision not in ("Approve", "Reject"):
        return jsonify({"ok": False, "error": "Invalid decision"}), 400

    current_case = cases[idx]
    current_case_id = current_case.get("case_id")
    if str(case_id_from_client) != str(current_case_id):
        return jsonify({"ok": False, "error": "Case mismatch"}), 400

    gt = int(current_case.get(TARGET_COL))
    correct = int((decision == "Approve" and gt == 1) or (decision == "Reject" and gt == 0))

    log_decision(
        participant_id=participant_id,
        condition=block,
        case_id=current_case_id,
        decision=decision,
        ground_truth=gt,
        correct=correct,
        time_ms=time_ms,
        ai_followed=payload.get("ai_followed"),
        ai_seen=payload.get("ai_seen"),
        explanation_opened=payload.get("explanation_opened"),
    )

    session["case_index"] = idx + 1

    # Finished this block?
    if session["case_index"] >= len(cases):
        if block == "baseline":
            session["block"] = "ai"
            session["case_index"] = 0
            log_event(participant_id, "baseline", case_id=None, event="baseline_block_complete", payload={})
            log_event(participant_id, "ai", case_id=None, event="ai_block_start", payload={})
            return jsonify({"ok": True, "next": "/transition"})
        log_event(participant_id, "ai", case_id=None, event="ai_block_complete", payload={})
        return jsonify({"ok": True, "next": "/survey"})

    return jsonify({"ok": True, "next": "/task"})


@app.route("/survey", methods=["GET", "POST"])
def survey():
    participant_id = session.get("participant_id")
    if not participant_id:
        return redirect(url_for("index"))

    if request.method == "GET":
        return render_template("survey.html")

    # Store final survey as "ai" (trust questions refer to AI)
    log_survey(participant_id=participant_id, condition="ai", answers=dict(request.form.items()))
    log_event(participant_id, "ai", case_id=None, event="survey_submitted", payload={})
    return redirect(url_for("done"))


@app.route("/done", methods=["GET"])
def done():
    return render_template("done.html")


# ---------------- Admin ----------------

@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    if request.method == "GET":
        return render_template("admin_login.html")

    pw = request.form.get("password", "")
    if pw == ADMIN_PASSWORD:
        session["is_admin"] = True
        return redirect(url_for("admin_dashboard"))

    return render_template("admin_login.html", error="Wrong password")


@app.route("/admin/logout", methods=["GET"])
def admin_logout():
    session.pop("is_admin", None)
    return redirect(url_for("index"))


@app.route("/admin/dashboard", methods=["GET"])
def admin_dashboard():
    r = _require_admin()
    if r:
        return r

    counts = {
        "decisions": db_count_rows("decisions"),
        "events": db_count_rows("events"),
        "surveys": db_count_rows("surveys"),
    }
    participants = db_list_participants()
    return render_template("admin_dashboard.html", counts=counts, participants=participants)


@app.route("/admin/results", methods=["GET"])
def admin_results():
    r = _require_admin()
    if r:
        return r

    results = generate_results(app.static_folder)
    return render_template("results.html", results=results)


@app.route("/admin/clear/<logname>", methods=["POST"])
def admin_clear_one(logname: str):
    r = _require_admin()
    if r:
        return r

    table_map = {"decisions": "decisions", "events": "events", "surveys": "surveys"}
    if logname in table_map:
        db_clear_table(table_map[logname])

    return redirect(url_for("admin_dashboard"))


@app.route("/admin/clear_all", methods=["POST"])
def admin_clear_all_route():
    r = _require_admin()
    if r:
        return r

    db_clear_all()
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/delete_participant", methods=["POST"])
def admin_delete_participant_route():
    r = _require_admin()
    if r:
        return r

    pid = request.form.get("participant_id", "").strip()
    if pid:
        db_delete_participant(pid)

    return redirect(url_for("admin_dashboard"))


if __name__ == "__main__":
    app.run(debug=True)

