# app/logger.py
# Logs study data to SQLite.

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.db import get_conn, init_db


def _now_utc_iso() -> str:
    # Returns current UTC time in ISO format
    return datetime.now(timezone.utc).isoformat()


# Initializes database tables when the module is imported
init_db()


def log_event(
    participant_id: str,
    condition: str,
    case_id: Optional[Any],
    event: str,
    payload: Dict[str, Any],
):
    # Inserts one row into the events table
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO events (ts_utc, participant_id, condition, case_id, event, payload_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            _now_utc_iso(),
            str(participant_id),
            str(condition),
            None if case_id is None else str(case_id),
            str(event),
            json.dumps(payload, ensure_ascii=False),
        ),
    )
    conn.commit()
    conn.close()


def log_decision(
    participant_id: str,
    condition: str,
    case_id: Any,
    decision: str,
    ground_truth: int,
    correct: int,
    time_ms: Optional[int],
    ai_followed: Optional[int],
    ai_seen: Optional[int],
    explanation_opened: Optional[int],
):
    # Inserts one row into the decisions table
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO decisions (
            ts_utc, participant_id, condition, case_id,
            decision, ground_truth, correct, time_ms,
            ai_followed, ai_seen, explanation_opened
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _now_utc_iso(),
            str(participant_id),
            str(condition),
            str(case_id),
            str(decision),
            int(ground_truth),
            int(correct),
            None if time_ms is None else int(time_ms),
            None if ai_followed is None else int(ai_followed),
            None if ai_seen is None else int(ai_seen),
            None if explanation_opened is None else int(explanation_opened),
        ),
    )
    conn.commit()
    conn.close()


def log_survey(participant_id: str, condition: str, answers: Dict[str, Any]):
    # Inserts one row into the surveys table
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO surveys (ts_utc, participant_id, condition, answers_json)
        VALUES (?, ?, ?, ?)
        """,
        (
            _now_utc_iso(),
            str(participant_id),
            str(condition),
            json.dumps(answers, ensure_ascii=False),
        ),
    )
    conn.commit()
    conn.close()
