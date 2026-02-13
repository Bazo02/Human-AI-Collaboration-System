# app/db.py
# Handles SQLite database setup and simple admin operations for the thesis prototype.

from __future__ import annotations

import os
import sqlite3
from typing import Optional, Any, Dict

from app.config import SQLITE_DB_PATH


def _ensure_parent_dir(path: str) -> None:
    # Creates the parent folder if it does not exist
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def get_conn() -> sqlite3.Connection:
    # Returns a SQLite connection and creates the DB file if missing
    _ensure_parent_dir(SQLITE_DB_PATH)
    conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    # Creates tables if they do not already exist
    conn = get_conn()
    cur = conn.cursor()

    # Creates events table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_utc TEXT NOT NULL,
        participant_id TEXT NOT NULL,
        condition TEXT NOT NULL,
        case_id TEXT,
        event TEXT NOT NULL,
        payload_json TEXT
    )
    """)

    # Creates decisions table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_utc TEXT NOT NULL,
        participant_id TEXT NOT NULL,
        condition TEXT NOT NULL,
        case_id TEXT NOT NULL,
        decision TEXT NOT NULL,
        ground_truth INTEGER NOT NULL,
        correct INTEGER NOT NULL,
        time_ms INTEGER,
        ai_followed INTEGER,
        ai_seen INTEGER,
        explanation_opened INTEGER
    )
    """)

    # Creates surveys table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS surveys (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_utc TEXT NOT NULL,
        participant_id TEXT NOT NULL,
        condition TEXT NOT NULL,
        answers_json TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()


# Admin DB functions


def db_count_rows(table: str) -> int:
    # Returns number of rows in a table
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    n = int(cur.fetchone()[0])
    conn.close()
    return n


def db_list_participants() -> list[str]:
    # Returns sorted unique participant IDs from all tables (excluding ADMIN)
    conn = get_conn()
    cur = conn.cursor()

    # Collects participant IDs from all tables
    cur.execute("""
        SELECT DISTINCT participant_id FROM decisions
        UNION
        SELECT DISTINCT participant_id FROM surveys
        UNION
        SELECT DISTINCT participant_id FROM events
    """)
    rows = [r[0] for r in cur.fetchall()]
    conn.close()

    cleaned = sorted([p for p in rows if p and str(p).strip() and str(p) != "ADMIN"])
    return cleaned


def db_delete_participant(participant_id: str) -> dict:
    # Deletes one participant from all tables and returns deleted row counts
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM decisions WHERE participant_id = ?", (participant_id,))
    d_before = int(cur.fetchone()[0])
    cur.execute("DELETE FROM decisions WHERE participant_id = ?", (participant_id,))
    d_deleted = d_before

    cur.execute("SELECT COUNT(*) FROM surveys WHERE participant_id = ?", (participant_id,))
    s_before = int(cur.fetchone()[0])
    cur.execute("DELETE FROM surveys WHERE participant_id = ?", (participant_id,))
    s_deleted = s_before

    cur.execute("SELECT COUNT(*) FROM events WHERE participant_id = ?", (participant_id,))
    e_before = int(cur.fetchone()[0])
    cur.execute("DELETE FROM events WHERE participant_id = ?", (participant_id,))
    e_deleted = e_before

    conn.commit()
    conn.close()

    return {"decisions": d_deleted, "surveys": s_deleted, "events": e_deleted}


def db_clear_table(table: str) -> int:
    # Deletes all rows in one table and returns number of deleted rows
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    before = int(cur.fetchone()[0])
    cur.execute(f"DELETE FROM {table}")
    conn.commit()
    conn.close()
    return before


def db_clear_all() -> dict:
    # Clears all study tables and returns deleted row counts
    return {
        "events": db_clear_table("events"),
        "decisions": db_clear_table("decisions"),
        "surveys": db_clear_table("surveys"),
    }
