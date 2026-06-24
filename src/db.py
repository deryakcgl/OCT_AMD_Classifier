# sqlite for patients and visits
# i chose sqlite because this is a simple local demo i am building for myself
# one file on disk, easy to reset when i want to try things again with seed_data.py

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import date
from typing import Optional

# db file lives in project root not in src/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(ROOT, "oct_clinic.db")


@dataclass
class Patient:
    id: int
    name: str
    created_at: str


@dataclass
class Visit:
    # one row per oct upload
    id: int
    patient_id: int
    visit_date: str
    complaints: str
    image_path: str
    class_label: str
    confidence: float
    probabilities: dict[str, float]  # loaded from json column


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # rows behave like dicts — easier to read
    conn.execute("PRAGMA foreign_keys = ON")  # deleting a patient removes their visits too
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    # run on app start. CREATE IF NOT EXISTS so safe to call many times
    with get_connection(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (date('now'))
            );

            CREATE TABLE IF NOT EXISTS visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                visit_date TEXT NOT NULL,
                complaints TEXT NOT NULL DEFAULT '',
                image_path TEXT NOT NULL,
                class_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                probabilities_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_visits_patient_date
                ON visits(patient_id, visit_date);
            """
        )
    # probabilities stored as json because sqlite has no dict type
    # a separate table would be more normalized but feels unnecessary for this demo


def add_patient(name: str, db_path: str = DEFAULT_DB_PATH) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute("INSERT INTO patients (name) VALUES (?)", (name,))
        return int(cur.lastrowid)


def list_patients(db_path: str = DEFAULT_DB_PATH) -> list[Patient]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT id, name, created_at FROM patients ORDER BY id"
        ).fetchall()
    return [Patient(id=r["id"], name=r["name"], created_at=r["created_at"]) for r in rows]


def get_patient(patient_id: int, db_path: str = DEFAULT_DB_PATH) -> Optional[Patient]:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT id, name, created_at FROM patients WHERE id = ?",
            (patient_id,),
        ).fetchone()
    if row is None:
        return None
    return Patient(id=row["id"], name=row["name"], created_at=row["created_at"])


def add_visit(
    patient_id: int,
    visit_date: str,
    complaints: str,
    image_path: str,
    class_label: str,
    confidence: float,
    probabilities: dict[str, float],
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    # called after onnx inference in app
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO visits
                (patient_id, visit_date, complaints, image_path,
                 class_label, confidence, probabilities_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id,
                visit_date,
                complaints,
                image_path,
                class_label,
                confidence,
                json.dumps(probabilities),
            ),
        )
        return int(cur.lastrowid)


def list_visits(patient_id: int, db_path: str = DEFAULT_DB_PATH) -> list[Visit]:
    # oldest first — visits[0] is baseline, visits[-1] is current for genai
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, patient_id, visit_date, complaints, image_path,
                   class_label, confidence, probabilities_json
            FROM visits
            WHERE patient_id = ?
            ORDER BY visit_date, id
            """,
            (patient_id,),
        ).fetchall()
    return [_row_to_visit(r) for r in rows]


def get_visit(visit_id: int, db_path: str = DEFAULT_DB_PATH) -> Optional[Visit]:
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, patient_id, visit_date, complaints, image_path,
                   class_label, confidence, probabilities_json
            FROM visits WHERE id = ?
            """,
            (visit_id,),
        ).fetchone()
    return _row_to_visit(row) if row else None


def _row_to_visit(row: sqlite3.Row) -> Visit:
    return Visit(
        id=row["id"],
        patient_id=row["patient_id"],
        visit_date=row["visit_date"],
        complaints=row["complaints"],
        image_path=row["image_path"],
        class_label=row["class_label"],
        confidence=row["confidence"],
        probabilities=json.loads(row["probabilities_json"]),
    )


def visit_to_result(visit: Visit):
    # connects db Visit to genai VisitResult
    # import inside function to avoid circular import when modules load
    from genai_compare import VisitResult

    return VisitResult(
        visit_date=visit.visit_date,
        complaints=visit.complaints,
        class_label=visit.class_label,
        confidence=visit.confidence,
        probabilities=visit.probabilities,
    )


def today_iso() -> str:
    # for default date in upload dialog
    return date.today().isoformat()
