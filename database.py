"""
database.py
────────────
SQLite-backed storage for EYE.

Deliberately minimal: earlier versions of this file carried a full
user-accounts table with password hashing and login/registration
logic, but eye.py never actually used any of it -- EYE runs as a
single local proctoring tool operated by whoever is running it, not a
multi-user service, so a login system was unused surface area that
only added risk (password handling to get right, a table to keep in
sync with nothing depending on it) for no real benefit. It's gone.
What's left is exactly what the app uses: an optional student roster,
exam sessions, and violations.

Concurrency note
─────────────────
SQLite allows only one writer at a time. In this app, the UI thread
can delete evidence (a write) at any moment while a proctoring session
is actively firing violations (also writes, fairly often) -- two
things writing to the same file close together in time is the normal
case here, not an edge case, so this takes real care to avoid
"database is locked" crashes:
  - WAL (Write-Ahead Logging) journal mode lets readers and a writer
    proceed concurrently instead of blocking each other the way the
    default rollback journal does.
  - Every connection sets a busy_timeout, so if two writes DO land at
    almost the same instant, the second one waits and retries
    automatically instead of raising immediately.
  - A process-local re-entrant lock additionally serialises this
    process's own threads before they even reach SQLite, so two
    threads in the same process never race each other for the
    database handle at all.
  - Every connection is opened, used, and closed within a single
    `with` block (see _connect()) -- nothing holds a connection open
    across calls, which is a common source of stale locks.
"""
import os
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime


class Database:

    def __init__(self, db_path="proctor_system.db"):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._init_database()

    # ── connection handling ───────────────────────────────────────────
    @contextmanager
    def _connect(self):
        """
        One connection per call, always closed, always given a
        busy_timeout so a momentarily-locked database is waited out
        instead of raising. Commits on success, rolls back on any
        exception so a half-finished write can't linger.
        """
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA busy_timeout = 8000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_database(self):
        with self._lock, self._connect() as conn:
            # WAL must be set outside a transaction; harmless to repeat.
            conn.execute("PRAGMA journal_mode = WAL")
            cur = conn.cursor()

            cur.execute('''
                CREATE TABLE IF NOT EXISTS students (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_name TEXT NOT NULL,
                    student_id   TEXT UNIQUE NOT NULL,
                    class_name   TEXT,
                    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cur.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_name   TEXT NOT NULL,
                    camera_source  TEXT,
                    started_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ended_at       TIMESTAMP
                )
            ''')

            cur.execute('''
                CREATE TABLE IF NOT EXISTS violations (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id      INTEGER,
                    student_id      TEXT NOT NULL,
                    violation_type  TEXT NOT NULL,
                    evidence_path   TEXT,
                    timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    datetime_str    TEXT,
                    exam_elapsed    TEXT,
                    confidence      REAL,
                    reason          TEXT,
                    extra_json      TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                )
            ''')

            cur.execute("CREATE INDEX IF NOT EXISTS idx_violations_student "
                        "ON violations(student_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_violations_session "
                        "ON violations(session_id)")

    # ── sessions ─────────────────────────────────────────────────────
    def start_session(self, session_name, camera_source=""):
        """Returns the new session's row id."""
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO sessions (session_name, camera_source) VALUES (?, ?)",
                (session_name, str(camera_source)),
            )
            return cur.lastrowid

    def end_session(self, session_id):
        if session_id is None:
            return
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET ended_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,),
            )

    # ── students (optional roster) ──────────────────────────────────
    def add_student(self, student_name, student_id, class_name=""):
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "INSERT INTO students (student_name, student_id, class_name) "
                    "VALUES (?, ?, ?)",
                    (student_name, student_id, class_name),
                )
            return True, "Student added successfully"
        except sqlite3.IntegrityError:
            return False, "That student ID already exists"
        except Exception as e:
            return False, str(e)

    def get_students(self):
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT student_id, student_name, class_name FROM students "
                "ORDER BY created_at DESC"
            )
            return cur.fetchall()

    # ── violations ───────────────────────────────────────────────────
    def add_violation(self, student_id, violation_type, evidence_path="",
                       session_id=None, datetime_str="", exam_elapsed="",
                       confidence=None, reason="", extra=None):
        """
        Inserts one violation record. `extra` is any small dict of
        additional fields specific to a violation type (paper_type,
        role/partner_id for a paper hand-off, via_pose for a
        gesture-only phone read, ...) -- stored as JSON so the schema
        doesn't need a new column every time a check gains a new
        detail worth recording. Returns (True, new_row_id) or
        (False, error_message).
        """
        try:
            with self._lock, self._connect() as conn:
                cur = conn.cursor()
                cur.execute(
                    '''INSERT INTO violations
                       (session_id, student_id, violation_type, evidence_path,
                        datetime_str, exam_elapsed, confidence, reason, extra_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (session_id, str(student_id), violation_type, evidence_path,
                     datetime_str, exam_elapsed,
                     float(confidence) if confidence is not None else None,
                     reason, json.dumps(extra) if extra else None),
                )
                return True, cur.lastrowid
        except Exception as e:
            return False, str(e)

    def get_violations(self, student_id=None, session_id=None):
        """
        Returns a list of dicts, newest first, each shaped to match
        what eye.py/evidence_viewer.py already expect (id, student_id,
        violation_type, evidence_path, datetime, exam_elapsed,
        confidence, reason), with any `extra` fields (paper_type,
        role, partner_id, ...) merged directly into the dict.
        """
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            query = "SELECT * FROM violations WHERE 1=1"
            params = []
            if student_id is not None:
                query += " AND student_id = ?"
                params.append(str(student_id))
            if session_id is not None:
                query += " AND session_id = ?"
                params.append(session_id)
            query += " ORDER BY timestamp DESC"
            cur.execute(query, params)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()

        out = []
        for row in rows:
            rec = dict(zip(cols, row))
            extra_json = rec.pop("extra_json", None)
            if extra_json:
                try:
                    rec.update(json.loads(extra_json))
                except Exception:
                    pass
            rec["datetime"] = rec.pop("datetime_str", "") or rec.get("timestamp", "")
            out.append(rec)
        return out

    def delete_violation(self, violation_id):
        """Deletes one violation by its row id. Returns True if a row was removed."""
        try:
            with self._lock, self._connect() as conn:
                cur = conn.cursor()
                cur.execute("DELETE FROM violations WHERE id = ?", (violation_id,))
                return cur.rowcount > 0
        except Exception:
            return False

    def clear_violations(self, session_id=None):
        """Deletes ALL violations (optionally scoped to one session)."""
        with self._lock, self._connect() as conn:
            if session_id is not None:
                conn.execute("DELETE FROM violations WHERE session_id = ?", (session_id,))
            else:
                conn.execute("DELETE FROM violations")

    def get_violation_stats(self, session_id=None):
        """{violation_type: count}, optionally scoped to one session."""
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            if session_id is not None:
                cur.execute(
                    "SELECT violation_type, COUNT(*) FROM violations "
                    "WHERE session_id = ? GROUP BY violation_type",
                    (session_id,),
                )
            else:
                cur.execute(
                    "SELECT violation_type, COUNT(*) FROM violations "
                    "GROUP BY violation_type"
                )
            return dict(cur.fetchall())