"""Leo's persistent memory (SQLite): student profile, history, explanation cache.

Stores compact facts only, never raw chat transcripts (saves tokens and
protects privacy). Run `python -m leo.memory` for a self-demo that uses a
temporary database and leaves your real data untouched.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from leo.config import get_settings
from leo.logging_setup import get_logger
from leo.schemas import Explanation, GradeResult, Level, StudentProfile, clean_text, normalize_topic

logger = get_logger("memory")

MAX_WEAK_AREAS = 8
MAX_HISTORY_PER_STUDENT = 20


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _name_key(name: str) -> str:
    """'  Palash  ' and 'palash' are the same student."""
    return re.sub(r"\s+", " ", (name or "").strip().lower())[:40] or "student"


class MemoryStore:
    """Small SQLite-backed store. Create one instance and reuse it."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else get_settings().memory_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ---------------- internals ----------------
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection, commit on success, roll back on error, always close."""
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _create_tables(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS students (
                    name_key TEXT PRIMARY KEY, display_name TEXT NOT NULL,
                    topic TEXT DEFAULT '', level TEXT DEFAULT 'beginner',
                    weak_areas TEXT DEFAULT '[]', last_score REAL,
                    sessions INTEGER DEFAULT 0, updated_at TEXT);
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, name_key TEXT NOT NULL,
                    topic TEXT, level TEXT, score REAL, weak_areas TEXT, ts TEXT);
                CREATE TABLE IF NOT EXISTS explanation_cache (
                    cache_key TEXT PRIMARY KEY, topic TEXT, level TEXT, style TEXT,
                    payload TEXT NOT NULL, created_at TEXT, hits INTEGER DEFAULT 0);
            """)

    def _init_db(self) -> None:
        """Create tables; if the file is corrupt, move it aside and start fresh."""
        try:
            self._create_tables()
        except sqlite3.DatabaseError as exc:
            if not self.db_path.exists():
                raise
            backup = self.db_path.with_suffix(f".corrupt-{int(time.time())}")
            logger.error("Memory DB unreadable (%s). Moving it to %s", exc, backup.name)
            self.db_path.rename(backup)
            self._create_tables()

    @staticmethod
    def _row_to_profile(row: sqlite3.Row) -> StudentProfile:
        try:
            weak = json.loads(row["weak_areas"] or "[]")
        except json.JSONDecodeError:
            weak = []
        return StudentProfile(
            name=row["display_name"],
            topic=row["topic"] or "",
            level=row["level"] or "beginner",
            weak_areas=weak,
            last_score=row["last_score"],
            sessions=row["sessions"] or 0,
            updated_at=row["updated_at"] or "",
        )

    # ---------------- student profile ----------------
    def get_profile(self, name: str) -> Optional[StudentProfile]:
        """Return the saved profile, or None for a new student."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM students WHERE name_key = ?", (_name_key(name),)
            ).fetchone()
        return self._row_to_profile(row) if row else None

    def save_profile(self, profile: StudentProfile) -> None:
        """Insert or update the student's row."""
        profile.updated_at = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO students (name_key, display_name, topic, level, weak_areas,
                                      last_score, sessions, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name_key) DO UPDATE SET
                    display_name=excluded.display_name, topic=excluded.topic,
                    level=excluded.level, weak_areas=excluded.weak_areas,
                    last_score=excluded.last_score, sessions=excluded.sessions,
                    updated_at=excluded.updated_at""",
                (
                    _name_key(profile.name),
                    profile.name,
                    profile.topic,
                    profile.level,
                    json.dumps(profile.weak_areas),
                    profile.last_score,
                    profile.sessions,
                    profile.updated_at,
                ),
            )

    def remember_topic(
        self, name: str, topic: str, level: Optional[Level] = None
    ) -> StudentProfile:
        """Create the student if new, then record the current topic (and level)."""
        profile = self.get_profile(name) or StudentProfile(name=name)
        profile.topic = clean_text(topic, 80)
        if level:
            profile.level = level
        self.save_profile(profile)
        return profile

    def update_after_quiz(
        self, name: str, topic: str, level: Level, grade: GradeResult
    ) -> StudentProfile:
        """Record a finished quiz: update weak areas, last score, session count, history."""
        profile = self.get_profile(name) or StudentProfile(name=name)
        strong = {c.lower() for c in grade.strong_concepts}
        merged = list(grade.weak_concepts)  # newest weaknesses first
        for old in profile.weak_areas:
            if old.lower() not in strong and old.lower() not in {m.lower() for m in merged}:
                merged.append(old)
        profile.weak_areas = merged[:MAX_WEAK_AREAS]
        profile.topic, profile.level = clean_text(topic, 80), level
        profile.last_score = grade.score
        profile.sessions += 1
        self.save_profile(profile)

        key = _name_key(name)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO history (name_key, topic, level, score, weak_areas, ts) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (key, profile.topic, level, grade.score, json.dumps(grade.weak_concepts), _now()),
            )
            # Keep only the newest MAX_HISTORY_PER_STUDENT rows for this student.
            conn.execute(
                "DELETE FROM history WHERE name_key = ? AND id NOT IN "
                "(SELECT id FROM history WHERE name_key = ? ORDER BY id DESC LIMIT ?)",
                (key, key, MAX_HISTORY_PER_STUDENT),
            )
        logger.info(
            "Memory updated for %s: score=%.0f%% weak=%s",
            profile.name,
            grade.score * 100,
            profile.weak_areas,
        )
        return profile

    def get_history(self, name: str, limit: int = 5) -> list[dict]:
        """Newest-first list of past quiz results."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT topic, level, score, weak_areas, ts FROM history "
                "WHERE name_key = ? ORDER BY id DESC LIMIT ?",
                (_name_key(name), limit),
            ).fetchall()
        return [
            {
                "topic": r["topic"],
                "level": r["level"],
                "score": r["score"],
                "weak_areas": json.loads(r["weak_areas"] or "[]"),
                "ts": r["ts"],
            }
            for r in rows
        ]

    def compact_context(self, name: str) -> str:
        """One short line describing the student, cheap to put into prompts."""
        profile = self.get_profile(name)
        return profile.summary() if profile else f"{name} | new student"

    def forget_student(self, name: str) -> bool:
        """Delete everything stored about a student (privacy). True if anything existed."""
        key = _name_key(name)
        with self._connect() as conn:
            conn.execute("DELETE FROM history WHERE name_key = ?", (key,))
            deleted = conn.execute("DELETE FROM students WHERE name_key = ?", (key,)).rowcount
        return deleted > 0

    # ---------------- explanation cache ----------------
    @staticmethod
    def _cache_key(topic: str, level: str, style: str) -> str:
        return f"{normalize_topic(topic)}|{level}|{style}"

    def get_cached_explanation(
        self, topic: str, level: str, style: str = "normal"
    ) -> Optional[Explanation]:
        """Return a saved lesson for (topic, level, style), or None on a cache miss."""
        key = self._cache_key(topic, level, style)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM explanation_cache WHERE cache_key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            try:
                explanation = Explanation.model_validate_json(row["payload"])
            except ValueError:  # includes pydantic.ValidationError
                logger.warning("Dropping unreadable cache entry %s", key)
                conn.execute("DELETE FROM explanation_cache WHERE cache_key = ?", (key,))
                return None
            conn.execute("UPDATE explanation_cache SET hits = hits + 1 WHERE cache_key = ?", (key,))
        logger.info("Explanation cache HIT for %s", key)
        return explanation

    def put_cached_explanation(
        self, topic: str, level: str, explanation: Explanation, style: str = "normal"
    ) -> None:
        """Save a lesson so the same request costs 0 Explainer tokens next time."""
        key = self._cache_key(topic, level, style)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO explanation_cache "
                "(cache_key, topic, level, style, payload, created_at, hits) "
                "VALUES (?, ?, ?, ?, ?, ?, 0)",
                (key, normalize_topic(topic), level, style, explanation.model_dump_json(), _now()),
            )

    def cache_stats(self) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(hits), 0) AS h " "FROM explanation_cache"
            ).fetchone()
        return {"entries": row["n"], "total_hits": row["h"]}

    def clear_cache(self) -> int:
        """Delete every cached lesson; returns how many were removed."""
        with self._connect() as conn:
            return conn.execute("DELETE FROM explanation_cache").rowcount


_store: Optional[MemoryStore] = None


def get_memory() -> MemoryStore:
    """Shared MemoryStore using the path from settings."""
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store


if __name__ == "__main__":
    import tempfile

    from leo.schemas import Quiz, grade_quiz

    with tempfile.TemporaryDirectory() as tmp:
        memory = MemoryStore(Path(tmp) / "demo.db")
        print("New student  :", memory.compact_context("Palash"))

        memory.remember_topic("Palash", "Fractions", "beginner")
        print("After topic  :", memory.compact_context("Palash"))

        quiz = Quiz.model_validate(
            {
                "topic": "Fractions",
                "questions": [
                    {
                        "id": 1,
                        "question": "Numerator of 3/4?",
                        "options": ["4", "3", "7", "1"],
                        "correct_index": 1,
                        "concept": "numerator",
                    },
                    {
                        "id": 2,
                        "question": "Simplify 2/4.",
                        "options": ["1/2", "2/2", "4/2", "1/4"],
                        "correct_index": 0,
                        "concept": "simplifying",
                    },
                ],
            }
        )
        profile = memory.update_after_quiz(
            "palash", "Fractions", "beginner", grade_quiz(quiz, {1: 1, 2: 3})
        )
        print("After quiz   :", profile.summary())
        print("History      :", memory.get_history("Palash"))

        lesson = Explanation(
            topic="Fractions",
            key_points=["a", "b"],
            explanation="A fraction shows part of a whole thing.",
        )
        print("Cache (miss) :", memory.get_cached_explanation("Fractions", "beginner"))
        memory.put_cached_explanation("Fractions", "beginner", lesson)
        hit = memory.get_cached_explanation("  fractions! ", "beginner")
        print("Cache (hit)  :", hit.topic if hit else None, memory.cache_stats())
        print("Simple style :", memory.get_cached_explanation("Fractions", "beginner", "simple"))
        print(
            "Forgotten    :",
            memory.forget_student("Palash"),
            "->",
            memory.compact_context("Palash"),
        )
