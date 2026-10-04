"""Shared test setup. Runs BEFORE any test module is imported.

Safety: tests must never touch your real API keys, logs, or memory database.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

_TMP = Path(tempfile.mkdtemp(prefix="leo-tests-"))
os.environ.update(
    {
        "GROQ_API_KEY": "gsk_test_key_for_unit_tests_only",
        "GEMINI_API_KEY": "AIza_test_key_for_unit_tests_only_0000",
        "LOG_DIR": str(_TMP / "logs"),
        "TRACE_DIR": str(_TMP / "logs" / "traces"),
        "MEMORY_DB_PATH": str(_TMP / "data" / "leo.db"),
        "LOG_LEVEL": "WARNING",
        "DEBUG_MODE": "false",
        "ENABLE_CALCULATOR": "true",
        "ENABLE_WEB_SEARCH": "false",
    }
)

import pytest  # noqa: E402

from leo.config import Settings  # noqa: E402
from leo.crew import LeoSession  # noqa: E402
from leo.fakes import ScriptedRunner  # noqa: E402
from leo.llm_factory import LLMFactory, usage_tracker  # noqa: E402
from leo.memory import MemoryStore  # noqa: E402


# ---------------------------------------------------------------------------
# Builders for valid fake agent outputs
# ---------------------------------------------------------------------------
def _learn(topic: str = "Fractions") -> dict:
    return {"intent": "learn", "topic": topic, "level": "", "reply": ""}


def _explanation(topic: str = "Fractions") -> dict:
    return {
        "topic": topic,
        "explanation": "A fraction shows part of a whole thing, written with two numbers.",
        "key_points": ["Numerator counts the parts", "Denominator is the total parts"],
        "example": "3/4 of a pizza.",
    }


def _quiz(n: int = 4, topic: str = "Fractions") -> dict:
    return {
        "topic": topic,
        "level": "beginner",
        "questions": [
            {
                "id": i + 1,
                "question": f"Question number {i + 1} about {topic}?",
                "options": ["alpha", "beta", "gamma", "delta"],
                "correct_index": i % 4,
                "concept": f"concept{i + 1}",
            }
            for i in range(n)
        ],
    }


def _feedback() -> dict:
    return {
        "overall": "Keep going, you can do this!",
        "tips": [{"concept": "concept1", "tip": "Review this idea once more."}],
    }


def _right(quiz: dict) -> dict[int, int]:
    return {q["id"]: q["correct_index"] for q in quiz["questions"]}


def _wrong(quiz: dict) -> dict[int, int]:
    return {q["id"]: (q["correct_index"] + 1) % 4 for q in quiz["questions"]}


def _script(coordinator=None, explainer=None, quiz_master=None, evaluator=None) -> dict:
    """A default script for a normal session; override any role."""
    return {
        "coordinator": coordinator or [_learn()],
        "explainer": explainer or [_explanation(), _explanation()],
        "quiz_master": quiz_master or [_quiz(4), _quiz(3)],
        "evaluator": evaluator or [_feedback()],
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def data() -> SimpleNamespace:
    """Toolbox of fake agent outputs: data.quiz(4), data.right(q), data.script(...)."""
    return SimpleNamespace(
        learn=_learn,
        explanation=_explanation,
        quiz=_quiz,
        feedback=_feedback,
        right=_right,
        wrong=_wrong,
        script=_script,
    )


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings that use fake keys and temporary folders."""
    return Settings(
        _env_file=None,
        log_dir=tmp_path / "logs",
        trace_dir=tmp_path / "logs" / "traces",
        memory_db_path=tmp_path / "data" / "leo.db",
        max_retries=2,
        max_rpm=60,
    )


@pytest.fixture
def memory(tmp_path) -> MemoryStore:
    """A fresh, empty memory database."""
    return MemoryStore(tmp_path / "memory.db")


@pytest.fixture(autouse=True)
def _reset_usage():
    """Start every test with a clean token counter."""
    usage_tracker.reset()
    yield


@pytest.fixture
def factory(settings) -> LLMFactory:
    """An LLMFactory whose waiting is recorded (not performed) and whose LLMs are fake."""
    sleeps: list[float] = []
    fake = LLMFactory(settings, sleep=sleeps.append)
    fake.build_llm = lambda model, max_tokens: f"fake-llm:{model}"  # no CrewAI, no network
    fake.sleeps = sleeps
    return fake


@pytest.fixture
def make_session(settings, memory):
    """Build a LeoSession wired to a ScriptedRunner: returns (session, runner)."""

    def _make(script=None, name="Tester", custom_settings=None, **kwargs):
        runner = ScriptedRunner(script if script is not None else _script())
        session = LeoSession(
            name,
            settings=custom_settings or settings,
            memory=memory,
            runner=runner,
            **kwargs,
        )
        return session, runner

    return _make
