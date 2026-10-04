"""Memory tests: profile, history, weak areas, explanation cache, corruption recovery."""

import sqlite3

from leo.memory import MAX_HISTORY_PER_STUDENT, MAX_WEAK_AREAS, MemoryStore
from leo.schemas import Explanation, GradeResult, StudentProfile


def grade(score: float, weak: list[str], strong: list[str]) -> GradeResult:
    return GradeResult(
        score=score,
        correct_count=0,
        total=2,
        results=[],
        weak_concepts=weak,
        strong_concepts=strong,
    )


def lesson() -> Explanation:
    return Explanation(
        topic="Fractions",
        key_points=["a", "b"],
        explanation="A fraction shows part of a whole thing.",
    )


def test_new_student_has_no_profile(memory):
    assert memory.get_profile("Ana") is None
    assert "new student" in memory.compact_context("Ana")


def test_remember_topic_creates_and_persists_profile(memory, tmp_path):
    memory.remember_topic("Ana", "Fractions", "intermediate")
    reopened = MemoryStore(tmp_path / "memory.db")  # a "restart"
    profile = reopened.get_profile("  ana ")  # names are case/space-insensitive
    assert profile.topic == "Fractions" and profile.level == "intermediate"


def test_update_after_quiz_merges_weak_areas(memory):
    memory.update_after_quiz("Ana", "Fractions", "beginner", grade(0.5, ["a", "b"], []))
    profile = memory.update_after_quiz("Ana", "Fractions", "beginner", grade(0.5, ["c"], ["a"]))
    assert profile.weak_areas == ["c", "b"]  # new weakness first; mastered "a" removed
    assert profile.sessions == 2 and profile.last_score == 0.5


def test_weak_areas_are_capped(memory):
    many = [f"concept{i}" for i in range(12)]
    profile = memory.update_after_quiz("Ana", "Fractions", "beginner", grade(0.0, many, []))
    assert len(profile.weak_areas) == MAX_WEAK_AREAS


def test_history_is_capped_and_newest_first(memory):
    for i in range(25):
        memory.update_after_quiz("Ana", f"topic{i}", "beginner", grade(0.5, [], []))
    history = memory.get_history("Ana", limit=50)
    assert len(history) == MAX_HISTORY_PER_STUDENT
    assert history[0]["topic"] == "topic24"


def test_forget_student_deletes_everything(memory):
    memory.update_after_quiz("Ana", "Fractions", "beginner", grade(0.5, ["a"], []))
    assert memory.forget_student("Ana") is True
    assert memory.get_profile("Ana") is None and memory.get_history("Ana") == []
    assert memory.forget_student("Ana") is False


def test_save_profile_round_trip(memory):
    memory.save_profile(StudentProfile(name="Ana", topic="X", weak_areas=["w"], last_score=0.8))
    assert memory.get_profile("Ana").weak_areas == ["w"]


def test_cache_hit_miss_and_normalised_key(memory):
    assert memory.get_cached_explanation("Fractions", "beginner") is None
    memory.put_cached_explanation("Fractions", "beginner", lesson())
    assert memory.get_cached_explanation("  FRACTIONS! ", "beginner").topic == "Fractions"
    assert memory.cache_stats() == {"entries": 1, "total_hits": 1}


def test_cache_separates_level_and_style(memory):
    memory.put_cached_explanation("Fractions", "beginner", lesson())
    assert memory.get_cached_explanation("Fractions", "advanced") is None
    assert memory.get_cached_explanation("Fractions", "beginner", "simple") is None


def test_unreadable_cache_entry_is_dropped(memory):
    memory.put_cached_explanation("Fractions", "beginner", lesson())
    with sqlite3.connect(memory.db_path) as conn:
        conn.execute("UPDATE explanation_cache SET payload = 'not json'")
    assert memory.get_cached_explanation("Fractions", "beginner") is None
    assert memory.cache_stats()["entries"] == 0


def test_clear_cache(memory):
    memory.put_cached_explanation("Fractions", "beginner", lesson())
    assert memory.clear_cache() == 1


def test_corrupt_database_file_is_replaced(tmp_path):
    db_path = tmp_path / "leo.db"
    db_path.write_bytes(b"this is definitely not a sqlite database " * 100)
    store = MemoryStore(db_path)  # must not crash
    store.remember_topic("Ana", "Fractions")
    assert store.get_profile("Ana").topic == "Fractions"
    assert list(tmp_path.glob("*.corrupt-*"))  # the bad file was kept aside
