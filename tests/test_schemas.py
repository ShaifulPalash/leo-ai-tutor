"""Schema tests: messy LLM output is cleaned, broken output is rejected, grading is exact."""

import pytest
from pydantic import ValidationError

from leo.schemas import (
    DEFAULT_CLARIFY,
    DEFAULT_REFUSE,
    EvaluationReport,
    Explanation,
    GradeResult,
    Quiz,
    RoutingDecision,
    StudentProfile,
    default_feedback,
    grade_quiz,
    normalize_topic,
)


def messy_quiz() -> dict:
    return {
        "topic": "Fractions",
        "level": "Beginner ",
        "questions": [
            {
                "id": 7,
                "question": "Which number is the numerator in 3/4?",
                "options": ["A) 4", "B) 3", "C) 7", "D) 1"],
                "correct_index": "B",
                "concept": "numerator",
            },
            {
                "id": 7,
                "question": "What is 2/4 in simplest form?",
                "options": ["1/2", "2/2", "4/2", "1/4"],
                "correct_index": 0,
                "concept": "simplifying",
            },
            {
                "id": 7,
                "question": "Which fraction is the largest?",
                "options": ["1/8", "1/2", "1/4", "1/3"],
                "correct_index": 1,
                "concept": "comparing",
                "extra_field": "ignored",
            },
        ],
    }


def test_messy_quiz_is_cleaned():
    quiz = Quiz.model_validate(messy_quiz())
    assert quiz.level == "beginner"
    assert [q.id for q in quiz.questions] == [1, 2, 3]  # duplicate ids renumbered
    assert quiz.questions[0].options == ["4", "3", "7", "1"]  # "A) " labels removed
    assert quiz.questions[0].correct_index == 1  # letter "B" -> index 1


def test_quiz_with_three_options_is_rejected():
    bad = {
        "questions": [
            {
                "id": 1,
                "question": "Only three options?",
                "options": ["a", "b", "c"],
                "correct_index": 0,
                "concept": "x",
            }
        ]
        * 2
    }
    with pytest.raises(ValidationError):
        Quiz.model_validate(bad)


def test_quiz_with_duplicate_options_is_rejected():
    bad = messy_quiz()
    bad["questions"][1]["options"] = ["same", "same", "x", "y"]
    with pytest.raises(ValidationError):
        Quiz.model_validate(bad)


def test_quiz_needs_at_least_two_questions():
    bad = messy_quiz()
    bad["questions"] = bad["questions"][:1]
    with pytest.raises(ValidationError):
        Quiz.model_validate(bad)


def test_public_view_hides_the_answers():
    view = Quiz.model_validate(messy_quiz()).public_view()
    assert all("correct_index" not in q for q in view)
    assert set(view[0]) == {"id", "question", "options"}


def test_grading_scores_weak_and_strong_concepts():
    quiz = Quiz.model_validate(messy_quiz())
    grade = grade_quiz(quiz, {1: 1, 2: 3, 3: 1})  # right, wrong, right
    assert grade.score == 0.67
    assert (grade.correct_count, grade.total) == (2, 3)
    assert grade.weak_concepts == ["simplifying"]
    assert grade.strong_concepts == ["numerator", "comparing"]


def test_unanswered_and_invalid_choices_count_as_wrong():
    quiz = Quiz.model_validate(messy_quiz())
    grade = grade_quiz(quiz, {1: 9, 2: -5})  # out of range; question 3 missing
    assert grade.score == 0.0
    assert all(r.chosen_index == -1 for r in grade.results)


def test_report_flags_reteach_below_threshold():
    quiz = Quiz.model_validate(messy_quiz())
    grade = grade_quiz(quiz, {1: 1, 2: 3, 3: 1})
    below = EvaluationReport.from_parts(grade, default_feedback(grade), threshold=0.7)
    at_limit = EvaluationReport.from_parts(grade, default_feedback(grade), threshold=0.67)
    assert below.needs_reteach is True and below.passed is False
    assert at_limit.needs_reteach is False  # exactly at the threshold passes


def test_default_feedback_mentions_weak_concepts():
    grade = GradeResult(
        score=0.5,
        correct_count=1,
        total=2,
        results=[],
        weak_concepts=["simplifying"],
        strong_concepts=[],
    )
    note = default_feedback(grade)
    assert "50%" in note.overall and note.tips[0].concept == "simplifying"


def test_routing_normalises_intent_and_level():
    decision = RoutingDecision(intent="Skip to Quiz", level="Intermediate ")
    assert decision.intent == "skip_to_quiz" and decision.level == "intermediate"
    assert RoutingDecision(intent="simpler", level="expert").level is None


def test_learn_without_topic_becomes_clarify():
    decision = RoutingDecision.model_validate({"intent": "learn", "topic": ""})
    assert decision.intent == "clarify" and decision.reply == DEFAULT_CLARIFY


def test_refuse_gets_a_default_reply():
    assert RoutingDecision(intent="refuse").reply == DEFAULT_REFUSE


def test_unknown_intent_is_rejected():
    with pytest.raises(ValidationError):
        RoutingDecision(intent="hack_the_planet")


def test_explanation_rules():
    with pytest.raises(ValidationError):  # explanation too short
        Explanation(topic="x", explanation="short", key_points=["a", "b"])
    with pytest.raises(ValidationError):  # needs at least 2 key points
        Explanation(topic="x", explanation="A long enough explanation here.", key_points=["a"])


def test_long_explanation_is_trimmed_and_brief_is_small():
    long_text = "word " * 500
    lesson = Explanation(topic="x", explanation=long_text, key_points=["a", "b"])
    assert lesson.word_count <= 351
    assert set(lesson.to_quiz_brief()) == {"topic", "key_points"}  # compact handoff


def test_profile_name_and_summary():
    assert StudentProfile(name="   ").name == "Student"
    profile = StudentProfile(
        name="Ana", topic="Fractions", weak_areas=["numerator"], last_score=0.5
    )
    assert "50%" in profile.summary() and "numerator" in profile.summary()


def test_normalize_topic():
    assert normalize_topic("  Fractions!  ") == "fractions"
    assert normalize_topic("The   Water  Cycle.") == "the water cycle"
