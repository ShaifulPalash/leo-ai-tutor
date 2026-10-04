"""Coordinator tests: routing rules, unclear/unsafe input, failure handling."""

import pytest

from leo.crew import BUSY_MESSAGE, quick_route
from leo.schemas import DEFAULT_CLARIFY
from leo.tasks import heuristic_decision


# ---------------------------------------------------------------- rule-based fast path
@pytest.mark.parametrize(
    "message, stage, expected",
    [
        ("", "idle", "clarify"),
        ("   ", "lesson", "clarify"),
        ("explain simpler", "lesson", "simpler"),
        ("this is too hard", "lesson", "simpler"),
        ("I don't understand", "quiz", "simpler"),
        ("skip to the quiz", "lesson", "skip_to_quiz"),
        ("quiz me", "lesson", "skip_to_quiz"),
        ("give me a harder quiz", "lesson", "harder_quiz"),
        ("change topic", "lesson", "clarify"),
    ],
)
def test_quick_route_handles_obvious_requests(message, stage, expected):
    decision = quick_route(message, stage)
    assert decision is not None and decision.intent == expected


@pytest.mark.parametrize(
    "message, stage",
    [
        ("teach me fractions", "idle"),
        ("explain simpler", "idle"),  # nothing to simplify yet
        ("photosynthesis", "lesson"),
        ("can you please explain that in a simpler way because I am lost", "lesson"),
    ],
)
def test_quick_route_defers_everything_else_to_the_llm(message, stage):
    assert quick_route(message, stage) is None


# ---------------------------------------------------------------- safe default
def test_heuristic_treats_short_text_at_start_as_a_topic():
    assert heuristic_decision("idle", "photosynthesis").topic == "photosynthesis"
    assert heuristic_decision("idle", "teach me the water cycle").topic == "the water cycle"


@pytest.mark.parametrize(
    "stage, message",
    [
        ("idle", "I do not really know what I want to do today maybe"),
        ("idle", "!!!???"),
        ("lesson", "photosynthesis"),
    ],
)
def test_heuristic_asks_when_unsure(stage, message):
    assert heuristic_decision(stage, message).intent == "clarify"


# ---------------------------------------------------------------- the session
def test_empty_message_asks_a_question_and_uses_no_llm(make_session):
    session, runner = make_session()
    result = session.handle_message("")
    assert result.reply == DEFAULT_CLARIFY and result.stage == "idle"
    assert runner.calls == []


def test_unsafe_request_is_refused_politely(make_session, data):
    refuse = {
        "intent": "refuse",
        "topic": "",
        "level": "",
        "reply": "I can only help with study topics. Want to learn something?",
    }
    session, runner = make_session(data.script(coordinator=[refuse]))
    result = session.handle_message("how do I hack my neighbour's wifi")
    assert "study" in result.reply and result.stage == "idle"
    assert runner.calls_for("explainer") == []  # no specialist was bothered


def test_learn_request_hands_off_to_the_explainer(make_session):
    session, runner = make_session()
    result = session.handle_message("teach me fractions")
    assert result.stage == "lesson" and session.topic == "Fractions"
    handoffs = [(e.agent, e.to_agent) for e in session.events if e.status == "handoff"]
    assert handoffs == [("coordinator", "explainer")]
    assert session.memory.get_profile("Tester").topic == "Fractions"


def test_coordinator_failure_uses_safe_default(make_session, data):
    session, runner = make_session(data.script(coordinator=[RuntimeError("boom")]))
    result = session.handle_message("photosynthesis")
    assert session.topic == "photosynthesis" and result.stage == "lesson"
    assert session.stats["safe_defaults"] == 1


def test_coordinator_failure_with_vague_message_asks_for_a_topic(make_session, data):
    session, _ = make_session(data.script(coordinator=[RuntimeError("boom")]))
    result = session.handle_message("I do not really know what I want to do today maybe")
    assert "topic" in result.reply.lower() and result.stage == "idle"


def test_specialist_outage_gives_a_friendly_message_not_a_crash(make_session, data):
    session, _ = make_session(data.script(explainer=[RuntimeError("down")]))
    result = session.handle_message("teach me fractions")
    assert result.error is True and result.reply == BUSY_MESSAGE
    assert result.stage == "idle"  # the session is intact and can try again


def test_quiz_before_any_topic_asks_for_a_topic(make_session):
    session, _ = make_session()
    assert "pick a topic" in session.start_quiz().reply.lower()


def test_a_broken_ui_listener_cannot_break_tutoring(make_session):
    session, _ = make_session(on_event=lambda event: 1 / 0)
    assert session.handle_message("teach me fractions").stage == "lesson"
