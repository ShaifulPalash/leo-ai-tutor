"""End-to-end flow tests with scripted agents: handoffs, loop, caps, memory, HITL."""

import json

from leo.llm_factory import LLMUnavailableError
from leo.schemas import StudentProfile


def start_quiz(session, data):
    session.handle_message("teach me fractions")
    return session.start_quiz()


def test_perfect_score_skips_the_evaluator_llm_and_levels_up(make_session, data):
    session, runner = make_session()
    result = start_quiz(session, data)
    answers = data.right(data.quiz(4))
    result = session.submit_answers(answers)
    assert result.stage == "done" and result.report.passed
    assert runner.calls_for("evaluator") == []  # 0 tokens for feedback
    assert session.memory.get_profile("Tester").level == "intermediate"
    assert any("intermediate" in note for note in result.notes)


def test_handoff_chain_is_coordinator_explainer_quiz_master_evaluator(make_session, data):
    session, _ = make_session()
    start_quiz(session, data)
    session.submit_answers(data.right(data.quiz(4)))
    chain = [(e.agent, e.to_agent) for e in session.events if e.status == "handoff"]
    assert chain == [
        ("coordinator", "explainer"),
        ("explainer", "quiz_master"),
        ("quiz_master", "evaluator"),
    ]


def test_quiz_master_receives_only_topic_and_key_points(make_session, data):
    session, _ = make_session()
    start_quiz(session, data)
    payload = next(
        e.payload for e in session.events if e.status == "handoff" and e.to_agent == "quiz_master"
    )
    assert set(payload) == {"topic", "key_points"}


def test_low_score_triggers_targeted_reteach_and_mini_quiz(make_session, data):
    session, runner = make_session()
    start_quiz(session, data)
    result = session.submit_answers(data.wrong(data.quiz(4)))

    assert result.report.needs_reteach and result.reteaching and result.round == 1
    assert result.stage == "quiz" and len(result.quiz.questions) == 3  # mini re-quiz
    assert len(runner.calls_for("evaluator")) == 1
    assert "concept1" in runner.calls_for("explainer")[-1]  # re-teach targets weak concepts
    assert session.memory.cache_stats()["entries"] == 1  # targeted lessons are NOT cached
    chain = [(e.agent, e.to_agent) for e in session.events if e.status == "handoff"]
    assert ("evaluator", "explainer") in chain


def test_passing_the_mini_quiz_ends_the_loop(make_session, data):
    session, _ = make_session()
    start_quiz(session, data)
    session.submit_answers(data.wrong(data.quiz(4)))
    result = session.submit_answers(data.right(data.quiz(3)))
    assert result.stage == "done" and result.report.passed


def test_reteach_loop_is_capped(make_session, data, settings):
    capped = settings.model_copy(update={"max_relearn_rounds": 1})
    session, _ = make_session(custom_settings=capped)
    start_quiz(session, data)
    first = session.submit_answers(data.wrong(data.quiz(4)))
    second = session.submit_answers(data.wrong(data.quiz(3)))
    assert first.reteaching is True
    assert second.reteaching is False and second.stage == "done"
    assert any("limit" in note for note in second.notes)


def test_threshold_is_configurable(make_session, data, settings):
    strict = settings.model_copy(update={"score_threshold": 0.9})
    session, _ = make_session(custom_settings=strict)
    start_quiz(session, data)
    answers = data.right(data.quiz(4))
    answers[1] = (answers[1] + 1) % 4  # one wrong -> 75%
    assert session.submit_answers(answers).reteaching is True


def test_reteach_failure_keeps_the_report(make_session, data):
    script = data.script(explainer=[data.explanation(), RuntimeError("down")])
    session, _ = make_session(script)
    start_quiz(session, data)
    result = session.submit_answers(data.wrong(data.quiz(4)))
    assert result.report is not None and result.error is False
    assert any("couldn't" in note for note in result.notes)


def test_evaluator_outage_falls_back_to_template_feedback(make_session, data):
    session, _ = make_session(data.script(evaluator=[RuntimeError("down")]))
    start_quiz(session, data)
    result = session.submit_answers(data.wrong(data.quiz(4)))
    assert "scored" in result.report.feedback.overall
    assert session.stats["safe_defaults"] >= 1


def test_submitting_without_a_quiz_is_harmless(make_session):
    session, _ = make_session()
    assert "no active quiz" in session.submit_answers({1: 0}).reply.lower()


def test_malformed_answers_are_ignored(make_session, data):
    session, _ = make_session()
    start_quiz(session, data)
    result = session.submit_answers({"x": "y", 1: "0", 2: None})
    assert result.report is not None


# ---------------------------------------------------------------- memory and cache
def test_same_topic_again_is_served_from_the_cache(make_session):
    session, runner = make_session()
    session.handle_message("teach me fractions")
    session.handle_message("teach me fractions")
    assert len(runner.calls_for("explainer")) == 1
    assert session.stats["cache_hits"] == 1
    assert any(e.status == "cached" for e in session.events)


def test_returning_student_is_greeted_and_weak_areas_guide_the_quiz(make_session, memory):
    memory.save_profile(
        StudentProfile(
            name="Ana",
            topic="Fractions",
            level="beginner",
            weak_areas=["denominator"],
            last_score=0.5,
        )
    )
    session, runner = make_session(name="Ana")
    assert "Welcome back, Ana" in session.greeting()
    session.handle_message("teach me fractions")
    session.start_quiz()
    assert "denominator" in runner.calls_for("quiz_master")[0]


def test_memory_focus_is_not_used_for_a_different_topic(make_session, memory, data):
    memory.save_profile(StudentProfile(name="Ana", topic="Geometry", weak_areas=["angles"]))
    session, runner = make_session(name="Ana")
    session.handle_message("teach me fractions")
    session.start_quiz()
    assert "angles" not in runner.calls_for("quiz_master")[0]


# ---------------------------------------------------------------- human-in-the-loop
def test_explain_simpler_works_without_an_llm_routing_call(make_session):
    session, runner = make_session()
    session.handle_message("teach me fractions")
    result = session.handle_message("explain simpler")
    assert session.style == "simple" and result.stage == "lesson"
    assert len(runner.calls_for("coordinator")) == 1  # the rule handled it
    assert "very simple words" in runner.calls_for("explainer")[-1]


def test_student_can_interrupt_a_quiz(make_session, data):
    session, _ = make_session()
    start_quiz(session, data)
    assert session.stage == "quiz"
    result = session.handle_message("explain simpler")  # mid-quiz intervention
    assert result.stage == "lesson" and session.quiz is None and session.style == "simple"


def test_skip_to_quiz_and_harder_quiz(make_session):
    session, runner = make_session()
    session.handle_message("teach me fractions")
    assert session.handle_message("give me a harder quiz").stage == "quiz"
    prompt = runner.calls_for("quiz_master")[-1]
    assert "Write 5 multiple-choice" in prompt and "harder" in prompt


def test_changing_topic_resets_the_lesson(make_session, data):
    script = data.script(coordinator=[data.learn("Fractions"), data.learn("Photosynthesis")])
    session, _ = make_session(script)
    session.handle_message("teach me fractions")
    session.start_quiz()
    session.handle_message("photosynthesis please")
    assert session.topic == "Photosynthesis" and session.quiz is None
    assert session.stage == "lesson" and session.style == "normal"


# ---------------------------------------------------------------- tracing
def test_every_turn_saves_a_json_trace(make_session):
    session, _ = make_session()
    session.handle_message("teach me fractions")
    saved = json.loads(session.trace.path.read_text(encoding="utf-8"))
    kinds = {entry["kind"] for entry in saved["entries"]}
    assert {"event", "step"} <= kinds and saved["student"] == "Tester"


def test_usage_summary_counts_tokens(make_session, data):
    session, _ = make_session()
    start_quiz(session, data)
    usage = session.usage_summary()
    assert usage["tokens"] > 0 and usage["llm_calls"] == 3


def test_failed_simpler_request_keeps_the_previous_lesson_state(make_session, data):
    script = data.script(explainer=[data.explanation(), RuntimeError("down")])
    session, _ = make_session(script)
    session.handle_message("teach me fractions")
    result = session.handle_message("explain simpler")
    assert result.error is True
    assert session.style == "normal" and session.stage == "lesson"


def test_rejected_keys_show_a_config_hint(make_session, data):
    def boom(prompt):
        raise LLMUnavailableError("keys rejected", kind="config")

    session, _ = make_session(data.script(explainer=[boom]))
    result = session.handle_message("teach me fractions")
    assert result.error and "check_setup" in result.reply
