"""Leo's orchestrator: a sequential pipeline steered by the Coordinator.

Public API (used by the CLI and Streamlit UI):
    session = LeoSession("Palash", on_event=callback)
    session.handle_message("teach me fractions")   # Coordinator -> Explainer
    session.start_quiz()                            # Explainer -> Quiz Master
    session.submit_answers({1: 0, 2: 3})            # Quiz Master -> Evaluator (-> loop)

Run:  python -m leo.crew --demo            offline scripted session, no API calls
      python -m leo.crew --live fractions  real LLM session (about 3-4 LLM calls)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

from leo.config import Settings, get_settings
from leo.llm_factory import LLMFactory, LLMUnavailableError
from leo.logging_setup import get_agent_logger, get_logger, log_handoff, redact
from leo.memory import MemoryStore, get_memory
from leo.schemas import (
    DEFAULT_CLARIFY,
    EvaluationReport,
    Explanation,
    FeedbackNote,
    Quiz,
    RoutingDecision,
    StudentProfile,
    clean_text,
    default_feedback,
    grade_quiz,
    normalize_topic,
)
from leo.tasks import (
    AgentRun,
    AgentRunner,
    coordinator_prompt,
    evaluator_prompt,
    explainer_prompt,
    heuristic_decision,
    quiz_prompt,
    run_agent,
)
from leo.tracing import RunTrace

logger = get_logger("crew")

QUIZ_SIZE = {"normal": 4, "harder": 5, "mini": 3}  # questions per quiz type
NEXT_LEVEL = {"beginner": "intermediate", "intermediate": "advanced", "advanced": "advanced"}

BUSY_MESSAGE = (
    "My teaching team is busy right now (free-tier limits). Please wait about "
    "a minute and try again. Your progress is saved."
)
CONFIG_MESSAGE = (
    "I can't reach my AI models. The API keys or model names in .env look "
    "wrong. Run `python scripts/check_setup.py` to find out which."
)
UNEXPECTED_MESSAGE = (
    "Something unexpected happened on my side. Your progress is saved. Please try again."
)


# ----------------------------------------------------------------------
# Data passed to the UI
# ----------------------------------------------------------------------
@dataclass
class ActivityEvent:
    """One line in the live Agent Activity panel."""

    time: str
    agent: str  # coordinator | explainer | quiz_master | evaluator | system
    status: str  # working | done | cached | handoff | info | failed
    detail: str
    model: str = ""
    tokens: int = 0
    duration_s: float = 0.0
    to_agent: str = ""  # only for handoffs
    payload: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TurnResult:
    """What a session method returns: the current situation, ready to display."""

    stage: str  # idle | lesson | quiz | done
    reply: str = ""  # Coordinator's plain message, if any
    explanation: Optional[Explanation] = None
    quiz: Optional[Quiz] = None
    report: Optional[EvaluationReport] = None  # set on the turn answers were graded
    reteaching: bool = False  # True when the feedback loop started
    round: int = 0  # re-teach rounds used so far
    error: bool = False
    notes: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# Rule-based fast path of the Coordinator (0 tokens)
# ----------------------------------------------------------------------
_RE_CHANGE = re.compile(r"^(please\s+)?(change|new|different|another)\s+topic[.!]?$", re.I)
_RE_HARDER = re.compile(r"\b(harder|tougher|more difficult|challenge me)\b", re.I)
_RE_SIMPLER = re.compile(
    r"\b(simpler|simplify|easier|eli5|too (hard|difficult|complex|complicated)|don'?t understand)\b",
    re.I,
)
_RE_SKIP = re.compile(
    r"\b(skip|quiz me|start (the )?quiz|go to (the )?quiz|take (the )?quiz|ready for (the )?quiz)\b",
    re.I,
)


def quick_route(message: str, stage: str) -> Optional[RoutingDecision]:
    """Decide obvious cases without an LLM call; return None if unsure.

    Rules only apply once a session is under way and the message is short, so a
    real topic like 'why is the sky hard to describe' still goes to the LLM.
    """
    text = (message or "").strip()
    if not text:
        return RoutingDecision(intent="clarify", reply=DEFAULT_CLARIFY)
    if stage == "idle" or len(text.split()) > 8:
        return None
    if _RE_CHANGE.match(text):
        return RoutingDecision(
            intent="clarify", reply="Sure! Which topic would you like to study next?"
        )
    if _RE_HARDER.search(text):
        return RoutingDecision(intent="harder_quiz")
    if _RE_SIMPLER.search(text):
        return RoutingDecision(intent="simpler")
    if _RE_SKIP.search(text):
        return RoutingDecision(intent="skip_to_quiz")
    return None


# ----------------------------------------------------------------------
# Graceful-failure decorator
# ----------------------------------------------------------------------
def graceful(method: Callable[..., TurnResult]) -> Callable[..., TurnResult]:
    """Turn ANY failure into a friendly message so the app never crashes.

    The session keeps its stage, so the student can simply try again.
    """

    @wraps(method)
    def wrapper(self: "LeoSession", *args: Any, **kwargs: Any) -> TurnResult:
        try:
            return method(self, *args, **kwargs)
        except LLMUnavailableError as exc:
            logger.error("All models unavailable in %s: %s", method.__name__, redact(str(exc)))
            self._emit(
                "coordinator",
                "info",
                "A specialist did not respond; sending a safe message instead",
            )
            reply = CONFIG_MESSAGE if getattr(exc, "kind", "") == "config" else BUSY_MESSAGE
            return self.snapshot(reply=reply, error=True)
        except Exception as exc:  # noqa: BLE001 - last line of defence
            logger.exception("Unexpected error in %s", method.__name__)
            self._emit("system", "failed", f"{type(exc).__name__}: {redact(str(exc))[:120]}")
            return self.snapshot(reply=UNEXPECTED_MESSAGE, error=True)
        finally:
            self.trace.save()

    return wrapper


# ----------------------------------------------------------------------
# The session / orchestrator
# ----------------------------------------------------------------------
class LeoSession:
    """One student's tutoring session, driven by the Coordinator's routing."""

    def __init__(
        self,
        student_name: str = "Student",
        *,
        settings: Settings | None = None,
        memory: MemoryStore | None = None,
        factory: LLMFactory | None = None,
        runner: AgentRunner = run_agent,  # replaceable in tests
        on_event: Callable[[ActivityEvent], None] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.memory = memory or get_memory()
        self.factory = factory
        self.runner = runner
        self.on_event = on_event

        self.student = clean_text(student_name, 40) or "Student"
        existing = self.memory.get_profile(self.student)
        self.returning = existing is not None
        self.profile: StudentProfile = existing or StudentProfile(name=self.student)

        # Conversation state
        self.stage = "idle"  # idle | lesson | quiz | done
        self.topic = ""
        self.level = self.profile.level
        self.style = "normal"  # normal | simple
        self.explanation: Optional[Explanation] = None
        self.quiz: Optional[Quiz] = None
        self.last_report: Optional[EvaluationReport] = None
        self.relearn_round = 0
        self.memory_focus: list[str] = []  # weak concepts remembered for this topic

        # Bookkeeping
        self.events: list[ActivityEvent] = []
        self.trace = RunTrace(self.student, self.settings)
        self._last = None  # most recent LLMResult (for handoff stats)
        self.stats: dict[str, Any] = {
            "llm_calls": 0,
            "tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_s": 0.0,
            "cache_hits": 0,
            "fallbacks": 0,
            "safe_defaults": 0,
        }

    # ---------------- small helpers ----------------
    def greeting(self) -> str:
        """Welcome message that uses what memory knows about the student."""
        if not self.returning:
            return f"Hi {self.student}! I'm Leo. Tell me a topic you'd like to learn."
        profile = self.profile
        text = f"Welcome back, {profile.name}!"
        if profile.topic:
            text += f" Last time: {profile.topic}"
            if profile.last_score is not None:
                text += f" ({round(profile.last_score * 100)}%)"
        if profile.weak_areas:
            text += f". Still to practise: {', '.join(profile.weak_areas[:3])}"
        return text + ". What would you like to learn today?"

    def profile_summary(self) -> str:
        return self.memory.compact_context(self.student)

    def usage_summary(self) -> dict[str, Any]:
        summary = dict(self.stats)
        summary["latency_s"] = round(summary["latency_s"], 2)
        summary["trace_file"] = str(self.trace.path)
        return summary

    def snapshot(
        self,
        reply: str = "",
        report: Optional[EvaluationReport] = None,
        reteaching: bool = False,
        error: bool = False,
        notes: Optional[list[str]] = None,
    ) -> TurnResult:
        """Package the current state for the UI."""
        return TurnResult(
            stage=self.stage,
            reply=reply,
            explanation=self.explanation if self.stage in ("lesson", "quiz") else None,
            quiz=self.quiz if self.stage == "quiz" else None,
            report=report,
            reteaching=reteaching,
            round=self.relearn_round,
            error=error,
            notes=notes or [],
        )

    def _emit(
        self,
        agent: str,
        status: str,
        detail: str,
        *,
        run: Optional[AgentRun] = None,
        to_agent: str = "",
        payload: Optional[dict] = None,
    ) -> None:
        """Record an activity event, log it, and tell the UI (never raises)."""
        result = run.result if run else None
        event = ActivityEvent(
            time=datetime.now().strftime("%H:%M:%S"),
            agent=agent,
            status=status,
            detail=detail,
            model=result.model if result else "",
            tokens=result.total_tokens if result else 0,
            duration_s=round(result.latency_s, 2) if result else 0.0,
            to_agent=to_agent,
            payload=payload or {},
        )
        self.events.append(event)
        self.trace.add("event", **event.as_dict())
        get_agent_logger(agent).info("%s: %s", status, detail)
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:  # noqa: BLE001 - a UI bug must not break tutoring
                logger.exception("on_event listener failed")

    def _handoff(self, from_agent: str, to_agent: str, payload: dict) -> None:
        """Pass work to the next agent: write the structured log AND show it in the UI."""
        last = self._last
        log_handoff(
            from_agent,
            to_agent,
            payload,
            duration_s=last.latency_s if last else 0.0,
            tokens=last.total_tokens if last else 0,
        )
        self._emit(
            from_agent, "handoff", f"hands work to {to_agent}", to_agent=to_agent, payload=payload
        )

    def _run(
        self,
        role: str,
        prompt: str,
        model_cls: type,
        safe_default: Optional[Callable[[], Any]] = None,
    ) -> AgentRun:
        """Run one agent step through the (replaceable) runner and record its cost."""
        run = self.runner(
            role,
            prompt,
            model_cls,
            factory=self.factory,
            safe_default=safe_default,
            settings=self.settings,
        )
        result = run.result
        self.stats["llm_calls"] += result.attempts
        self.stats["tokens"] += result.total_tokens
        self.stats["prompt_tokens"] += result.prompt_tokens
        self.stats["completion_tokens"] += result.completion_tokens
        self.stats["latency_s"] += result.latency_s
        if result.used_safe_default:
            self.stats["safe_defaults"] += 1
        elif result.used_fallback_model:
            self.stats["fallbacks"] += 1
        self._last = result

        step: dict[str, Any] = {
            "agent": result.agent,
            "model": result.model,
            "attempts": result.attempts,
            "tokens": result.total_tokens,
            "latency_s": round(result.latency_s, 2),
            "fallback": result.used_fallback_model,
            "safe_default": result.used_safe_default,
        }
        if self.settings.debug_mode:  # prompts/outputs only saved in debug mode
            step["prompt"] = redact(run.prompt[:1500])
            step["raw_output"] = redact(run.raw[:1500])
        self.trace.add("step", **step)
        return run

    # ---------------- Coordinator ----------------
    def _route(self, message: str) -> RoutingDecision:
        """Coordinator step: rules first (0 tokens), LLM only when unsure."""
        quick = quick_route(message, self.stage)
        if quick is not None:
            self._last = None
            self._emit(
                "coordinator", "done", f"intent={quick.intent} (rule-based fast path, 0 tokens)"
            )
            return quick

        prompt = coordinator_prompt(self.stage, self.topic, self.level, message)
        self._emit("coordinator", "working", "Reading your message and choosing who acts next")
        run = self._run(
            "coordinator",
            prompt,
            RoutingDecision,
            safe_default=lambda: heuristic_decision(self.stage, message),
        )
        decision: RoutingDecision = run.result.value
        detail = f"intent={decision.intent}"
        if decision.topic:
            detail += f", topic='{decision.topic}'"
        if run.result.used_safe_default:
            detail += " (LLM unavailable: used safe rule-based default)"
        self._emit("coordinator", "done", detail, run=run)
        return decision

    def _begin_topic(self, topic: str, level: Optional[str]) -> None:
        """Reset the lesson state for a new topic and consult memory."""
        stored = self.memory.get_profile(self.student)
        self.topic = clean_text(topic, 80)
        self.level = level or (stored.level if stored else "beginner")
        # Remember weak concepts ONLY if the student studied this same topic before.
        same_topic = bool(stored and normalize_topic(stored.topic) == normalize_topic(self.topic))
        self.memory_focus = stored.weak_areas[:3] if (stored and same_topic) else []
        self.style, self.stage = "normal", "idle"
        self.explanation = self.quiz = self.last_report = None
        self.relearn_round = 0
        self.memory.remember_topic(self.student, self.topic, self.level)
        self._emit(
            "coordinator",
            "info",
            f"Plan: teach '{self.topic}' ({self.level}) -> quiz -> feedback"
            + (
                f"; memory says to revisit: {', '.join(self.memory_focus)}"
                if self.memory_focus
                else ""
            ),
        )

    # ---------------- Explainer ----------------
    def _teach(self, weak: Optional[list[str]] = None) -> Explanation:
        """Explainer step. Whole-topic lessons use the cache; targeted ones never do."""
        targeted = bool(weak)
        if not targeted:
            cached = self.memory.get_cached_explanation(self.topic, self.level, self.style)
            if cached is not None:
                self.stats["cache_hits"] += 1
                self._last = None
                self.explanation, self.stage = cached, "lesson"
                self._emit(
                    "explainer", "cached", f"Reused a saved lesson on '{self.topic}' (0 tokens)"
                )
                return cached

        focus = ", ".join(weak) if weak else "the whole topic"
        prompt = explainer_prompt(
            self.topic, self.level, self.style, focus, max_words=120 if targeted else 250
        )
        self._emit(
            "explainer",
            "working",
            (
                f"Re-teaching weak concepts: {focus}"
                if targeted
                else f"Teaching '{self.topic}' ({self.level}, {self.style} style)"
            ),
        )
        run = self._run("explainer", prompt, Explanation)
        explanation: Explanation = run.result.value
        if not explanation.topic:
            explanation.topic = self.topic
        if not targeted:
            self.memory.put_cached_explanation(self.topic, self.level, explanation, self.style)
        self.explanation, self.stage = explanation, "lesson"
        self._emit(
            "explainer",
            "done",
            f"{explanation.word_count} words, {len(explanation.key_points)} key points",
            run=run,
        )
        return explanation

    # ---------------- Quiz Master ----------------
    def _make_quiz(self, *, num: int, harder: bool, focus: list[str]) -> Quiz:
        """Quiz Master step: receives ONLY the topic and key points (compact handoff)."""
        assert self.explanation is not None, "a lesson must exist before a quiz"
        brief = self.explanation.to_quiz_brief()
        self._handoff("explainer", "quiz_master", brief)
        prompt = quiz_prompt(brief, self.level, num, harder, focus)
        self._emit(
            "quiz_master", "working", f"Writing {num} questions" + (" (harder)" if harder else "")
        )
        run = self._run("quiz_master", prompt, Quiz)
        quiz: Quiz = run.result.value
        if not quiz.topic:
            quiz.topic = self.topic
        self.quiz, self.stage = quiz, "quiz"
        self._emit("quiz_master", "done", f"{len(quiz.questions)} questions ready", run=run)
        return quiz

    def _start_quiz(self, harder: bool) -> None:
        if self.explanation is None:  # quiz needs key points from a lesson first
            self._handoff(
                "coordinator",
                "explainer",
                {"topic": self.topic, "level": self.level, "reason": "quiz needs a lesson first"},
            )
            self._teach()
        self.relearn_round = 0
        self._make_quiz(
            num=QUIZ_SIZE["harder" if harder else "normal"], harder=harder, focus=self.memory_focus
        )

    # ---------------- Public API ----------------
    @graceful
    def handle_message(self, message: str) -> TurnResult:
        """Student typed something. The Coordinator decides what happens next.

        This is also the human-in-the-loop entry point: it works at ANY stage,
        so the student can interrupt a lesson or quiz at any time.
        """
        message = clean_text(message, 300)
        decision = self._route(message)
        intent = decision.intent

        if intent in ("clarify", "refuse"):
            self._emit("coordinator", "info", f"{intent}: answering directly, no specialist needed")
            return self.snapshot(reply=decision.reply)

        if intent in ("learn", "change_topic"):
            self._begin_topic(decision.topic, decision.level)
            self._handoff(
                "coordinator",
                "explainer",
                {"topic": self.topic, "level": self.level, "style": self.style},
            )
            self._teach()
            return self.snapshot()

        if not self.topic:  # simpler / harder / skip with nothing to work on yet
            return self.snapshot(reply="Let's pick a topic first. What would you like to learn?")

        if intent == "simpler":
            previous = (self.style, self.quiz, self.relearn_round)
            self.style, self.quiz, self.relearn_round = "simple", None, 0
            self._handoff(
                "coordinator", "explainer", {"topic": self.topic, "request": "explain more simply"}
            )
            try:
                self._teach()
            except Exception:
                # Failed: restore the previous state, then let @graceful report it.
                self.style, self.quiz, self.relearn_round = previous
                raise
            return self.snapshot()

        # harder_quiz or skip_to_quiz
        self._start_quiz(harder=(intent == "harder_quiz"))
        return self.snapshot()

    @graceful
    def start_quiz(self, harder: bool = False) -> TurnResult:
        """Student pressed 'Start quiz' (Explainer -> Quiz Master)."""
        if not self.topic:
            return self.snapshot(reply="Let's pick a topic first. What would you like to learn?")
        self._start_quiz(harder)
        return self.snapshot()

    @graceful
    def submit_answers(self, answers: dict[int, int]) -> TurnResult:
        """Student submitted answers: grade, give feedback, maybe loop back to re-teach."""
        if self.stage != "quiz" or self.quiz is None:
            return self.snapshot(reply="There is no active quiz. Tell me a topic to begin.")
        quiz = self.quiz

        clean_answers: dict[int, int] = {}
        for key, value in (answers or {}).items():
            try:
                clean_answers[int(key)] = int(value)
            except (TypeError, ValueError):
                continue  # ignore malformed entries

        # 1) Objective grading in plain Python (0 tokens, cannot hallucinate).
        grade = grade_quiz(quiz, clean_answers)
        self._emit(
            "evaluator",
            "info",
            f"Graded locally: {grade.correct_count}/{grade.total} correct (0 tokens)",
        )
        self._handoff(
            "quiz_master",
            "evaluator",
            {
                "topic": self.topic,
                "questions": quiz.public_view(),
                "answers": clean_answers,
                "score": grade.score,
                "weak_concepts": grade.weak_concepts,
            },
        )

        # 2) The Evaluator's LLM only writes feedback, and only if something was wrong.
        if grade.weak_concepts:
            prompt = evaluator_prompt(self.topic, self.level, quiz, grade)
            self._emit("evaluator", "working", "Writing feedback on the wrong answers")
            run = self._run(
                "evaluator", prompt, FeedbackNote, safe_default=lambda: default_feedback(grade)
            )
            feedback: FeedbackNote = run.result.value
            note = " (LLM unavailable: template feedback)" if run.result.used_safe_default else ""
            self._emit("evaluator", "done", f"{len(feedback.tips)} tips written{note}", run=run)
        else:
            feedback = default_feedback(grade)
            self._last = None
            self._emit("evaluator", "done", "Perfect score: template feedback (0 tokens)")

        report = EvaluationReport.from_parts(grade, feedback, self.settings.score_threshold)
        self.last_report = report

        # 3) Update memory with a compact result (not a transcript).
        profile = self.memory.update_after_quiz(self.student, self.topic, self.level, grade)
        notes: list[str] = []
        if (
            report.passed
            and grade.score >= 0.9
            and self.relearn_round == 0
            and self.level != "advanced"
        ):
            profile.level = NEXT_LEVEL[self.level]  # level up for the next topic
            self.memory.save_profile(profile)
            notes.append(f"Excellent! Your next topic will start at {profile.level} level.")
        self.profile = profile

        # 4) BONUS: feedback loop for weak scores (capped at MAX_RELEARN_ROUNDS).
        if report.needs_reteach and self.relearn_round < self.settings.max_relearn_rounds:
            return self._reteach(report, notes)

        self.stage = "done"
        if report.needs_reteach:
            notes.append(
                f"We reached the re-teach limit ({self.settings.max_relearn_rounds} "
                "rounds). Review the tips above and try again later."
            )
        return self.snapshot(report=report, notes=notes)

    def _reteach(self, report: EvaluationReport, notes: list[str]) -> TurnResult:
        """Feedback loop: Evaluator -> Explainer (targeted) -> Quiz Master (mini quiz)."""
        self.relearn_round += 1
        weak = report.grade.weak_concepts
        self._emit(
            "evaluator",
            "info",
            f"Score {round(report.grade.score * 100)}% is below "
            f"{round(report.threshold * 100)}%: sending weak concepts back "
            f"(round {self.relearn_round}/{self.settings.max_relearn_rounds})",
        )
        self._handoff(
            "evaluator",
            "explainer",
            {"weak_concepts": weak, "score": report.grade.score, "round": self.relearn_round},
        )
        try:
            self._teach(weak=weak)
            self._make_quiz(num=QUIZ_SIZE["mini"], harder=False, focus=weak)
        except LLMUnavailableError:
            # Do not lose the report: show it and let the student retry the quiz.
            logger.warning("Re-teach step unavailable; returning the report only")
            notes.append("I couldn't prepare the re-teach just now. Press 'Start quiz' to retry.")
            return self.snapshot(report=report, notes=notes)
        return self.snapshot(report=report, reteaching=True, notes=notes)


# ----------------------------------------------------------------------
# Pretty printing (shared with the CLI in Phase 7)
# ----------------------------------------------------------------------
ICONS = {
    "coordinator": "🧭",
    "explainer": "📖",
    "quiz_master": "📝",
    "evaluator": "✅",
    "system": "⚙️",
}


def format_event(event: ActivityEvent) -> str:
    """One readable line for an activity event."""
    icon = ICONS.get(event.agent, "•")
    if event.status == "handoff":
        payload = json.dumps(event.payload, ensure_ascii=False)
        payload = payload if len(payload) <= 110 else payload[:107] + "..."
        return f"   ↳ HANDOFF {event.agent} → {event.to_agent}   payload={payload}"
    cost = f"  [{event.model}, {event.tokens} tok, {event.duration_s}s]" if event.model else ""
    return f"{icon} {event.agent:<12} {event.status:<8} {event.detail}{cost}"


# ----------------------------------------------------------------------
# Demos
# ----------------------------------------------------------------------
def _run_demo() -> int:
    """Scripted session: no API calls, no tokens. Shows every feature working."""
    from leo.fakes import ScriptedRunner, demo_script

    with tempfile.TemporaryDirectory() as tmp:
        memory = MemoryStore(Path(tmp) / "demo.db")  # temporary database
        runner = ScriptedRunner(demo_script())
        session = LeoSession(
            "Demo Student",
            memory=memory,
            runner=runner,
            on_event=lambda event: print(format_event(event)),
        )

        def say(text: str) -> None:
            print(f"\n\033[1mSTUDENT:\033[0m {text}")

        print(session.greeting())

        say("(sends an empty message)")
        print("LEO:", session.handle_message("").reply)

        say("how do I hack my neighbour's wifi")
        print("LEO:", session.handle_message("how do I hack my neighbour's wifi").reply)

        say("teach me fractions")
        result = session.handle_message("teach me fractions")
        print(f"LEO: lesson ready ({result.explanation.word_count} words), stage={result.stage}")

        say("explain simpler            <- human-in-the-loop intervention")
        result = session.handle_message("explain simpler")
        print(f"LEO: simpler lesson ready, stage={result.stage}")

        say("(clicks Start quiz)")
        result = session.start_quiz()
        print(f"LEO: quiz ready with {len(result.quiz.questions)} questions")

        say("(answers mostly wrong)")
        result = session.submit_answers({1: 0, 2: 1, 3: 2, 4: 0})
        print(
            f"LEO: score {round(result.report.grade.score * 100)}%, "
            f"reteaching={result.reteaching}, round={result.round}, "
            f"mini-quiz questions={len(result.quiz.questions) if result.quiz else 0}"
        )

        say("(answers the mini quiz correctly)")
        result = session.submit_answers({1: 0, 2: 1, 3: 0})
        print(f"LEO: score {round(result.report.grade.score * 100)}%, stage={result.stage}")
        print("     " + session.profile_summary())

        say("teach me fractions        <- same topic again")
        session.handle_message("teach me fractions")

        print("\nUSAGE:", session.usage_summary())
        print("Cache:", memory.cache_stats())
        session.trace.save()
        print("Trace file:", session.trace.path)
    return 0


def _run_live(topic: str) -> int:
    """Real LLM session (about 3-4 calls). Answers use the answer key to stay cheap."""
    session = LeoSession("LiveTest", on_event=lambda event: print(format_event(event)))
    print(session.greeting(), "\n")
    result = session.handle_message(f"teach me {topic}")
    if result.stage != "lesson":
        print("Stopped:", result.reply or "no lesson was produced")
        return 1
    print(f"\nLESSON ({result.explanation.word_count} words):\n{result.explanation.explanation}\n")
    result = session.start_quiz()
    if result.quiz is None:
        print("Stopped:", result.reply or "no quiz was produced")
        return 1
    for question in result.quiz.questions:
        print(f"Q{question.id}. {question.question}")
        for letter, option in zip("ABCD", question.options, strict=True):
            print(f"    {letter}) {option}")
    answers = {q.id: q.correct_index for q in session.quiz.questions}  # cheat: all correct
    result = session.submit_answers(answers)
    print(
        f"\nScore: {round(result.report.grade.score * 100)}%  |  {result.report.feedback.overall}"
    )
    print("\nUSAGE:", session.usage_summary())
    session.memory.forget_student("LiveTest")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leo orchestrator demos")
    parser.add_argument("--demo", action="store_true", help="offline scripted session (default)")
    parser.add_argument(
        "--live",
        nargs="?",
        const="fractions",
        metavar="TOPIC",
        help="real LLM session on TOPIC (default: fractions)",
    )
    args = parser.parse_args()
    sys.exit(_run_live(args.live) if args.live else _run_demo())
