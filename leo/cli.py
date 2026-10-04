"""Leo's command-line interface.

Run:  python -m leo.cli                              real LLMs (free tier)
      python -m leo.cli --offline                    scripted demo, NO API calls
      python -m leo.cli --name Palash --topic fractions
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import textwrap
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from leo.config import get_settings
from leo.crew import ActivityEvent, LeoSession, TurnResult, format_event
from leo.fakes import ScriptedRunner, demo_script
from leo.logging_setup import setup_logging
from leo.memory import MemoryStore
from leo.schemas import EvaluationReport

# ----------------------------------------------------------------------
# Terminal colors (automatically off when output is piped or NO_COLOR is set)
# ----------------------------------------------------------------------
USE_COLOR = sys.stdout.isatty() and not os.getenv("NO_COLOR")
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
RED, GREEN, YELLOW, MAGENTA, CYAN = ("\033[91m", "\033[92m", "\033[93m", "\033[95m", "\033[96m")
STATUS_COLOR = {
    "working": YELLOW,
    "done": GREEN,
    "cached": CYAN,
    "handoff": MAGENTA,
    "failed": RED,
    "info": DIM,
}

HELP_TEXT = """
Just talk to Leo. Examples:
  teach me photosynthesis        start a topic
  explain simpler                easier explanation   (you can say this any time)
  skip to the quiz               go straight to the quiz
  give me a harder quiz          5 harder questions
  change topic                   pick something else

Commands:
  /quiz     start the quiz        /harder   start a harder quiz
  /stats    tokens and timing     /memory   what Leo remembers about you
  /forget   delete your memory    /help     show this help      /quit   exit

During a quiz: type A, B, C or D. Press Enter to skip a question, or type a
request (like "explain simpler") to interrupt the quiz.
"""


def paint(text: str, *codes: str) -> str:
    """Wrap text in color codes if colors are enabled."""
    return "".join(codes) + text + RESET if USE_COLOR else text


def wrap(text: str, indent: str = "   ") -> str:
    """Word-wrap long text for the terminal."""
    return textwrap.fill(text, width=88, initial_indent=indent, subsequent_indent=indent)


def print_event(event: ActivityEvent) -> None:
    """Live activity line: shown the moment an agent starts, finishes or hands off."""
    print(paint(format_event(event), STATUS_COLOR.get(event.status, "")), flush=True)


def quiet_console_logs() -> None:
    """Hide INFO logs on the console (the file log keeps them). Debug mode keeps all."""
    if get_settings().debug_mode:
        return
    for handler in logging.getLogger("leo").handlers:
        if not isinstance(handler, RotatingFileHandler):
            handler.setLevel(logging.WARNING)


# ----------------------------------------------------------------------
# Offline (scripted) components, also used by the Streamlit app
# ----------------------------------------------------------------------
def offline_components() -> tuple[MemoryStore, ScriptedRunner]:
    """A temporary memory DB and a scripted runner: no API calls, no tokens."""
    script = demo_script()
    script["coordinator"].pop(0)  # drop the scripted refusal so the first message works
    memory = MemoryStore(Path(tempfile.mkdtemp(prefix="leo-offline-")) / "offline.db")
    return memory, ScriptedRunner(script)


# ----------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------
def show_lesson(session: LeoSession) -> None:
    lesson = session.explanation
    if lesson is None:
        return
    print(
        paint(
            f"\n📖 {lesson.topic or session.topic}  " f"({session.level}, {session.style} style)",
            BOLD,
        )
    )
    print(wrap(lesson.explanation))
    print(paint("   Key points:", BOLD))
    for point in lesson.key_points:
        print(f"      • {point}")
    if lesson.example:
        print(wrap("💡 " + lesson.example))


def show_report(report: EvaluationReport) -> None:
    grade = report.grade
    pct = round(grade.score * 100)
    verdict = "PASSED" if report.passed else "needs another look"
    print(
        paint(
            f"\n📊 Score: {pct}% ({grade.correct_count}/{grade.total}) - {verdict}",
            BOLD,
            GREEN if report.passed else YELLOW,
        )
    )
    print(wrap(report.feedback.overall))
    for tip in report.feedback.tips:
        print(wrap(f"• {tip.concept}: {tip.tip}", indent="   "))
    if grade.strong_concepts:
        print(paint(f"   Strong: {', '.join(grade.strong_concepts)}", GREEN))


def render(result: TurnResult, session: LeoSession) -> None:
    """Show everything new in a TurnResult."""
    if result.reply:
        print(paint(f"\n🦁 Leo: {result.reply}", BOLD, RED if result.error else ""))
    if result.report:
        show_report(result.report)
    for note in result.notes:
        print(paint(f"ℹ️  {note}", CYAN))
    if result.reteaching:
        print(
            paint(
                f"\n🔁 Re-teach round {result.round}: a short lesson on your weak "
                "concepts, then a mini quiz.",
                YELLOW,
            )
        )
    if result.error:
        return  # a failed turn: show only the friendly message
    if (result.stage == "lesson" or result.reteaching) and session.explanation:
        show_lesson(session)
    if result.stage == "lesson":
        print(paint("\n👉 Type /quiz when you're ready, or say 'explain simpler'.", DIM))
    elif result.stage == "done":
        print(paint("\n🎉 Round complete! Type a new topic to keep learning, or /quit.", DIM))


def ask_quiz(quiz) -> tuple[dict[int, int], Optional[str]]:
    """Ask each question. Returns (answers, interruption_text).

    If the student types something other than A-D / Enter, the quiz stops and that
    text is returned so the Coordinator can handle it (human-in-the-loop).
    """
    print(paint(f"\n📝 Quiz: {quiz.topic or 'practice'} ({len(quiz.questions)} questions)", BOLD))
    answers: dict[int, int] = {}
    for question in quiz.questions:
        print(paint(f"\nQ{question.id}. {question.question}", BOLD))
        for letter, option in zip("ABCD", question.options, strict=True):
            print(f"     {letter}) {option}")
        while True:
            try:
                raw = input("   Your answer (A-D, Enter = skip, or type a request): ").strip()
            except (EOFError, KeyboardInterrupt):
                return answers, None
            if raw == "":
                break
            if len(raw) == 1 and raw.upper() in "ABCD":
                answers[question.id] = "ABCD".index(raw.upper())
                break
            return {}, raw  # an interruption, not an answer
    return answers, None


def run_turn(session: LeoSession, first_result: TurnResult) -> None:
    """Render a result; keep running quiz rounds while a quiz is active."""
    result = first_result
    while True:
        render(result, session)
        if result.stage != "quiz" or result.quiz is None:
            return
        answers, interruption = ask_quiz(result.quiz)
        if interruption is not None:
            print(paint("Okay, let me adjust...", DIM))
            result = session.handle_message(interruption)
            continue
        if not answers:
            print(
                paint(
                    "No answers given, so the quiz is paused. " "Type /quiz for a fresh one.", DIM
                )
            )
            return
        result = session.submit_answers(answers)


def print_stats(session: LeoSession) -> None:
    usage = session.usage_summary()
    print(paint("\n📈 Session usage", BOLD))
    for key in (
        "llm_calls",
        "tokens",
        "prompt_tokens",
        "completion_tokens",
        "latency_s",
        "cache_hits",
        "fallbacks",
        "safe_defaults",
    ):
        print(f"   {key:<18}{usage[key]}")
    print(f"   {'trace_file':<18}{usage['trace_file']}")


def print_memory(session: LeoSession) -> None:
    print(paint("\n🧠 What Leo remembers", BOLD))
    print("  ", session.profile_summary())
    for row in session.memory.get_history(session.student, limit=5):
        print(
            f"   - {row['topic']}: {round((row['score'] or 0) * 100)}% "
            f"(weak: {', '.join(row['weak_areas']) or 'none'})"
        )


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m leo.cli", description="Leo AI Tutor (CLI)")
    parser.add_argument("--name", help="student name (skips the name prompt)")
    parser.add_argument("--topic", help="start learning this topic immediately")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="scripted demo mode: no API calls, always teaches Fractions",
    )
    args = parser.parse_args(argv)

    setup_logging()
    quiet_console_logs()
    settings = get_settings()

    if not args.offline:
        for tier in ("fast", "smart"):
            if not settings.model_chain(tier):
                print(
                    paint(
                        f"No usable '{tier}' model. Check your API keys in .env "
                        "(or try --offline).",
                        RED,
                    )
                )
                return 1

    print(paint("\n🦁  LEO - Multi-Agent AI Tutor", BOLD, CYAN))
    print(paint("    Coordinator → Explainer → Quiz Master → Evaluator\n", DIM))
    if args.offline:
        print(
            paint(
                "    OFFLINE DEMO: scripted answers, no API calls. Topic is always " "Fractions.\n",
                YELLOW,
            )
        )

    name = args.name
    if not name:
        try:
            name = input("Your name: ").strip()
        except (EOFError, KeyboardInterrupt):
            name = ""
    name = name or "Student"

    if args.offline:
        memory, runner = offline_components()
        session = LeoSession(name, memory=memory, runner=runner, on_event=print_event)
    else:
        session = LeoSession(name, on_event=print_event)

    print(paint(f"\n🦁 Leo: {session.greeting()}", BOLD))
    print(paint("   (type /help for commands)", DIM))

    if args.topic:
        run_turn(session, session.handle_message(f"teach me {args.topic}"))

    while True:
        try:
            text = input(f"\n{session.student} [{session.stage}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        command = text.lower()
        if command in ("/quit", "/exit", "quit", "exit"):
            break
        if command == "/help":
            print(HELP_TEXT)
        elif command == "/stats":
            print_stats(session)
        elif command == "/memory":
            print_memory(session)
        elif command == "/forget":
            deleted = session.memory.forget_student(session.student)
            print("Memory deleted." if deleted else "Nothing was stored for you yet.")
        elif command in ("/quiz", "/harder"):
            run_turn(session, session.start_quiz(harder=(command == "/harder")))
        else:
            run_turn(session, session.handle_message(text))

    session.trace.save()
    print(paint("\nGoodbye! 👋", BOLD))
    print_stats(session)
    return 0


if __name__ == "__main__":
    sys.exit(main())
