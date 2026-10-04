"""Tasks: prompt building, JSON parsing, and guarded execution of one CrewAI step.

CrewAI is imported lazily inside run_agent(), so importing this module is fast and
tests that use a fake runner never load CrewAI.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterator, Optional, TypeVar

from pydantic import BaseModel

from leo.config import Settings, get_settings
from leo.llm_factory import CallContext, LLMFactory, LLMResult, get_factory
from leo.logging_setup import get_agent_logger, redact
from leo.prompts.loader import load_prompt
from leo.schemas import GradeResult, Quiz, RoutingDecision, clean_text

M = TypeVar("M", bound=BaseModel)

# What each task should return (shown to the agent as 'expected output').
EXPECTED_OUTPUT: dict[str, str] = {
    "coordinator": "One JSON object with keys intent, topic, level, reply.",
    "explainer": "One JSON object with keys topic, explanation, key_points, example.",
    "quiz_master": "One JSON object with keys topic, level, questions.",
    "evaluator": "One JSON object with keys overall, tips.",
}


@dataclass
class AgentRun(Generic[M]):
    """Everything one agent step produced."""

    result: LLMResult[M]  # validated output + model, tokens, latency, fallback flags
    prompt: str  # the prompt that was sent (kept for the debug trace)
    raw: str  # raw model text (kept for the debug trace)


# A "runner" is any function with run_agent's signature. Tests pass a fake one.
AgentRunner = Callable[..., AgentRun]


# ----------------------------------------------------------------------
# JSON extraction and validation
# ----------------------------------------------------------------------
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def extract_json(text: str) -> Any:
    """Find and decode the first JSON object in model output.

    Handles ```json fences``` and extra chatter before or after the object.
    Raises ValueError (json.JSONDecodeError is a ValueError) if nothing usable exists.
    """
    text = (text or "").strip()
    fence = _FENCE.search(text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in the model output")
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(text)):  # walk to the matching closing brace
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])
    raise ValueError("JSON object in the model output is incomplete (was it cut off?)")


def parse_model(raw: str, model_cls: type[M]) -> M:
    """Raw LLM text -> validated Pydantic object (raises ValueError if invalid)."""
    data = extract_json(raw)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object, got something else")
    return model_cls.model_validate(data)  # ValidationError is a ValueError


# ----------------------------------------------------------------------
# Prompt builders (fill the YAML templates from Phase 4)
# ----------------------------------------------------------------------
def coordinator_prompt(stage: str, current_topic: str, level: str, message: str) -> str:
    safe_message = clean_text(message, 300).replace('"""', "'''")  # keep the data fenced
    return load_prompt("coordinator").render(
        stage=stage,
        current_topic=current_topic.replace('"', "'"),
        level=level,
        message=safe_message,
    )


def explainer_prompt(topic: str, level: str, style: str, focus: str, max_words: int) -> str:
    style_hint = (
        "Use very simple words, short sentences and one everyday analogy."
        if style == "simple"
        else ""
    )
    return load_prompt("explainer").render(
        topic=topic, level=level, style_hint=style_hint, focus=focus, max_words=max_words
    )


def quiz_prompt(brief: dict, level: str, num_questions: int, harder: bool, focus: list[str]) -> str:
    key_points = "\n".join(f"- {point}" for point in brief.get("key_points", []))
    difficulty = (
        "Make the questions harder: test understanding and application, " "not just definitions."
        if harder
        else ""
    )
    return load_prompt("quiz_master").render(
        num_questions=num_questions,
        topic=brief.get("topic", ""),
        level=level,
        difficulty_hint=difficulty,
        key_points=key_points,
        focus=", ".join(focus) or "none",
    )


def describe_wrong_answers(quiz: Quiz, grade: GradeResult) -> str:
    """Compact text listing only the wrong answers (all the Evaluator needs)."""
    by_id = {q.id: q for q in quiz.questions}
    lines = []
    for result in grade.results:
        if result.correct:
            continue
        question = by_id[result.question_id]
        chosen = question.options[result.chosen_index] if result.chosen_index >= 0 else "no answer"
        lines.append(
            f'- {result.concept}: Q "{question.question[:70]}" '
            f'chose "{chosen}"; correct "{question.options[result.correct_index]}"'
        )
    return "\n".join(lines) or "- none"


def evaluator_prompt(topic: str, level: str, quiz: Quiz, grade: GradeResult) -> str:
    return load_prompt("evaluator").render(
        score_pct=round(grade.score * 100),
        correct=grade.correct_count,
        total=grade.total,
        topic=topic,
        level=level,
        wrong_summary=describe_wrong_answers(quiz, grade),
    )


def heuristic_decision(stage: str, message: str) -> RoutingDecision:
    """Coordinator safe default, used only if every model fails to route.

    Deliberately simple: a short message at the start is treated as a topic;
    anything else gets a clarifying question. (It cannot judge safety, which
    is why it only applies when no LLM is reachable at all.)
    """
    text = clean_text(message, 120)
    text = re.sub(
        r"^(please\s+)?(teach me|explain|i want to learn|learn( about)?)\s+", "", text, flags=re.I
    )
    words = text.split()
    if stage == "idle" and 1 <= len(words) <= 6 and re.fullmatch(r"[\w\s,'\-+&/().]+", text):
        return RoutingDecision(intent="learn", topic=text)
    return RoutingDecision(
        intent="clarify",
        reply="I'm having trouble right now. Please name the topic in a few words, "
        "for example 'photosynthesis'.",
    )


# ----------------------------------------------------------------------
# Running one agent step
# ----------------------------------------------------------------------
@contextlib.contextmanager
def _quiet(settings: Settings) -> Iterator[None]:
    """Hide CrewAI's console panels unless DEBUG_MODE=true (our logs are unaffected)."""
    if settings.debug_mode:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def _record_usage(ctx: CallContext, output: Any, prompt: str, raw: str) -> None:
    """Report token usage: exact numbers if CrewAI gives them, else an estimate."""
    usage = getattr(output, "token_usage", None)
    total = int(getattr(usage, "total_tokens", 0) or 0) if usage is not None else 0
    if total:
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        if prompt_tokens + completion_tokens == 0:
            completion_tokens = total
        ctx.set_usage(prompt_tokens, completion_tokens)
    else:
        ctx.estimate_usage(prompt, raw)


def run_agent(
    role: str,
    prompt: str,
    model_cls: type[M],
    *,
    factory: LLMFactory | None = None,
    safe_default: Optional[Callable[[], M]] = None,
    settings: Settings | None = None,
) -> AgentRun[M]:
    """Run ONE agent step as a one-task sequential Crew, with retries and fallback.

    Each attempt rebuilds the Agent with the model chosen by run_with_fallback,
    so a fallback to Gemini really switches the agent's brain.
    """
    from crewai import Crew, Process, Task  # lazy: loading CrewAI is slow

    from leo.agents import build_agent, get_spec

    settings = settings or get_settings()
    factory = factory or get_factory()
    spec = get_spec(role)
    agent_log = get_agent_logger(role)
    captured = {"raw": ""}

    def operation(ctx: CallContext) -> M:
        full_prompt = ctx.with_repair(prompt)  # adds a repair hint after invalid output
        agent = build_agent(role, ctx.llm, settings)
        task = Task(description=full_prompt, expected_output=EXPECTED_OUTPUT[role], agent=agent)
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        with _quiet(settings):
            output = crew.kickoff()
        raw = str(getattr(output, "raw", output) or "").strip()
        captured["raw"] = raw
        _record_usage(ctx, output, full_prompt, raw)
        agent_log.debug("prompt (%d chars): %s", len(full_prompt), redact(full_prompt[:800]))
        agent_log.debug("raw output (%d chars): %s", len(raw), redact(raw[:800]))
        if not raw:
            raise ValueError("empty output from model")  # triggers repair/retry/fallback
        return parse_model(raw, model_cls)

    result = factory.run_with_fallback(
        tier=spec.tier,
        agent=role,
        operation=operation,
        max_tokens=settings.max_tokens_by_role[role],
        safe_default=safe_default,
    )
    return AgentRun(result=result, prompt=prompt, raw=captured["raw"])
