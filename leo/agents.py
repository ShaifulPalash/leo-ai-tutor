"""Leo's four agents, built with CrewAI from the YAML prompt templates.

Run:  python -m leo.agents          # show the roster, NO API calls
      python -m leo.agents --live   # one tiny real call through a 1-task Crew
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from crewai import Agent

from leo.config import Settings, Tier, get_settings
from leo.logging_setup import get_logger
from leo.prompts.loader import load_prompt
from leo.tools import get_tools

logger = get_logger("agents")


@dataclass(frozen=True)
class AgentSpec:
    """Static facts about one agent (who it is, which model tier, how strict)."""

    key: str  # matches prompts/<key>.yaml and settings.max_tokens_by_role
    display_name: str  # shown in the UI
    emoji: str
    tier: Tier  # "fast" (cheap) or "smart" (better quality)
    max_iter: int  # max reasoning/tool steps per task
    uses_tools: bool
    summary: str  # one line used in the README and UI


AGENT_SPECS: dict[str, AgentSpec] = {
    "coordinator": AgentSpec(
        "coordinator",
        "Coordinator",
        "🧭",
        "fast",
        3,
        False,
        "Understands the request, routes work, handles unclear or unsafe input.",
    ),
    "explainer": AgentSpec(
        "explainer",
        "Explainer",
        "📖",
        "smart",
        4,
        True,
        "Teaches the topic in 250 words or fewer, with one example.",
    ),
    "quiz_master": AgentSpec(
        "quiz_master",
        "Quiz Master",
        "📝",
        "smart",
        3,
        False,
        "Writes 3-5 multiple-choice questions as validated JSON.",
    ),
    "evaluator": AgentSpec(
        "evaluator",
        "Evaluator",
        "✅",
        "fast",
        3,
        False,
        "Turns graded results into specific, encouraging feedback.",
    ),
}
ROLE_ORDER: tuple[str, ...] = ("coordinator", "explainer", "quiz_master", "evaluator")


def get_spec(key: str) -> AgentSpec:
    """Look up an agent spec, with a helpful error for typos."""
    try:
        return AGENT_SPECS[key]
    except KeyError:
        raise KeyError(f"Unknown agent '{key}'. Choose from {list(AGENT_SPECS)}") from None


def build_agent(key: str, llm: Any, settings: Settings | None = None) -> Agent:
    """Create one CrewAI Agent using its YAML prompt and the given LLM."""
    settings = settings or get_settings()
    spec = get_spec(key)
    template = load_prompt(key)  # role / goal / backstory from YAML
    tools = get_tools(settings) if spec.uses_tools else []

    base: dict[str, Any] = dict(
        **template.agent_fields,
        llm=llm,
        tools=tools,
        allow_delegation=False,  # the orchestrator decides who acts, not the LLM
        max_iter=spec.max_iter,
        max_rpm=settings.max_rpm,
        verbose=settings.debug_mode,
    )
    try:
        # Hard time cap per task. Skipped automatically if this CrewAI version rejects it.
        agent = Agent(**base, max_execution_time=settings.llm_timeout_seconds * 3)
    except (TypeError, ValueError) as exc:
        logger.debug(
            "max_execution_time not accepted (%s); building without it", type(exc).__name__
        )
        agent = Agent(**base)

    logger.debug(
        "Built agent %s (tier=%s, tools=%s)", key, spec.tier, [t.name for t in tools] or "none"
    )
    return agent


# ----------------------------------------------------------------------
# Self-test helpers
# ----------------------------------------------------------------------
def _print_roster() -> int:
    """Build all four agents with their first-choice model. Makes NO API calls."""
    from leo.llm_factory import LLMFactory

    settings = get_settings()
    factory = LLMFactory(settings)
    print("Leo agent roster (no API calls)\n")
    problems = 0
    for key in ROLE_ORDER:
        spec = get_spec(key)
        chain = factory.model_chain(spec.tier)
        if not chain:
            print(f"  [FAIL] {spec.display_name}: no usable model for tier '{spec.tier}'")
            problems += 1
            continue
        try:
            llm = factory.build_llm(chain[0], settings.max_tokens_by_role[key])
            agent = build_agent(key, llm, settings)
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {spec.display_name}: {type(exc).__name__}: {exc}")
            problems += 1
            continue
        tool_names = ", ".join(t.name for t in (agent.tools or [])) or "-"
        print(
            f"  {spec.emoji} {spec.display_name:<12} tier={spec.tier:<6} "
            f"model={chain[0]:<30} tools={tool_names:<12} "
            f"max_iter={getattr(agent, 'max_iter', '?')}"
        )
        print(f"     role: {agent.role}")
    return 1 if problems else 0


def _live_test() -> int:
    """One tiny real call using the same pattern Phase 6 uses."""
    from crewai import Crew, Process, Task

    from leo.llm_factory import CallContext, get_factory, usage_tracker

    settings = get_settings()
    template = load_prompt("coordinator")
    prompt = template.render(
        stage="idle", current_topic="", level="beginner", message="teach me fractions"
    )

    def operation(ctx: CallContext) -> str:
        agent = build_agent("coordinator", ctx.llm, settings)
        task = Task(
            description=prompt,
            expected_output="One JSON object with keys intent, topic, level, reply.",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        output = crew.kickoff()
        raw = str(getattr(output, "raw", output) or "").strip()
        if not raw:
            raise ValueError("empty output from model")  # triggers retry / fallback
        usage = getattr(output, "token_usage", None)
        if usage is not None and getattr(usage, "total_tokens", 0):
            ctx.set_usage(
                int(getattr(usage, "prompt_tokens", 0) or 0),
                int(getattr(usage, "completion_tokens", 0) or 0),
            )
        else:
            ctx.estimate_usage(prompt, raw)
        return raw

    print("Live test: Coordinator -> Task -> sequential Crew (1 real LLM call)")
    result = get_factory().run_with_fallback(
        tier="fast", agent="coordinator", operation=operation, max_tokens=settings.max_tokens_router
    )
    print(f"  model used    : {result.model} (fallback used: {result.used_fallback_model})")
    print(f"  tokens / time : {result.total_tokens} tokens, {result.latency_s:.1f}s")
    print(f"  raw output    : {str(result.value)[:300]}")
    print(f"  usage totals  : {usage_tracker.totals()}")
    return 0


if __name__ == "__main__":
    code = _print_roster()
    if code == 0 and "--live" in sys.argv:
        code = _live_test()
    sys.exit(code)
