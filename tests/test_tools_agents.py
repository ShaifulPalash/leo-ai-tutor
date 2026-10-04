"""Tool and agent tests (these import CrewAI, so the first run is slow)."""

import pytest

pytestmark = pytest.mark.slow


@pytest.mark.parametrize(
    "expression, expected",
    [("2 + 3 * 4", "14"), ("sqrt(16) + 2**3", "12"), ("(10 - 4) / 4", "1.5"), ("-5 + 2", "-3")],
)
def test_calculator_computes(expression, expected):
    from leo.tools import safe_calculate

    assert safe_calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "1/0",
        "sqrt(-1)",
        "__import__('os').system('echo hi')",
        "open('x')",
        "9**9**9",
        "",
        "a" * 300,
    ],
)
def test_calculator_refuses_unsafe_or_invalid_input(expression):
    from leo.tools import safe_calculate

    with pytest.raises(ValueError):
        safe_calculate(expression)


def test_tool_wrapper_returns_errors_as_text_not_exceptions():
    from leo.tools import SafeCalculatorTool

    tool = SafeCalculatorTool()
    assert tool._run("2+2") == "4"
    assert tool._run("__import__('os')").startswith("Calculator error")


def test_tools_follow_the_config_switches(settings):
    from leo.tools import get_tools

    both = settings.model_copy(update={"enable_calculator": True, "enable_web_search": True})
    only_search = settings.model_copy(
        update={"enable_calculator": False, "enable_web_search": True}
    )
    none = settings.model_copy(update={"enable_calculator": False, "enable_web_search": False})
    assert [t.name for t in get_tools(both)] == ["calculator", "web_search"]
    assert [t.name for t in get_tools(only_search)] == ["web_search"]
    assert get_tools(none) == []


def test_web_search_respects_its_call_limit():
    from leo.tools import WebSearchTool

    assert "limit reached" in WebSearchTool(max_calls=0)._run("anything").lower()


@pytest.mark.parametrize("role", ["coordinator", "explainer", "quiz_master", "evaluator"])
def test_each_agent_is_built_from_its_own_prompt(role, settings):
    from leo.agents import AGENT_SPECS, build_agent
    from leo.llm_factory import LLMFactory
    from leo.prompts.loader import load_prompt

    factory = LLMFactory(settings)
    llm = factory.build_llm(factory.model_chain(AGENT_SPECS[role].tier)[0], 200)
    agent = build_agent(role, llm, settings)  # builds objects only: no network call
    assert agent.role == load_prompt(role).role
    assert agent.allow_delegation is False  # the orchestrator decides, not the LLM
    names = [tool.name for tool in agent.tools]
    assert names == (["calculator"] if role == "explainer" else [])
