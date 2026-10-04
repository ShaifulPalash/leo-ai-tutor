"""Optional tools for Leo's Explainer: a SAFE calculator and a web search.

Tools are switched on/off in .env (ENABLE_CALCULATOR / ENABLE_WEB_SEARCH).
Run:  python -m leo.tools           (offline tests)
      python -m leo.tools --web     (also tries one real web search)
"""

from __future__ import annotations

import ast
import math
import operator
import sys
from typing import Any, Callable, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from leo.config import Settings, get_settings
from leo.logging_setup import get_logger

logger = get_logger("tools")

# ----------------------------------------------------------------------
# Safe math evaluator (no eval!)
# ----------------------------------------------------------------------
MAX_EXPR_LEN = 200  # longest expression we accept
MAX_ABS_EXPONENT = 100  # 2**100 is fine, 9**9**9 is refused
MAX_ABS_RESULT = 1e50  # stop numbers from growing out of control
MAX_DEPTH = 30  # stop absurdly nested expressions

_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "floor": math.floor,
    "ceil": math.ceil,
}
_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e}


def _eval_node(node: ast.AST, depth: int = 0) -> Any:
    """Evaluate one node of the parsed expression; reject anything not whitelisted."""
    if depth > MAX_DEPTH:
        raise ValueError("expression is too deeply nested")

    if isinstance(node, ast.Constant):  # a literal number
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("only numbers are allowed")

    if isinstance(node, ast.Name):  # pi or e
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ValueError(f"unknown name '{node.id}'")

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand, depth + 1))

    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left, depth + 1)
        right = _eval_node(node.right, depth + 1)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_ABS_EXPONENT:
            raise ValueError(f"exponent is too large (limit {MAX_ABS_EXPONENT})")
        result = _BIN_OPS[type(node.op)](left, right)
        if abs(result) > MAX_ABS_RESULT:
            raise ValueError("result is too large")
        return result

    if isinstance(node, ast.Call):  # sqrt(16), round(2.5) ...
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS
            and not node.keywords
            and len(node.args) <= 5
        ):
            args = [_eval_node(arg, depth + 1) for arg in node.args]
            return _FUNCTIONS[node.func.id](*args)
        raise ValueError("that function is not allowed")

    raise ValueError("unsupported expression")


def safe_calculate(expression: str) -> str:
    """Evaluate a math expression safely. Raises ValueError with a clear reason."""
    expression = (expression or "").strip()
    if not expression:
        raise ValueError("empty expression")
    if len(expression) > MAX_EXPR_LEN:
        raise ValueError(f"expression longer than {MAX_EXPR_LEN} characters")
    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval_node(tree.body)
    except ValueError:
        raise
    except ZeroDivisionError:
        raise ValueError("division by zero") from None
    except (SyntaxError, TypeError, OverflowError, MemoryError, RecursionError) as exc:
        raise ValueError(f"cannot evaluate ({type(exc).__name__})") from None
    if isinstance(value, float):
        return f"{value:.10g}"  # 12.0 -> '12',  1/3 -> '0.3333333333'
    return str(value)


# ----------------------------------------------------------------------
# CrewAI tool wrappers
# ----------------------------------------------------------------------
class CalculatorInput(BaseModel):
    """Arguments the LLM must supply when it calls the calculator."""

    expression: str = Field(..., description="Math expression, e.g. '3/4 + 1/2' or 'sqrt(81)'")


class SafeCalculatorTool(BaseTool):
    """Lets the Explainer verify arithmetic instead of guessing."""

    name: str = "calculator"
    description: str = (
        "Evaluate a math expression (+ - * / // % **, sqrt, abs, round, "
        "min, max, sin, cos, tan, log, pi, e). Input: expression."
    )
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        try:
            result = safe_calculate(expression)
            logger.info("calculator: %s = %s", expression[:60], result)
            return result
        except ValueError as exc:  # return the problem as text; never crash the agent
            logger.warning("calculator refused %r: %s", expression[:60], exc)
            return f"Calculator error: {exc}"


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Short search query, 3-8 words")


class WebSearchTool(BaseTool):
    """Free DuckDuckGo search, capped to save tokens."""

    name: str = "web_search"
    description: str = (
        "Search the web for a fact you are unsure of. Returns up to 3 short "
        "snippets. Use sparingly. Input: query."
    )
    args_schema: Type[BaseModel] = WebSearchInput
    max_calls: int = 2  # searches allowed per tool instance
    _calls: int = PrivateAttr(default=0)

    def _run(self, query: str) -> str:
        if self._calls >= self.max_calls:
            return "Search limit reached. Answer from your own knowledge."
        self._calls += 1
        try:
            from ddgs import DDGS  # imported lazily: only needed when enabled

            hits = list(DDGS().text(query[:120], max_results=3))
        except Exception as exc:  # noqa: BLE001 - network or library problems
            logger.warning("web_search failed: %s", type(exc).__name__)
            return "Web search is unavailable right now. Answer from your own knowledge."
        if not hits:
            return "No results found."
        lines = [f"- {str(h.get('title', ''))[:80]}: {str(h.get('body', ''))[:200]}" for h in hits]
        logger.info("web_search %r -> %d results", query[:60], len(lines))
        return "\n".join(lines)


def get_tools(settings: Settings | None = None) -> list[BaseTool]:
    """Build the list of enabled tools according to .env settings."""
    settings = settings or get_settings()
    tools: list[BaseTool] = []
    if settings.enable_calculator:
        tools.append(SafeCalculatorTool())
    if settings.enable_web_search:
        tools.append(WebSearchTool())
    return tools


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------
def _self_test(try_web: bool) -> int:
    # (expression, substring that must appear in the answer)
    cases = [
        ("2 + 3 * 4", "14"),
        ("sqrt(16) + 2**3", "12"),
        ("(10 - 4) / 4", "1.5"),
        ("-5 + 2", "-3"),
        ("1/0", "error"),
        ("sqrt(-1)", "error"),
        ("__import__('os').system('echo hi')", "error"),
        ("open('secret.txt')", "error"),
        ("9**9**9", "error"),
        ("", "error"),
    ]
    failures = 0
    print("Safe calculator:")
    for expression, expected in cases:
        try:
            answer = safe_calculate(expression)
        except ValueError as exc:
            answer = f"error: {exc}"
        passed = expected in answer
        failures += not passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {expression[:38]!r:42} -> {answer}")

    print("\nCrewAI tool wrapper:")
    try:
        out = SafeCalculatorTool().run(expression="2+2")
        passed = "4" in str(out)
        failures += not passed
        print(f"  [{'PASS' if passed else 'FAIL'}] SafeCalculatorTool.run('2+2') -> {out}")
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"  [FAIL] wrapper raised {type(exc).__name__}: {exc}")

    if try_web:
        print("\nWeb search (live):")
        print(" ", WebSearchTool().run(query="photosynthesis definition")[:300])

    print("\nEnabled tools from .env:", [t.name for t in get_tools()] or "none")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_self_test(try_web="--web" in sys.argv))
