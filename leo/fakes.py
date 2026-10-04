"""Test doubles: a scripted stand-in for the real agent runner (no API needed).

Used by the offline demo (python -m leo.crew --demo) and by the pytest suite.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from pydantic import BaseModel

from leo.llm_factory import LLMResult, LLMUnavailableError
from leo.tasks import AgentRun

_TIER = {"coordinator": "fast", "evaluator": "fast", "explainer": "smart", "quiz_master": "smart"}


class ScriptedRunner:
    """Returns canned answers per role. Records every call for assertions."""

    def __init__(self, script: dict[str, list[Any]]) -> None:
        self._script = {role: list(items) for role, items in script.items()}
        self.calls: list[tuple[str, str]] = []  # (role, prompt)

    def calls_for(self, role: str) -> list[str]:
        """Prompts that were sent to one role."""
        return [prompt for r, prompt in self.calls if r == role]

    def __call__(
        self,
        role: str,
        prompt: str,
        model_cls: type[BaseModel],
        *,
        factory: Any = None,
        safe_default: Optional[Callable[[], Any]] = None,
        settings: Any = None,
    ) -> AgentRun:
        self.calls.append((role, prompt))
        queue = self._script.get(role)
        if not queue:
            raise LLMUnavailableError(f"no scripted reply left for '{role}'")
        item = queue.pop(0) if len(queue) > 1 else queue[0]  # last item repeats
        if callable(item) and not isinstance(item, BaseException):
            item = item(prompt)

        used_default = False
        if isinstance(item, BaseException):  # simulate "every model failed"
            if safe_default is None:
                raise LLMUnavailableError(f"scripted failure for '{role}'") from item
            value, used_default = safe_default(), True
        elif isinstance(item, model_cls):
            value = item
        else:
            value = model_cls.model_validate(item)  # dict -> validated model

        raw = value.model_dump_json()
        result = LLMResult(
            value=value,
            model="none" if used_default else "fake/scripted",
            tier=_TIER.get(role, "fast"),
            agent=role,
            attempts=1,
            latency_s=0.0,
            prompt_tokens=0 if used_default else max(1, len(prompt) // 4),
            completion_tokens=0 if used_default else max(1, len(raw) // 4),
            used_fallback_model=used_default,
            used_safe_default=used_default,
        )
        return AgentRun(result=result, prompt=prompt, raw=raw)


def demo_script() -> dict[str, list[Any]]:
    """A full scripted session: refuse, learn, simpler, quiz, fail, re-teach, pass."""
    quiz_full = {
        "topic": "Fractions",
        "level": "beginner",
        "questions": [
            {
                "id": 1,
                "question": "Which number is the numerator in 3/4?",
                "options": ["4", "3", "7", "1"],
                "correct_index": 1,
                "concept": "numerator",
            },
            {
                "id": 2,
                "question": "What is 2/4 in simplest form?",
                "options": ["1/2", "2/2", "4/2", "1/4"],
                "correct_index": 0,
                "concept": "simplifying",
            },
            {
                "id": 3,
                "question": "What does the denominator tell you?",
                "options": [
                    "Parts you have",
                    "The sign",
                    "Total equal parts in the whole",
                    "How big the fraction is",
                ],
                "correct_index": 2,
                "concept": "denominator",
            },
            {
                "id": 4,
                "question": "Which fraction equals 1/2?",
                "options": ["2/3", "3/4", "1/3", "2/4"],
                "correct_index": 3,
                "concept": "equivalent fractions",
            },
        ],
    }
    quiz_mini = {
        "topic": "Fractions",
        "level": "beginner",
        "questions": [
            {
                "id": 1,
                "question": "Simplify 6/8.",
                "options": ["3/4", "2/3", "6/4", "1/2"],
                "correct_index": 0,
                "concept": "simplifying",
            },
            {
                "id": 2,
                "question": "In 5/9, which number is the numerator?",
                "options": ["9", "5", "4", "14"],
                "correct_index": 1,
                "concept": "numerator",
            },
            {
                "id": 3,
                "question": "Which fraction equals 1/2?",
                "options": ["3/6", "2/5", "1/3", "4/10"],
                "correct_index": 0,
                "concept": "equivalent fractions",
            },
        ],
    }
    return {
        "coordinator": [
            {
                "intent": "refuse",
                "topic": "",
                "level": "",
                "reply": "I can only help with study topics, so I can't help with that. "
                "Want to learn something instead?",
            },
            {"intent": "learn", "topic": "Fractions", "level": "", "reply": ""},
        ],
        "explainer": [
            {
                "topic": "Fractions",
                "explanation": "A fraction shows part of a whole. The top number (numerator) "
                "counts the parts you have. The bottom number (denominator) says "
                "how many equal parts make the whole.",
                "key_points": [
                    "Numerator counts the parts you have",
                    "Denominator is the total number of equal parts",
                    "Different-looking fractions can be equal",
                ],
                "example": "In 3/4 of a pizza you have 3 of 4 equal slices.",
            },
            {
                "topic": "Fractions",
                "explanation": "Think of a pizza cut into equal slices. The bottom number says "
                "how many slices there are. The top number says how many you take.",
                "key_points": ["Bottom number = total slices", "Top number = slices you take"],
                "example": "3/4 means you take 3 of the 4 slices.",
            },
            {
                "topic": "Fractions",
                "explanation": "To simplify, divide the top and bottom by the same number. "
                "So 2/4 becomes 1/2, and 6/8 becomes 3/4.",
                "key_points": [
                    "Divide top and bottom by the same number",
                    "A simplified fraction has the same value",
                ],
                "example": "6/8 divided by 2 on both sides is 3/4.",
            },
        ],
        "quiz_master": [quiz_full, quiz_mini],
        "evaluator": [
            {
                "overall": "Good start! Let's strengthen a few ideas together.",
                "tips": [
                    {"concept": "numerator", "tip": "The numerator is the TOP number."},
                    {"concept": "simplifying", "tip": "Divide both numbers by the same value."},
                    {
                        "concept": "equivalent fractions",
                        "tip": "Equal fractions name the same amount.",
                    },
                ],
            },
        ],
    }
