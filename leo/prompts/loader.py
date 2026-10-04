"""Loads prompt templates from YAML files and fills in their {{variables}}.

Run `python -m leo.prompts.loader` to render every template with demo values
and see roughly how many tokens each one uses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROMPT_DIR = Path(__file__).resolve().parent
_REQUIRED_KEYS = ("role", "goal", "backstory", "task_template")
_VAR_PATTERN = re.compile(r"\{\{\s*(\w+)\s*\}\}")  # matches {{name}}


@dataclass(frozen=True)
class PromptTemplate:
    """One agent's prompt: identity fields plus a fill-in-the-blanks task."""

    name: str
    role: str
    goal: str
    backstory: str
    task_template: str

    @property
    def required_variables(self) -> set[str]:
        """Names of all {{variables}} used in the task template."""
        return set(_VAR_PATTERN.findall(self.task_template))

    @property
    def agent_fields(self) -> dict[str, str]:
        """role/goal/backstory, ready to pass into a CrewAI Agent."""
        return {"role": self.role, "goal": self.goal, "backstory": self.backstory}

    def render(self, **variables: Any) -> str:
        """Fill the template. Raises KeyError listing any missing variables."""
        missing = self.required_variables - variables.keys()
        if missing:
            raise KeyError(f"Prompt '{self.name}' is missing variables: {sorted(missing)}")
        # One regex pass: inserted values are never re-scanned for {{...}}.
        return _VAR_PATTERN.sub(lambda m: str(variables[m.group(1)]), self.task_template).strip()


@lru_cache(maxsize=None)
def load_prompt(name: str) -> PromptTemplate:
    """Load 'coordinator', 'explainer', 'quiz_master' or 'evaluator'."""
    path = PROMPT_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = [key for key in _REQUIRED_KEYS if not str(data.get(key, "")).strip()]
    if missing:
        raise ValueError(f"{path.name} is missing required keys: {missing}")
    return PromptTemplate(name=name, **{key: str(data[key]).strip() for key in _REQUIRED_KEYS})


_DEMO_VARS: dict[str, dict[str, Any]] = {
    "coordinator": dict(
        stage="idle", current_topic="", level="beginner", message="teach me fractions"
    ),
    "explainer": dict(
        topic="Fractions", level="beginner", style_hint="", focus="the whole topic", max_words=250
    ),
    "quiz_master": dict(
        num_questions=3,
        topic="Fractions",
        level="beginner",
        difficulty_hint="",
        key_points="- a fraction is part of a whole",
        focus="none",
    ),
    "evaluator": dict(
        score_pct=67,
        correct=2,
        total=3,
        topic="Fractions",
        level="beginner",
        wrong_summary="- simplifying: chose '2/2', correct was '1/2'",
    ),
}

if __name__ == "__main__":
    print(f"{'prompt':<13}{'variables':<11}{'~tokens':<9}role")
    for prompt_name, demo in _DEMO_VARS.items():
        template = load_prompt(prompt_name)
        text = template.render(**demo)
        print(
            f"{prompt_name:<13}{len(template.required_variables):<11}"
            f"{len(text) // 4:<9}{template.role}"
        )
    # Prove that missing variables are caught:
    try:
        load_prompt("explainer").render(topic="x")
    except KeyError as exc:
        print("\nMissing-variable guard works ->", exc)
