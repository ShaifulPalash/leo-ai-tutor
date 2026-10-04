"""Pydantic schemas: the 'contracts' between Leo's agents.

Every agent returns one of these models instead of free text. Pydantic
validates the shape, cleans harmless quirks, and raises ValidationError
(a ValueError) on broken output so the caller can retry.

Run `python -m leo.schemas` for a self-demo.
"""

from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

# ----------------------------------------------------------------------
# Shared types and small helpers
# ----------------------------------------------------------------------
Level = Literal["beginner", "intermediate", "advanced"]
LEVELS: tuple[str, ...] = ("beginner", "intermediate", "advanced")

Intent = Literal[
    "learn", "change_topic", "simpler", "harder_quiz", "skip_to_quiz", "clarify", "refuse"
]

DEFAULT_CLARIFY = "Which topic would you like to study? For example: 'photosynthesis'."
DEFAULT_REFUSE = (
    "I can only help with study topics. " "Tell me a subject you'd like to learn and we'll start!"
)


def normalize_topic(text: str) -> str:
    """Lowercase, trim, collapse spaces. Used for cache keys and comparisons."""
    text = re.sub(r"\s+", " ", (text or "").strip().lower())
    return text.strip(" .,:;!?\"'")


def clean_text(text: str, max_len: int) -> str:
    """Collapse whitespace and cap the length (a cheap guard on huge inputs)."""
    return re.sub(r"\s+", " ", (text or "").strip())[:max_len]


def truncate_words(text: str, max_words: int) -> str:
    """Trim text to max_words (safety net if the model ignores the word limit)."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


def _coerce_level(value: object, default: Optional[str]) -> Optional[str]:
    """'Beginner ' -> 'beginner'. Unknown values fall back to `default`."""
    if isinstance(value, str):
        value = value.strip().lower()
        if value in LEVELS:
            return value
    return default


class LeoModel(BaseModel):
    """Base class: ignore unknown keys the LLM adds, and trim whitespace."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


# ----------------------------------------------------------------------
# 1) Coordinator output
# ----------------------------------------------------------------------
class RoutingDecision(LeoModel):
    """What the Coordinator decided to do with the student's message."""

    intent: Intent
    topic: str = ""
    level: Optional[Level] = None
    reply: str = ""  # used for 'clarify' and 'refuse'

    @field_validator("intent", mode="before")
    @classmethod
    def _norm_intent(cls, v: object) -> object:
        # 'Skip to Quiz' -> 'skip_to_quiz'
        return re.sub(r"[\s\-]+", "_", v.strip().lower()) if isinstance(v, str) else v

    @field_validator("level", mode="before")
    @classmethod
    def _norm_level(cls, v: object) -> Optional[str]:
        return _coerce_level(v, None)  # empty or unknown -> None

    @field_validator("topic")
    @classmethod
    def _clean_topic(cls, v: str) -> str:
        return clean_text(v, 80)

    @model_validator(mode="after")
    def _repair(self) -> "RoutingDecision":
        # A 'learn' request with no topic is really an unclear request.
        if self.intent in ("learn", "change_topic") and not self.topic:
            self.intent = "clarify"
        if self.intent == "clarify" and not self.reply:
            self.reply = DEFAULT_CLARIFY
        if self.intent == "refuse" and not self.reply:
            self.reply = DEFAULT_REFUSE
        return self


# ----------------------------------------------------------------------
# 2) Explainer output
# ----------------------------------------------------------------------
class Explanation(LeoModel):
    """A short lesson on one topic."""

    topic: str = ""
    explanation: str = Field(min_length=20)
    key_points: list[str] = Field(min_length=2, max_length=6)
    example: str = ""

    @field_validator("explanation")
    @classmethod
    def _limit_length(cls, v: str) -> str:
        return truncate_words(v, 350)  # prompt asks <= 250; this is a safety net

    @property
    def word_count(self) -> int:
        return len(self.explanation.split())

    def to_quiz_brief(self) -> dict:
        """The SMALL payload handed to the Quiz Master (saves tokens).

        The Quiz Master does not need the whole lesson, only the topic and key points.
        """
        return {"topic": self.topic, "key_points": self.key_points}


# ----------------------------------------------------------------------
# 3) Quiz Master output
# ----------------------------------------------------------------------
class QuizQuestion(LeoModel):
    """One multiple-choice question with exactly four options."""

    id: int = Field(ge=1)
    question: str = Field(min_length=5)
    options: list[str] = Field(min_length=4, max_length=4)
    correct_index: int = Field(ge=0, le=3)
    concept: str = Field(min_length=2)  # label used to track weak areas
    explanation: str = ""

    @field_validator("options", mode="before")
    @classmethod
    def _strip_option_labels(cls, v: object) -> object:
        # 'A) 4' / 'b. 3' -> '4' / '3'
        if isinstance(v, list):
            return [re.sub(r"^\s*[A-Da-d][\)\.:]\s+", "", str(o)).strip() for o in v]
        return v

    @field_validator("options")
    @classmethod
    def _options_distinct(cls, v: list[str]) -> list[str]:
        if len({o.lower() for o in v}) != len(v) or any(not o for o in v):
            raise ValueError("options must be four distinct, non-empty answers")
        return v

    @field_validator("correct_index", mode="before")
    @classmethod
    def _letter_to_index(cls, v: object) -> object:
        # Some models answer 'B' instead of 1.
        if isinstance(v, str):
            s = v.strip().upper()
            if len(s) == 1 and s in "ABCD":
                return "ABCD".index(s)
        return v


class Quiz(LeoModel):
    """A set of 2-5 questions (2-3 for a mini re-quiz)."""

    topic: str = ""
    level: Level = "beginner"
    questions: list[QuizQuestion] = Field(min_length=2, max_length=5)

    @field_validator("level", mode="before")
    @classmethod
    def _norm_level(cls, v: object) -> str:
        return _coerce_level(v, "beginner") or "beginner"

    @model_validator(mode="after")
    def _renumber(self) -> "Quiz":
        # Force ids 1..n so duplicate or odd ids from the LLM cannot break grading.
        for i, question in enumerate(self.questions, start=1):
            question.id = i
        return self

    def public_view(self) -> list[dict]:
        """Questions WITHOUT answers. This is what the student (and UI) sees."""
        return [{"id": q.id, "question": q.question, "options": q.options} for q in self.questions]


# ----------------------------------------------------------------------
# 4) Grading (plain Python, 0 tokens) and Evaluator output
# ----------------------------------------------------------------------
class QuestionResult(LeoModel):
    question_id: int
    concept: str
    correct: bool
    chosen_index: int = Field(ge=-1, le=3)  # -1 means unanswered
    correct_index: int = Field(ge=0, le=3)


class GradeResult(LeoModel):
    """Objective result of comparing the student's answers to the answer key."""

    score: float = Field(ge=0.0, le=1.0)
    correct_count: int
    total: int
    results: list[QuestionResult]
    weak_concepts: list[str] = []
    strong_concepts: list[str] = []


def grade_quiz(quiz: Quiz, answers: dict[int, int]) -> GradeResult:
    """Grade answers ({question_id: chosen_index}) against the quiz. Costs 0 tokens."""
    results: list[QuestionResult] = []
    for q in quiz.questions:
        chosen = answers.get(q.id, -1)
        if not 0 <= chosen <= 3:
            chosen = -1  # treat invalid input as unanswered
        results.append(
            QuestionResult(
                question_id=q.id,
                concept=q.concept,
                correct=(chosen == q.correct_index),
                chosen_index=chosen,
                correct_index=q.correct_index,
            )
        )

    weak: list[str] = []
    for r in results:
        if not r.correct and r.concept not in weak:
            weak.append(r.concept)
    strong = [r.concept for r in results if r.correct and r.concept not in weak]
    strong = list(dict.fromkeys(strong))  # unique, keep order

    correct_count = sum(r.correct for r in results)
    total = len(results)
    return GradeResult(
        score=round(correct_count / total, 2),
        correct_count=correct_count,
        total=total,
        results=results,
        weak_concepts=weak,
        strong_concepts=strong,
    )


class ConceptTip(LeoModel):
    concept: str
    tip: str


class FeedbackNote(LeoModel):
    """What the Evaluator's LLM writes: encouragement plus one tip per weak concept."""

    overall: str = Field(min_length=3)
    tips: list[ConceptTip] = Field(default_factory=list, max_length=5)

    @field_validator("overall")
    @classmethod
    def _limit(cls, v: str) -> str:
        return truncate_words(v, 80)


def default_feedback(grade: GradeResult) -> FeedbackNote:
    """Safe feedback used when the LLM is unavailable (the last fallback)."""
    pct = round(grade.score * 100)
    if grade.weak_concepts:
        overall = f"You scored {pct}%. Let's review: {', '.join(grade.weak_concepts[:3])}."
    else:
        overall = f"Great work! You scored {pct}%. Keep it up."
    tips = [
        ConceptTip(concept=c, tip="Re-read this part of the lesson, then try again.")
        for c in grade.weak_concepts[:4]
    ]
    return FeedbackNote(overall=overall, tips=tips)


class EvaluationReport(LeoModel):
    """The Evaluator's final output: objective grade + friendly feedback + decision."""

    grade: GradeResult
    feedback: FeedbackNote
    threshold: float
    needs_reteach: bool

    @property
    def passed(self) -> bool:
        return not self.needs_reteach

    @classmethod
    def from_parts(
        cls, grade: GradeResult, feedback: FeedbackNote, threshold: float
    ) -> "EvaluationReport":
        """Combine grade and feedback; below the threshold triggers the re-teach loop."""
        return cls(
            grade=grade,
            feedback=feedback,
            threshold=threshold,
            needs_reteach=grade.score < threshold,
        )


# ----------------------------------------------------------------------
# 5) What memory stores about a student
# ----------------------------------------------------------------------
class StudentProfile(LeoModel):
    """Compact memory: NOT a transcript, just a few facts."""

    name: str = "Student"
    topic: str = ""
    level: Level = "beginner"
    weak_areas: list[str] = Field(default_factory=list)
    last_score: Optional[float] = None
    sessions: int = 0
    updated_at: str = ""

    @field_validator("name", mode="before")
    @classmethod
    def _clean_name(cls, v: object) -> str:
        cleaned = clean_text(str(v or ""), 40)
        return cleaned or "Student"

    def summary(self) -> str:
        score = "n/a" if self.last_score is None else f"{round(self.last_score * 100)}%"
        weak = ", ".join(self.weak_areas[:5]) or "none"
        return (
            f"{self.name} | topic: {self.topic or '-'} | level: {self.level} "
            f"| weak: {weak} | last score: {score}"
        )


# ----------------------------------------------------------------------
# Self-demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # A deliberately MESSY quiz, like a real LLM might return.
    raw_quiz = {
        "topic": "Fractions",
        "level": "Beginner",
        "questions": [
            {
                "id": 7,
                "question": "Which number is the numerator in 3/4?",
                "options": ["A) 4", "B) 3", "C) 7", "D) 1"],
                "correct_index": "B",
                "concept": "numerator",
            },
            {
                "id": 7,
                "question": "What is 2/4 in simplest form?",
                "options": ["1/2", "2/2", "4/2", "1/4"],
                "correct_index": 0,
                "concept": "simplifying",
            },
            {
                "id": 7,
                "question": "Which fraction is the largest?",
                "options": ["1/8", "1/2", "1/4", "1/3"],
                "correct_index": 1,
                "concept": "comparing",
                "extra_field": "ignored",
            },
        ],
    }
    quiz = Quiz.model_validate(raw_quiz)
    print(
        f"Quiz accepted: {len(quiz.questions)} questions, level={quiz.level}, "
        f"ids={[q.id for q in quiz.questions]}"
    )
    print(
        f"Q1 options cleaned: {quiz.questions[0].options}, "
        f"correct_index={quiz.questions[0].correct_index}"
    )

    # Student answers: Q1 right (1), Q2 wrong (3), Q3 right (1).
    grade = grade_quiz(quiz, {1: 1, 2: 3, 3: 1})
    print(
        f"Score: {round(grade.score * 100)}% ({grade.correct_count}/{grade.total}), "
        f"weak={grade.weak_concepts}, strong={grade.strong_concepts}"
    )
    report = EvaluationReport.from_parts(grade, default_feedback(grade), threshold=0.7)
    print(f"Needs re-teach at threshold 70%? {report.needs_reteach}")

    decision = RoutingDecision.model_validate({"intent": "Learn", "topic": ""})
    print(f"Routing 'learn' with no topic became: {decision.intent} -> {decision.reply!r}")

    try:
        Quiz.model_validate(
            {
                "questions": [
                    {
                        "id": 1,
                        "question": "Only three options?",
                        "options": ["a", "b", "c"],
                        "correct_index": 0,
                        "concept": "x",
                    }
                ]
                * 2
            }
        )
    except ValidationError as exc:
        print(f"Broken quiz rejected: {exc.error_count()} validation error(s)")
