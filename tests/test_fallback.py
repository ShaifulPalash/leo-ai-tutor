"""Fallback tests: classification, backoff, retries, repair prompts, model switching."""

import time
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from leo import llm_factory
from leo.config import Settings
from leo.llm_factory import (
    LLMFactory,
    LLMUnavailableError,
    RateLimiter,
    backoff_delay,
    classify_error,
    usage_tracker,
)
from leo.schemas import Quiz


def run(factory, operation, **kwargs):
    return factory.run_with_fallback(
        tier="fast", agent="coordinator", operation=operation, max_tokens=100, **kwargs
    )


def invalid_quiz_error() -> Exception:
    try:
        Quiz.model_validate({})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a validation error")


# ---------------------------------------------------------------- classification
@pytest.mark.parametrize(
    "error, expected",
    [
        (ValueError("bad json"), "invalid_output"),
        (invalid_quiz_error(), "invalid_output"),
        (ImportError("native provider missing"), "model_unavailable"),
        (RuntimeError("Error code: 429 - rate limit reached"), "rate_limit"),
        (RuntimeError("RESOURCE_EXHAUSTED: quota exceeded"), "rate_limit"),
        (RuntimeError("HTTP 401 invalid api key"), "model_unavailable"),
        (RuntimeError("403 Forbidden"), "model_unavailable"),
        (RuntimeError("404 NOT_FOUND: model no longer available"), "model_unavailable"),
        (
            RuntimeError("litellm.BadRequestError: property 'cache_breakpoint' is unsupported"),
            "model_unavailable",
        ),
        (RuntimeError("Read timed out"), "transient"),
        (RuntimeError("connection reset by peer"), "transient"),
        (
            RuntimeError("400 INVALID_ARGUMENT. API key not valid. Please pass a valid API key."),
            "model_unavailable",
        ),
        (RuntimeError('GroqException - {"code":"invalid_api_key"}'), "model_unavailable"),
    ],
)
def test_classify_error(error, expected):
    assert classify_error(error) == expected


def test_backoff_grows_is_capped_and_honours_hints():
    assert 1.0 <= backoff_delay(0) <= 1.5
    assert backoff_delay(3) >= 8.0
    assert backoff_delay(10) <= 20.5  # capped
    assert backoff_delay(0, RuntimeError("Please retry in 12.5s")) >= 12.5
    assert backoff_delay(0, RuntimeError("try again in 2.5s")) >= 2.5


def test_rate_limiter_waits_when_the_window_is_full(monkeypatch):
    class Clock:
        now = 1000.0
        slept: list[float] = []

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.slept.append(seconds)
            self.now += seconds

    clock = Clock()
    monkeypatch.setattr(
        llm_factory,
        "time",
        SimpleNamespace(
            monotonic=clock.monotonic, sleep=clock.sleep, perf_counter=time.perf_counter
        ),
    )
    limiter = RateLimiter(2, sleep=clock.sleep)
    for _ in range(3):  # the third call must wait for the 60 s window to move on
        limiter.acquire()
    assert len(clock.slept) == 1 and clock.slept[0] >= 60


# ---------------------------------------------------------------- run_with_fallback
def test_success_on_first_try_records_usage(factory):
    def op(ctx):
        ctx.set_usage(10, 5)
        return "ok"

    result = run(factory, op)
    assert result.value == "ok" and result.attempts == 1
    assert result.model == factory.model_chain("fast")[0]
    assert result.used_fallback_model is False and result.total_tokens == 15
    assert usage_tracker.totals()["total_tokens"] == 15


def test_rate_limit_waits_then_retries_same_model(factory):
    calls = {"n": 0}

    def op(ctx):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Error code: 429 - rate limit reached")
        return "ok"

    result = run(factory, op)
    assert result.value == "ok" and result.attempts == 2
    assert len(factory.sleeps) == 1 and factory.sleeps[0] >= 1.0
    assert result.used_fallback_model is False


def test_long_provider_wait_switches_model_instead_of_waiting(factory):
    def op(ctx):
        if ctx.model.startswith("groq/"):
            raise RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 45.0s")
        return "ok"

    result = run(factory, op)
    assert result.model.startswith("gemini/") and result.used_fallback_model is True
    assert factory.sleeps == []  # it did NOT sit and wait


def test_short_provider_wait_is_honoured(factory):
    calls = {"n": 0}

    def op(ctx):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("429 rate limit. Please try again in 2.5s")
        return "ok"

    run(factory, op)
    assert factory.sleeps[0] >= 2.5


def test_bad_key_skips_to_fallback_without_waiting(factory):
    def op(ctx):
        if ctx.model.startswith("groq/"):
            raise RuntimeError("HTTP 401 invalid api key")
        return "ok"

    result = run(factory, op)
    assert result.model.startswith("gemini/") and result.attempts == 2
    assert factory.sleeps == []


def test_bad_request_skips_to_fallback_without_waiting(factory):
    def op(ctx):
        if ctx.model.startswith("groq/"):
            raise RuntimeError("litellm.BadRequestError: cache_breakpoint is unsupported")
        return "ok"

    result = run(factory, op)
    assert result.used_fallback_model is True and factory.sleeps == []


def test_invalid_json_gets_one_repair_retry(factory):
    seen = []

    def op(ctx):
        seen.append(ctx.repair_hint)
        if len(seen) == 1:
            raise ValueError("no JSON object found")
        assert "PROMPT" in ctx.with_repair("PROMPT")
        assert "ONLY valid JSON" in ctx.with_repair("PROMPT")
        return "fixed"

    result = run(factory, op)
    assert result.value == "fixed" and result.used_fallback_model is False
    assert seen[0] is None and seen[1] is not None  # the repair hint appeared on retry


def test_persistent_invalid_json_moves_to_next_model(factory):
    def op(ctx):
        if ctx.model.startswith("groq/"):
            raise ValueError("still not JSON")
        return "ok"

    result = run(factory, op)
    assert result.model.startswith("gemini/") and result.attempts == 3


def test_transient_errors_retry_with_backoff_then_switch(factory):
    def op(ctx):
        if ctx.model.startswith("groq/"):
            raise RuntimeError("connection reset by peer")
        return "ok"

    result = run(factory, op)
    assert result.model.startswith("gemini/")
    assert len(factory.sleeps) == 2  # max_retries=2 -> two waits before giving up


def test_all_models_fail_returns_safe_default(factory):
    def op(ctx):
        raise RuntimeError("503 service unavailable")

    result = run(factory, op, safe_default=lambda: "safe message")
    assert result.value == "safe message"
    assert result.used_safe_default is True and result.model == "none"


def test_all_models_fail_without_default_raises(factory):
    def op(ctx):
        raise RuntimeError("503 service unavailable")

    with pytest.raises(LLMUnavailableError):
        run(factory, op)


def test_no_usable_model_raises_clear_error():
    no_keys = Settings(_env_file=None, groq_api_key="", gemini_api_key="")
    with pytest.raises(LLMUnavailableError, match="No usable model"):
        LLMFactory(no_keys).run_with_fallback(
            tier="fast", agent="coordinator", operation=lambda ctx: "x", max_tokens=100
        )


def test_all_invalid_keys_are_reported_as_a_config_problem(factory):
    def op(ctx):
        raise RuntimeError("HTTP 401 invalid api key")

    with pytest.raises(LLMUnavailableError) as info:
        run(factory, op)
    assert info.value.kind == "config" and factory.sleeps == []


def test_capacity_failures_are_reported_as_busy(factory):
    def op(ctx):
        raise RuntimeError("503 service unavailable")

    with pytest.raises(LLMUnavailableError) as info:
        run(factory, op)
    assert info.value.kind == "busy"
