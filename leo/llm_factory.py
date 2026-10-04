"""LLM factory: builds models, rate-limits, retries, falls back, counts tokens.

Run these to test:
    python -m leo.llm_factory --dry-run   # shows the fallback chains, no API calls
    python -m leo.llm_factory             # sends ONE tiny test prompt per model
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Callable, Generic, Optional, TypeVar

from leo.config import Settings, Tier, get_settings
from leo.logging_setup import get_logger, redact

logger = get_logger("llm")
T = TypeVar("T")


class LLMUnavailableError(RuntimeError):
    """Raised when every model in the fallback chain has failed."""

    def __init__(self, message: str, kind: str = "busy") -> None:
        super().__init__(message)
        self.kind = kind  # "config" = keys/model names rejected, "busy" = anything else


# ----------------------------------------------------------------------
# Token and latency tracking (shown in the UI and logs)
# ----------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per token) when the API gives none."""
    return max(1, len(text) // 4)


@dataclass
class CallRecord:
    """One LLM attempt: who called, which model, cost, speed, and outcome."""

    agent: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    ok: bool
    error: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class UsageTracker:
    """Thread-safe running total of all LLM calls in this session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[CallRecord] = []

    def record(self, rec: CallRecord) -> None:
        with self._lock:
            self._records.append(rec)

    def reset(self) -> None:
        with self._lock:
            self._records.clear()

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return [{**asdict(r), "total_tokens": r.total_tokens} for r in self._records]

    def totals(self) -> dict[str, Any]:
        """Aggregate numbers for the UI counter."""
        with self._lock:
            recs = list(self._records)
        ok = [r for r in recs if r.ok]
        return {
            "calls": len(recs),
            "successful_calls": len(ok),
            "failed_calls": len(recs) - len(ok),
            "prompt_tokens": sum(r.prompt_tokens for r in ok),
            "completion_tokens": sum(r.completion_tokens for r in ok),
            "total_tokens": sum(r.total_tokens for r in ok),
            "total_latency_s": round(sum(r.latency_s for r in recs), 2),
        }


usage_tracker = UsageTracker()  # one shared tracker for the whole app


# ----------------------------------------------------------------------
# Error classification: decides what to do after a failure
# ----------------------------------------------------------------------
def classify_error(exc: BaseException) -> str:
    """Return one of: 'invalid_output', 'model_unavailable', 'rate_limit', 'transient'.

    - invalid_output    : the model answered but the JSON/schema was wrong
    - model_unavailable : bad key, forbidden, retired model, missing package
                          -> retrying the SAME model is pointless, so skip it
    - rate_limit        : HTTP 429 / quota -> wait (backoff) and retry
    - transient         : timeout / 5xx / network blip -> wait and retry
    """
    if isinstance(exc, ImportError):
        return "model_unavailable"
    # pydantic.ValidationError and json.JSONDecodeError are both ValueError subclasses.
    if isinstance(exc, ValueError):
        return "invalid_output"

    text = f"{type(exc).__name__} {exc}".lower()
    if (
        re.search(r"\b429\b", text)
        or "rate limit" in text
        or "ratelimit" in text
        or "quota" in text
        or "resource_exhausted" in text
    ):
        return "rate_limit"
    if (
        re.search(r"\b(401|403)\b", text)
        or "invalid api key" in text
        or "authentication" in text
        or "permission" in text
    ):
        return "model_unavailable"
    # HTTP 400 / rejected key: the SAME request fails again, so skip this model at once.
    if (
        "badrequest" in text
        or "bad request" in text
        or "invalid_request_error" in text
        or "invalid_argument" in text
        or "api key not valid" in text
        or "invalid_api_key" in text
        or "api_key_invalid" in text
    ):
        return "model_unavailable"
    # 404 / retired / closed-to-new-users models can never succeed: skip them.
    if re.search(r"\b404\b", text) or "not_found" in text or "no longer available" in text:
        return "model_unavailable"
    if "model" in text and any(
        k in text for k in ("not found", "decommission", "does not exist", "deprecated")
    ):
        return "model_unavailable"
    return "transient"


def _retry_after_seconds(exc: BaseException | None) -> Optional[float]:
    """Parse hints like 'try again in 12.5s' from provider error messages."""
    if exc is None:
        return None
    match = re.search(r"(?:try again|retry) in\s+(?:(\d+)m)?\s*(\d+(?:\.\d+)?)s", str(exc).lower())
    if not match:
        return None
    return int(match.group(1) or 0) * 60 + float(match.group(2))


def backoff_delay(
    attempt: int, exc: BaseException | None = None, base: float = 1.0, cap: float = 20.0
) -> float:
    """Exponential backoff: 1s, 2s, 4s, ... capped, plus random jitter.

    If the provider told us how long to wait, we honour that (up to 30s).
    """
    delay = min(cap, base * (2**attempt))
    hinted = _retry_after_seconds(exc)
    if hinted is not None:
        delay = max(delay, min(hinted, 30.0))
    return delay + random.uniform(0, 0.5)


# ----------------------------------------------------------------------
# Rate limiter (protects free-tier requests-per-minute)
# ----------------------------------------------------------------------
class RateLimiter:
    """Sliding-window limiter: at most `max_rpm` calls in any 60 seconds."""

    def __init__(self, max_rpm: int, sleep: Callable[[float], None] = time.sleep) -> None:
        self.max_rpm = max_rpm
        self._stamps: deque[float] = deque()
        self._lock = threading.Lock()
        self._sleep = sleep

    def acquire(self) -> None:
        """Block until making one more call is allowed."""
        while True:
            with self._lock:
                now = time.monotonic()
                while self._stamps and now - self._stamps[0] >= 60:
                    self._stamps.popleft()
                if len(self._stamps) < self.max_rpm:
                    self._stamps.append(now)
                    return
                wait = 60 - (now - self._stamps[0]) + 0.05
            logger.info("Rate limiter: pausing %.1fs to stay under %d RPM", wait, self.max_rpm)
            self._sleep(wait)


# ----------------------------------------------------------------------
# Data passed to / returned from a guarded call
# ----------------------------------------------------------------------
@dataclass
class CallContext:
    """What an operation receives for each attempt."""

    model: str
    llm: Any  # a crewai.LLM instance
    attempt: int
    repair_hint: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def with_repair(self, prompt: str) -> str:
        """Append the repair hint (if a previous attempt gave invalid output)."""
        return f"{prompt}\n\n{self.repair_hint}" if self.repair_hint else prompt

    def set_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Report exact token usage if the framework provides it."""
        self.prompt_tokens, self.completion_tokens = prompt_tokens, completion_tokens

    def estimate_usage(self, prompt_text: str, output_text: str) -> None:
        """Fallback: estimate tokens from text length."""
        self.prompt_tokens = estimate_tokens(prompt_text)
        self.completion_tokens = estimate_tokens(output_text)


@dataclass
class LLMResult(Generic[T]):
    """What run_with_fallback returns."""

    value: T
    model: str
    tier: str
    agent: str
    attempts: int
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    used_fallback_model: bool
    used_safe_default: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ----------------------------------------------------------------------
# The factory
# ----------------------------------------------------------------------
class LLMFactory:
    """Builds CrewAI LLM objects and runs guarded calls against a fallback chain."""

    def __init__(
        self, settings: Settings | None = None, sleep: Callable[[float], None] = time.sleep
    ) -> None:
        self.settings = settings or get_settings()
        self._sleep = sleep  # injectable so tests don't really wait
        self.limiter = RateLimiter(self.settings.max_rpm, sleep=sleep)
        self._cache: dict[tuple[str, int], Any] = {}

    def model_chain(self, tier: Tier) -> list[str]:
        return self.settings.model_chain(tier)

    def build_llm(self, model: str, max_tokens: int) -> Any:
        """Create (or reuse) a crewai.LLM for one model."""
        key = (model, max_tokens)
        if key in self._cache:
            return self._cache[key]

        from crewai import LLM  # imported lazily so tests can run without loading CrewAI

        from leo.compat import apply_compat_patches

        apply_compat_patches()  # work around CrewAI's cache_breakpoint bug with Groq

        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": self.settings.temperature,
            "max_tokens": max_tokens,
            "timeout": self.settings.llm_timeout_seconds,
        }
        api_key = self.settings.api_key_for(model)
        if api_key:
            kwargs["api_key"] = api_key
        if self.settings.provider_of(model) == "ollama":
            kwargs["base_url"] = self.settings.ollama_base_url

        # gpt-oss models "think" before answering. Low effort saves tokens.
        if "gpt-oss" in model:
            try:
                llm = LLM(**kwargs, reasoning_effort="low")
            except TypeError:
                logger.debug("This CrewAI version has no reasoning_effort; continuing without")
                llm = LLM(**kwargs)
        else:
            llm = LLM(**kwargs)

        self._cache[key] = llm
        return llm

    def run_with_fallback(
        self,
        *,
        tier: Tier,
        agent: str,
        operation: Callable[[CallContext], T],
        max_tokens: int,
        safe_default: Optional[Callable[[], T]] = None,
    ) -> LLMResult[T]:
        """Run `operation` with retries, backoff, repair prompts and model fallback.

        `operation(ctx)` must perform the call using ctx.llm, validate its own
        output (raising ValueError/ValidationError if invalid), and optionally
        report tokens via ctx.set_usage() / ctx.estimate_usage().
        """
        chain = self.model_chain(tier)
        if not chain:
            raise LLMUnavailableError(f"No usable model for tier '{tier}'. Check API keys in .env.")

        started_all = time.perf_counter()
        attempts_total = 0
        last_error: BaseException | None = None
        repair_hint: Optional[str] = None
        failure_kinds: set[str] = set()

        for model_index, model in enumerate(chain):
            for attempt in range(self.settings.max_retries + 1):
                self.limiter.acquire()
                attempts_total += 1
                started = time.perf_counter()
                ctx: CallContext | None = None
                try:
                    ctx = CallContext(
                        model=model,
                        llm=self.build_llm(model, max_tokens),
                        attempt=attempt,
                        repair_hint=repair_hint,
                    )
                    value = operation(ctx)
                except Exception as exc:  # noqa: BLE001 - we classify every failure
                    latency = time.perf_counter() - started
                    kind = classify_error(exc)
                    last_error = exc
                    failure_kinds.add(kind)
                    usage_tracker.record(
                        CallRecord(
                            agent, model, 0, 0, latency, False, f"{kind}: {redact(str(exc))[:200]}"
                        )
                    )
                    logger.warning(
                        "[%s] %s attempt %d failed (%s): %s",
                        agent,
                        model,
                        attempt + 1,
                        kind,
                        redact(str(exc))[:200],
                    )

                    if kind == "model_unavailable":
                        break  # skip straight to next model
                    if kind == "rate_limit" and (_retry_after_seconds(exc) or 0) > 30:
                        break  # long wait: use the next model instead
                    if kind == "invalid_output":
                        repair_hint = (
                            "Your previous reply was invalid "
                            f"({redact(str(exc))[:150]}). Reply again with ONLY "
                            "valid JSON that matches the requested schema."
                        )
                        if attempt >= 1:  # one repair try per model
                            break
                        continue
                    if attempt < self.settings.max_retries:  # rate_limit / transient
                        delay = backoff_delay(attempt, exc)
                        logger.info("[%s] backing off %.1fs before retry", agent, delay)
                        self._sleep(delay)
                        continue
                    break
                else:
                    latency = time.perf_counter() - started
                    usage_tracker.record(
                        CallRecord(
                            agent, model, ctx.prompt_tokens, ctx.completion_tokens, latency, True
                        )
                    )
                    logger.info(
                        "[%s] OK via %s in %.2fs (%d tokens)",
                        agent,
                        model,
                        latency,
                        ctx.prompt_tokens + ctx.completion_tokens,
                    )
                    return LLMResult(
                        value=value,
                        model=model,
                        tier=tier,
                        agent=agent,
                        attempts=attempts_total,
                        latency_s=time.perf_counter() - started_all,
                        prompt_tokens=ctx.prompt_tokens,
                        completion_tokens=ctx.completion_tokens,
                        used_fallback_model=model_index > 0,
                    )
            if model_index < len(chain) - 1:
                logger.warning(
                    "[%s] giving up on %s, switching to %s", agent, model, chain[model_index + 1]
                )

        # Every model failed.
        logger.error("[%s] ALL models failed. Last error: %s", agent, redact(str(last_error))[:200])
        if safe_default is not None:
            return LLMResult(
                value=safe_default(),
                model="none",
                tier=tier,
                agent=agent,
                attempts=attempts_total,
                latency_s=time.perf_counter() - started_all,
                prompt_tokens=0,
                completion_tokens=0,
                used_fallback_model=True,
                used_safe_default=True,
            )
        config_problem = failure_kinds == {"model_unavailable"}
        raise LLMUnavailableError(
            f"All models failed for {agent}", kind="config" if config_problem else "busy"
        ) from last_error


_factory: LLMFactory | None = None


def get_factory() -> LLMFactory:
    """Return the shared factory (created on first use)."""
    global _factory
    if _factory is None:
        _factory = LLMFactory()
    return _factory


# ----------------------------------------------------------------------
# Command-line smoke test
# ----------------------------------------------------------------------
def _smoke_test(dry_run: bool) -> int:
    settings = get_settings()
    factory = LLMFactory(settings)
    exit_code = 0
    for tier in ("fast", "smart"):
        chain = factory.model_chain(tier)  # type: ignore[arg-type]
        print(f"\n{tier.upper()} chain: {' -> '.join(chain) or '(EMPTY - check keys!)'}")
        if not chain:
            exit_code = 1
        if dry_run:
            continue
        working = 0
        for model in chain:
            started = time.perf_counter()
            try:
                llm = factory.build_llm(model, max_tokens=300)
                reply = str(llm.call("Reply with exactly one word: OK"))
                print(
                    f"  [ OK ] {model}  ({time.perf_counter() - started:.1f}s) "
                    f"-> {reply.strip()[:40]!r}"
                )
                working += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  [FAIL] {model}: {type(exc).__name__}: {redact(str(exc))[:250]}")
        if working == 0:
            exit_code = 1
    print("\nDone." if exit_code == 0 else "\nSome tiers have no working model.")
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leo LLM factory smoke test")
    parser.add_argument(
        "--dry-run", action="store_true", help="print fallback chains only; make no API calls"
    )
    sys.exit(_smoke_test(parser.parse_args().dry_run))
