"""Compatibility shims for third-party bugs (delete once CrewAI fixes them).

BUG: CrewAI (seen in 1.15.23) tags system/user messages with an internal
"cache_breakpoint" key that only Anthropic understands. On the LiteLLM path that
key reaches Groq, which answers HTTP 400:
    "property 'cache_breakpoint' is unsupported"
FIX: strip the key from the messages right before LiteLLM sends them. Leo only
uses Groq, Gemini and Ollama, so removing it is harmless.

Run `python -m leo.compat` for an offline self-test.
"""

from __future__ import annotations

import functools
import inspect
import sys
from typing import Any, Callable

from leo.logging_setup import get_logger

logger = get_logger("compat")

CACHE_KEY = "cache_breakpoint"
stats: dict[str, Any] = {"stripped_messages": 0, "patched": False}


def strip_cache_breakpoint(messages: Any) -> Any:
    """Return the messages without the internal marker (the input is never modified)."""
    if not isinstance(messages, list):
        return messages
    cleaned = []
    for message in messages:
        if isinstance(message, dict) and CACHE_KEY in message:
            message = {k: v for k, v in message.items() if k != CACHE_KEY}
            if stats["stripped_messages"] == 0:  # log once, not on every call
                logger.info(
                    "Stripping CrewAI's internal '%s' key from outgoing messages "
                    "(Groq rejects it)",
                    CACHE_KEY,
                )
            stats["stripped_messages"] += 1
        cleaned.append(message)
    return cleaned


def _sanitize(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """litellm.completion(model, messages, ...): messages may be positional or keyword."""
    if "messages" in kwargs:
        kwargs = {**kwargs, "messages": strip_cache_breakpoint(kwargs["messages"])}
    elif len(args) >= 2:
        args = (args[0], strip_cache_breakpoint(args[1]), *args[2:])
    return args, kwargs


def _wrap_sync(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        args, kwargs = _sanitize(args, kwargs)
        return func(*args, **kwargs)

    wrapper._leo_patched = True  # type: ignore[attr-defined]
    return wrapper


def _wrap_async(func: Callable) -> Callable:
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        args, kwargs = _sanitize(args, kwargs)
        return await func(*args, **kwargs)

    wrapper._leo_patched = True  # type: ignore[attr-defined]
    return wrapper


def apply_compat_patches() -> bool:
    """Wrap litellm.completion / acompletion once. Safe to call repeatedly."""
    if stats["patched"]:
        return True
    try:
        import litellm
    except ImportError:
        logger.warning("litellm is not installed; compatibility patch skipped")
        return False
    for name in ("completion", "acompletion"):
        original = getattr(litellm, name, None)
        if original is None or getattr(original, "_leo_patched", False):
            continue
        wrapper = (
            _wrap_async(original) if inspect.iscoroutinefunction(original) else _wrap_sync(original)
        )
        setattr(litellm, name, wrapper)
    stats["patched"] = True
    logger.debug("Compatibility patch applied to litellm")
    return True


if __name__ == "__main__":
    sample = [
        {"role": "system", "content": "hi", "cache_breakpoint": True},
        {"role": "user", "content": "yo"},
    ]
    cleaned = strip_cache_breakpoint(sample)
    checks = [
        ("marker removed", all(CACHE_KEY not in m for m in cleaned)),
        ("input list not modified", CACHE_KEY in sample[0]),
        ("other keys kept", cleaned[0]["content"] == "hi"),
        (
            "wrapper strips keyword messages",
            CACHE_KEY not in _wrap_sync(lambda **kw: kw["messages"])(messages=sample)[0],
        ),
        (
            "wrapper strips positional messages",
            CACHE_KEY not in _wrap_sync(lambda model, messages: messages)("m", sample)[0],
        ),
        (
            "patch is applied to litellm",
            apply_compat_patches()
            and getattr(__import__("litellm").completion, "_leo_patched", False),
        ),
    ]
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    sys.exit(0 if all(ok for _, ok in checks) else 1)
