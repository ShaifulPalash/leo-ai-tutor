"""Central logging for Leo.

Usage in any module:
    from leo.logging_setup import get_logger
    logger = get_logger("crew")          # -> logger named "leo.crew"
    agent_logger = get_agent_logger("explainer")   # -> "leo.agent.explainer"
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any

import colorlog

from leo.config import Settings, get_settings

ROOT_LOGGER_NAME = "leo"
HANDOFF_LOGGER_NAME = "leo.handoff"
_configured = False

# Patterns that look like secrets. Anything matching is masked in every log.
_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"gsk_[A-Za-z0-9]{16,}"), "gsk_***REDACTED***"),
    (re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), "AIza***REDACTED***"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{12,}"), r"\1***REDACTED***"),
    (
        re.compile(r"(?i)((?:api[_-]?key|token|secret)[\"']?\s*[:=]\s*[\"']?)[^\s\"',]{8,}"),
        r"\1***REDACTED***",
    ),
]


def redact(text: str) -> str:
    """Mask anything that looks like an API key or token."""
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class SecretRedactionFilter(logging.Filter):
    """Logging filter that scrubs secrets from every record before output."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = redact(record.getMessage())
            record.args = ()  # message is already fully formatted
        except Exception:  # never let logging itself crash the app
            pass
        return True


class HandoffJsonFormatter(logging.Formatter):
    """Writes the structured 'handoff' dict attached to a record as one JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        data = getattr(record, "handoff", {})
        return redact(json.dumps(data, ensure_ascii=False, default=str))


class OnlyHandoffRecords(logging.Filter):
    """Lets only records that carry structured handoff data through."""

    def filter(self, record: logging.LogRecord) -> bool:
        return hasattr(record, "handoff")


def setup_logging(settings: Settings | None = None, force: bool = False) -> logging.Logger:
    """Configure logging once for the whole app (safe to call repeatedly)."""
    global _configured
    settings = settings or get_settings()
    root = logging.getLogger(ROOT_LOGGER_NAME)
    if _configured and not force:
        return root

    # Remove old handlers on forced re-setup to prevent duplicate lines.
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handoff_logger = logging.getLogger(HANDOFF_LOGGER_NAME)
    for handler in list(handoff_logger.handlers):
        handoff_logger.removeHandler(handler)

    level = getattr(logging, settings.effective_log_level)
    root.setLevel(level)
    root.propagate = False  # avoid duplicate lines via Python's root logger
    redactor = SecretRedactionFilter()

    # 1) Colored console handler --------------------------------------
    console = colorlog.StreamHandler(sys.stdout)
    console.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s %(levelname)-8s%(reset)s"
            "%(cyan)s%(name)-24s%(reset)s| %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG": "white",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    )
    console.addFilter(redactor)
    root.addHandler(console)

    # 2) Rotating file handler: max ~1 MB per file, keep 5 old files ----
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        settings.log_dir / "leo.log",
        maxBytes=1_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    file_handler.addFilter(redactor)
    root.addHandler(file_handler)

    # 3) Structured handoff log (one JSON object per line) -------------
    handoff_file = RotatingFileHandler(
        settings.log_dir / "handoffs.jsonl",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    handoff_file.setFormatter(HandoffJsonFormatter())
    handoff_file.addFilter(OnlyHandoffRecords())
    handoff_logger.addHandler(handoff_file)  # also propagates to "leo" (console)

    # 4) Quiet down chatty third-party libraries unless we are debugging.
    noisy_level = logging.DEBUG if settings.debug_mode else logging.WARNING
    for name in ("httpx", "httpcore", "urllib3", "LiteLLM", "litellm", "openai"):
        logging.getLogger(name).setLevel(noisy_level)

    _configured = True
    root.debug("Logging configured (level=%s)", settings.effective_log_level)
    return root


def get_logger(name: str) -> logging.Logger:
    """Get a module logger such as 'leo.crew'. Ensures logging is configured."""
    setup_logging()
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def get_agent_logger(agent: str) -> logging.Logger:
    """Get a per-agent logger such as 'leo.agent.explainer'."""
    return get_logger(f"agent.{agent}")


def log_handoff(
    from_agent: str,
    to_agent: str,
    payload: Any,
    duration_s: float = 0.0,
    tokens: int = 0,
    note: str = "",
) -> dict[str, Any]:
    """Record one agent-to-agent handoff and return it (the UI displays it too)."""
    settings = get_settings()
    setup_logging()
    text = json.dumps(payload, ensure_ascii=False, default=str)
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "from": from_agent,
        "to": to_agent,
        "payload_bytes": len(text.encode("utf-8")),
        "duration_s": round(duration_s, 2),
        "tokens": tokens,
        "note": note,
    }
    if settings.debug_mode:
        # Only in debug mode do we keep a (redacted, truncated) copy of the payload.
        record["payload_preview"] = redact(text[:500])

    logging.getLogger(HANDOFF_LOGGER_NAME).info(
        "HANDOFF %s -> %s | %d bytes | %.2fs | %d tokens %s",
        from_agent,
        to_agent,
        record["payload_bytes"],
        duration_s,
        tokens,
        note,
        extra={"handoff": record},
    )
    return record


if __name__ == "__main__":
    # Demo: proves colors, file logging, handoffs, and key redaction all work.
    demo = get_logger("demo")
    demo.debug("This DEBUG line only shows when DEBUG_MODE=true")
    demo.info("Hello from Leo logging")
    demo.warning("Careful: this is a warning")
    demo.error("Example error")
    fake_key = "gsk_" + "a" * 30  # a FAKE key just to test redaction
    demo.info("Testing redaction with key=%s", fake_key)
    log_handoff(
        "coordinator",
        "explainer",
        {"topic": "fractions", "level": "beginner"},
        duration_s=0.42,
        tokens=120,
        note="demo",
    )
    print("\nCheck the files: logs/leo.log and logs/handoffs.jsonl")
