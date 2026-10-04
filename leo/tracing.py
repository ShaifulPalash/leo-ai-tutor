"""Per-session run trace, saved as JSON in logs/traces/ (debugging aid).

Every agent step, handoff and event is appended in order, so after a run you can
open the file and see exactly what happened, how long it took, and what it cost.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from leo.config import Settings, get_settings
from leo.logging_setup import get_logger, redact

logger = get_logger("tracing")


class RunTrace:
    """An ordered list of what happened during one student session."""

    def __init__(self, student: str, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        now = datetime.now(timezone.utc)
        self.session_id = f"{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        self.started_at = now.isoformat(timespec="seconds")
        self.student = student
        self.entries: list[dict[str, Any]] = []
        self.path: Path = self.settings.trace_dir / f"session-{self.session_id}.json"

    def add(self, kind: str, **data: Any) -> None:
        """Append one record (kind is e.g. 'event' or 'step')."""
        self.entries.append(
            {
                "seq": len(self.entries) + 1,
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "kind": kind,
                **data,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "student": self.student,
            "started_at": self.started_at,
            "debug_mode": self.settings.debug_mode,
            "entries": self.entries,
        }

    def save(self) -> Optional[Path]:
        """Write the trace atomically. Never raises: tracing must not break the app."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            text = redact(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str))
            temp = self.path.with_suffix(".tmp")
            temp.write_text(text, encoding="utf-8")
            os.replace(temp, self.path)  # atomic rename
            return self.path
        except OSError as exc:
            logger.warning("Could not save trace: %s", exc)
            return None
