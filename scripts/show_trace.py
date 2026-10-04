"""Pretty-print a Leo run trace (the JSON files in logs/traces/).

python scripts/show_trace.py              # the newest trace
python scripts/show_trace.py some.json    # a specific one
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TRACE_DIR = Path(__file__).resolve().parent.parent / "logs" / "traces"


def newest_trace() -> Path | None:
    files = sorted(TRACE_DIR.glob("session-*.json"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else newest_trace()
    if path is None or not path.exists():
        print("No trace found. Run a session first, e.g.: python -m leo.cli --offline")
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"Session {data['session_id']}  student={data['student']}  debug={data['debug_mode']}")
    print(f"File: {path}\n")

    tokens = 0
    seconds = 0.0
    llm_steps = 0
    for entry in data["entries"]:
        number = f"#{entry['seq']:02d}"
        if entry["kind"] == "event":
            clock = entry.get("time", "")
            if entry["status"] == "handoff":
                payload = json.dumps(entry.get("payload", {}), ensure_ascii=False)[:90]
                print(f"{number} {clock}  >>> {entry['agent']} -> {entry['to_agent']}  {payload}")
            else:
                print(
                    f"{number} {clock}  {entry['agent']:<12} {entry['status']:<8} "
                    f"{entry['detail'][:75]}"
                )
        elif entry["kind"] == "step":
            flags = []
            if entry.get("fallback"):
                flags.append("FALLBACK")
            if entry.get("safe_default"):
                flags.append("SAFE-DEFAULT")
            print(
                f"{number}          LLM call: {entry['model']}  attempts={entry['attempts']}  "
                f"{entry['tokens']} tokens  {entry['latency_s']}s  {' '.join(flags)}"
            )
            tokens += entry["tokens"]
            seconds += entry["latency_s"]
            llm_steps += 1

    print(f"\nTotals: {llm_steps} LLM steps, {tokens} tokens, {seconds:.1f}s of LLM time")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
