"""Zero-token setup checker for Leo AI Tutor.

What it does:
  1. Confirms your Python version is supported by CrewAI (3.10 - 3.13).
  2. Confirms the key packages are installed and prints their versions.
  3. Confirms your API keys work by LISTING models (no tokens are spent).
  4. Confirms the model names in your .env actually exist for your key.

Run:  python scripts/check_setup.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from importlib import metadata

from dotenv import load_dotenv

# Read the .env file in the current folder into environment variables.
load_dotenv()

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"
MODEL_VARS = [
    "FAST_MODEL_PRIMARY",
    "FAST_MODEL_FALLBACK",
    "SMART_MODEL_PRIMARY",
    "SMART_MODEL_FALLBACK",
]


def ok(msg: str) -> None:
    print(f"{GREEN}[ OK ]{RESET} {msg}")


def bad(msg: str) -> None:
    print(f"{RED}[FAIL]{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}[WARN]{RESET} {msg}")


def check_python() -> bool:
    """CrewAI supports Python >=3.10 and <3.14."""
    version = sys.version_info[:2]
    label = f"{version[0]}.{version[1]}"
    if (3, 10) <= version < (3, 14):
        ok(f"Python {label} is supported")
        return True
    bad(f"Python {label} is NOT supported. Use 3.10 - 3.13")
    return False


def check_packages() -> bool:
    """Print the installed version of each key package."""
    all_found = True
    for pkg in [
        "crewai",
        "litellm",
        "pydantic",
        "pydantic-settings",
        "streamlit",
        "PyYAML",
        "colorlog",
        "ddgs",
    ]:
        try:
            ok(f"{pkg} {metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            bad(f"{pkg} is not installed (run: pip install -r requirements-dev.txt)")
            all_found = False
    return all_found


def fetch_json(url: str, headers: dict[str, str]) -> dict:
    """GET a URL and return parsed JSON. A User-Agent avoids some 403 blocks."""
    headers = {"User-Agent": "leo-setup-check/1.0", **headers}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def list_models(provider: str) -> set[str] | None:
    """Return the model IDs a provider offers for your key, or None on failure."""
    try:
        if provider == "groq":
            key = os.getenv("GROQ_API_KEY", "")
            if not key or "your_" in key:
                warn("GROQ_API_KEY is not set in .env")
                return None
            data = fetch_json(
                "https://api.groq.com/openai/v1/models",
                {"Authorization": f"Bearer {key}"},
            )
            return {m["id"] for m in data.get("data", [])}
        if provider == "gemini":
            key = os.getenv("GEMINI_API_KEY", "")
            if not key or "your_" in key:
                warn("GEMINI_API_KEY is not set in .env")
                return None
            data = fetch_json(
                "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                {"x-goog-api-key": key},
            )
            # Gemini returns names like "models/gemini-2.5-flash"; strip the prefix.
            return {m["name"].removeprefix("models/") for m in data.get("models", [])}
    except urllib.error.HTTPError as exc:
        # Only the status code is printed - never the key or URL.
        bad(f"{provider}: HTTP {exc.code} (401/403 usually means an invalid key)")
    except (urllib.error.URLError, TimeoutError) as exc:
        bad(f"{provider}: network problem ({type(exc).__name__})")
    return None


def check_models() -> bool:
    """Verify each configured model exists for the matching provider."""
    catalogs = {"groq": list_models("groq"), "gemini": list_models("gemini")}
    for provider, models in catalogs.items():
        if models is not None:
            ok(f"{provider} key works ({len(models)} models visible)")

    healthy = True
    for var in MODEL_VARS:
        value = os.getenv(var, "")
        if "/" not in value:
            bad(f"{var} must look like 'provider/model-id' (got '{value}')")
            healthy = False
            continue
        provider, model_id = value.split("/", 1)
        catalog = catalogs.get(provider)
        if catalog is None:
            warn(f"{var}={value}: cannot verify (provider unchecked)")
        elif model_id in catalog:
            ok(f"{var}={value}")
        else:
            bad(f"{var}={value} is NOT available. Pick one from the list below.")
            print("        Available:", ", ".join(sorted(catalog)[:25]))
            healthy = False
    return healthy


def main() -> int:
    print("=== Leo AI Tutor: setup check (no tokens used) ===")
    results = [check_python(), check_packages(), check_models()]
    if all(results):
        print(f"\n{GREEN}All checks passed. You are ready for Phase 3.{RESET}")
        return 0
    print(f"\n{RED}Some checks failed. Fix them and re-run.{RESET}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
