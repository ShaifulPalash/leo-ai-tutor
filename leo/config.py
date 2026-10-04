"""Central, typed configuration for Leo.

All values come from environment variables or the .env file. Nothing secret
is ever hard-coded. Run `python -m leo.config` to print a safe summary.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = the folder that contains the "leo" package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# "fast" = small/cheap model; "smart" = higher-quality model.
Tier = Literal["fast", "smart"]

_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class Settings(BaseSettings):
    """Every configurable value in Leo, with safe defaults and validation."""

    # Read PROJECT_ROOT/.env regardless of the folder you run Python from.
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unknown variables instead of crashing
        case_sensitive=False,  # GROQ_API_KEY == groq_api_key
    )

    # ---- API keys (SecretStr hides the value when printed) ----
    groq_api_key: SecretStr = SecretStr("")
    gemini_api_key: SecretStr = SecretStr("")

    # ---- Model names in "provider/model-id" format ----
    fast_model_primary: str = "groq/openai/gpt-oss-20b"
    fast_model_fallback: str = "gemini/gemini-3.5-flash-lite"
    smart_model_primary: str = "groq/openai/gpt-oss-120b"
    smart_model_fallback: str = "gemini/gemini-3.8-flash"

    # ---- Optional local fallback via Ollama ----
    ollama_enabled: bool = False
    ollama_model: str = "ollama/llama3.2"
    ollama_base_url: str = "http://localhost:11434"

    # ---- Token and rate controls ----
    max_tokens_router: int = Field(400, ge=50, le=8000)
    max_tokens_explainer: int = Field(900, ge=100, le=8000)
    max_tokens_quiz: int = Field(1200, ge=100, le=8000)
    max_tokens_evaluator: int = Field(700, ge=100, le=8000)
    max_rpm: int = Field(20, ge=1, le=60)
    llm_timeout_seconds: int = Field(45, ge=5, le=300)
    max_retries: int = Field(3, ge=0, le=6)
    temperature: float = Field(0.3, ge=0.0, le=1.5)

    # ---- Learning-loop behaviour ----
    score_threshold: float = Field(0.6, ge=0.0, le=1.0)
    max_relearn_rounds: int = Field(2, ge=0, le=5)

    # ---- Optional tools ----
    enable_calculator: bool = True
    enable_web_search: bool = False

    # ---- Logging and debugging ----
    log_level: str = "INFO"
    debug_mode: bool = False
    log_dir: Path = Path("logs")
    trace_dir: Path = Path("logs/traces")

    # ---- Storage ----
    memory_db_path: Path = Path("data/leo_memory.db")

    # ------------------------------------------------------------------
    # Validators: reject bad values early with a clear error message.
    # ------------------------------------------------------------------
    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, value: str) -> str:
        value = value.upper().strip()
        if value not in _VALID_LOG_LEVELS:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(_VALID_LOG_LEVELS)}")
        return value

    @field_validator(
        "fast_model_primary",
        "fast_model_fallback",
        "smart_model_primary",
        "smart_model_fallback",
        "ollama_model",
    )
    @classmethod
    def _check_model_format(cls, value: str) -> str:
        if "/" not in value:
            raise ValueError("Model names must look like 'provider/model-id'")
        return value.strip()

    @field_validator("log_dir", "trace_dir", "memory_db_path")
    @classmethod
    def _make_absolute(cls, value: Path) -> Path:
        # Relative paths in .env are resolved against the project root.
        return value if value.is_absolute() else PROJECT_ROOT / value

    # ------------------------------------------------------------------
    # Helper properties and methods used by the rest of the code.
    # ------------------------------------------------------------------
    @property
    def effective_log_level(self) -> str:
        """DEBUG_MODE=true forces DEBUG regardless of LOG_LEVEL."""
        return "DEBUG" if self.debug_mode else self.log_level

    @property
    def max_tokens_by_role(self) -> dict[str, int]:
        """Token ceiling for each agent role."""
        return {
            "coordinator": self.max_tokens_router,
            "explainer": self.max_tokens_explainer,
            "quiz_master": self.max_tokens_quiz,
            "evaluator": self.max_tokens_evaluator,
        }

    @staticmethod
    def provider_of(model: str) -> str:
        """'groq/openai/gpt-oss-20b' -> 'groq'."""
        return model.split("/", 1)[0].lower()

    def has_key(self, provider: str) -> bool:
        """True if a real (non-placeholder) key exists for the provider."""
        secrets = {"groq": self.groq_api_key, "gemini": self.gemini_api_key}
        if provider not in secrets:
            return False
        raw = secrets[provider].get_secret_value().strip()
        return bool(raw) and not raw.startswith("your_")

    def api_key_for(self, model: str) -> str | None:
        """Return the API key for a model's provider (None for local models)."""
        provider = self.provider_of(model)
        if provider == "groq" and self.has_key("groq"):
            return self.groq_api_key.get_secret_value().strip()
        if provider == "gemini" and self.has_key("gemini"):
            return self.gemini_api_key.get_secret_value().strip()
        return None

    def model_chain(self, tier: Tier) -> list[str]:
        """Ordered list of models to try: primary -> fallback -> optional Ollama.

        Models whose provider has no API key are skipped, so a missing Gemini
        key just shortens the chain instead of causing errors.
        """
        candidates = (
            [self.fast_model_primary, self.fast_model_fallback]
            if tier == "fast"
            else [self.smart_model_primary, self.smart_model_fallback]
        )
        if self.ollama_enabled:
            candidates.append(self.ollama_model)

        chain: list[str] = []
        for model in candidates:
            provider = self.provider_of(model)
            usable = provider == "ollama" or self.has_key(provider)
            if usable and model not in chain:
                chain.append(model)
        return chain

    def ensure_dirs(self) -> None:
        """Create the folders Leo writes to (logs, traces, database)."""
        for folder in (self.log_dir, self.trace_dir, self.memory_db_path.parent):
            folder.mkdir(parents=True, exist_ok=True)

    def safe_summary(self) -> dict:
        """A printable snapshot of the config that never contains secrets."""
        return {
            "groq_key": "set" if self.has_key("groq") else "MISSING",
            "gemini_key": "set" if self.has_key("gemini") else "MISSING",
            "fast_chain": self.model_chain("fast"),
            "smart_chain": self.model_chain("smart"),
            "max_tokens": self.max_tokens_by_role,
            "max_rpm": self.max_rpm,
            "timeout_s": self.llm_timeout_seconds,
            "max_retries": self.max_retries,
            "score_threshold": self.score_threshold,
            "max_relearn_rounds": self.max_relearn_rounds,
            "tools": {"calculator": self.enable_calculator, "web_search": self.enable_web_search},
            "log_level": self.effective_log_level,
            "debug_mode": self.debug_mode,
            "memory_db": str(self.memory_db_path),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the one shared Settings object (created on first use)."""
    settings = Settings()
    settings.ensure_dirs()
    return settings


def reload_settings() -> Settings:
    """Clear the cache and re-read .env (used by tests)."""
    get_settings.cache_clear()
    return get_settings()


if __name__ == "__main__":
    print(json.dumps(get_settings().safe_summary(), indent=2))
