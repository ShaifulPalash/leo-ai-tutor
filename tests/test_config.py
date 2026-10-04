"""Config tests: validation, fallback chains, secret safety."""

import json

import pytest
from pydantic import ValidationError

from leo.config import Settings


def test_bad_values_are_rejected_early():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, score_threshold=5)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, fast_model_primary="no-provider-prefix")


def test_log_level_is_normalised_and_debug_mode_wins():
    assert Settings(_env_file=None, log_level="debug").log_level == "DEBUG"
    assert (
        Settings(_env_file=None, log_level="INFO", debug_mode=True).effective_log_level == "DEBUG"
    )


def test_chain_skips_providers_without_keys():
    only_groq = Settings(_env_file=None, groq_api_key="gsk_x", gemini_api_key="")
    assert only_groq.model_chain("fast") == [only_groq.fast_model_primary]
    placeholder = Settings(
        _env_file=None, groq_api_key="gsk_x", gemini_api_key="your_gemini_key_here"
    )
    assert len(placeholder.model_chain("smart")) == 1


def test_full_chain_and_optional_ollama():
    both = Settings(_env_file=None, groq_api_key="gsk_x", gemini_api_key="AIza_x")
    assert len(both.model_chain("fast")) == 2
    local = Settings(_env_file=None, groq_api_key="", gemini_api_key="", ollama_enabled=True)
    assert local.model_chain("fast") == [local.ollama_model]


def test_secrets_never_appear_in_repr_or_summary():
    settings = Settings(_env_file=None, groq_api_key="gsk_super_secret_value")
    assert "gsk_super_secret_value" not in repr(settings)
    assert "gsk_super_secret_value" not in json.dumps(settings.safe_summary())


def test_relative_paths_become_absolute():
    assert Settings(_env_file=None, log_dir="somewhere").log_dir.is_absolute()
