"""Tests for environment-driven configuration."""

from __future__ import annotations

from src.config import Settings


def test_settings_defaults_load_without_env_file(monkeypatch) -> None:
    monkeypatch.delenv("BASE_MODEL_NAME", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.base_model_name == "unsloth/llama-3-8b-bnb-4bit"
    assert settings.max_seq_length == 2048
    assert settings.domain_name == "legal"
    assert settings.hf_token is None


def test_settings_reads_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("BASE_MODEL_NAME", "unsloth/mistral-7b-bnb-4bit")
    monkeypatch.setenv("DOMAIN_NAME", "medical")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.base_model_name == "unsloth/mistral-7b-bnb-4bit"
    assert settings.domain_name == "medical"