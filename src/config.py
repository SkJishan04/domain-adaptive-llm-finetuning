"""Centralized, environment-driven configuration.

All runtime configuration is sourced from environment variables (with
sensible defaults), loaded once via a cached pydantic-settings object.
Secrets (API keys/tokens) are never hardcoded and are always optional at
import time so that non-training code paths (e.g. the API, tests) can run
without them being set.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Secrets
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    wandb_api_key: str | None = Field(default=None, alias="WANDB_API_KEY")

    # Model / run
    base_model_name: str = Field(default="unsloth/llama-3-8b-bnb-4bit", alias="BASE_MODEL_NAME")
    max_seq_length: int = Field(default=2048, alias="MAX_SEQ_LENGTH")
    domain_name: str = Field(default="legal", alias="DOMAIN_NAME")

    # Paths
    output_dir: Path = Field(default=Path("./outputs"), alias="OUTPUT_DIR")
    cpt_adapter_dir: Path = Field(default=Path("./outputs/cpt"), alias="CPT_ADAPTER_DIR")
    sft_adapter_dir: Path = Field(default=Path("./outputs/sft"), alias="SFT_ADAPTER_DIR")
    domain_corpus_dir: Path = Field(default=Path("./data/domain_corpus"), alias="DOMAIN_CORPUS_DIR")
    domain_instructions_path: Path = Field(
        default=Path("./data/instructions/legal_instructions.jsonl"),
        alias="DOMAIN_INSTRUCTIONS_PATH",
    )
    domain_eval_path: Path = Field(
        default=Path("./data/domain_eval/legal_eval.jsonl"), alias="DOMAIN_EVAL_PATH"
    )

    # Experiment tracking
    wandb_project: str = Field(default="domain-adaptive-llm-finetuning", alias="WANDB_PROJECT")
    wandb_mode: str = Field(default="disabled", alias="WANDB_MODE")

    # API
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    return Settings()