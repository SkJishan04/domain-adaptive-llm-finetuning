"""FastAPI dependency providers, overridable in tests."""

from __future__ import annotations

from src.config import Settings, get_settings
from src.inference.model_loader import ModelLoader, get_model_loader


def provide_settings() -> Settings:
    return get_settings()


def provide_model_loader() -> ModelLoader:
    return get_model_loader()