"""Cached model loading for inference/evaluation, used by both the
evaluation harness and the serving API so a model is loaded at most once."""

from __future__ import annotations

from functools import lru_cache
from typing import Any


class ModelLoader:
    """Lazily loads and caches a (base + optional adapter) model/tokenizer pair.

    Kept as an instantiable class (rather than a bare module-level singleton)
    so the API layer can override it in tests via dependency injection.
    """

    def __init__(self) -> None:
        self._cache: dict[str, tuple[Any, Any]] = {}

    def load(
        self,
        base_model_name: str,
        max_seq_length: int,
        adapter_dir: str | None = None,
        load_in_4bit: bool = True,
    ) -> tuple[Any, Any]:
        cache_key = f"{base_model_name}::{adapter_dir}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if adapter_dir:
            from src.training.lora_utils import load_model_with_adapter

            model, tokenizer = load_model_with_adapter(
                base_model_name, adapter_dir, max_seq_length, load_in_4bit
            )
        else:
            from src.training.lora_utils import load_base_model

            model, tokenizer = load_base_model(base_model_name, max_seq_length, load_in_4bit)

        model.eval()
        self._cache[cache_key] = (model, tokenizer)
        return model, tokenizer


@lru_cache(maxsize=1)
def get_model_loader() -> ModelLoader:
    return ModelLoader()