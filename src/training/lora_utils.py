"""LoRA adapter construction shared by both the CPT and SFT stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LoraSettings:
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 42


def load_base_model(
    model_name: str, max_seq_length: int, load_in_4bit: bool = True
) -> tuple[Any, Any]:
    """Load a 4-bit quantized base model + tokenizer via Unsloth's FastLanguageModel."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer


def attach_lora(model: Any, settings: LoraSettings) -> Any:
    """Wrap a base model with trainable LoRA adapters on the configured target
    modules (attention projections + MLP projections, per the diagram in the
    project spec)."""
    from unsloth import FastLanguageModel

    return FastLanguageModel.get_peft_model(
        model,
        r=settings.r,
        target_modules=settings.target_modules,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        bias=settings.bias,
        use_gradient_checkpointing=settings.use_gradient_checkpointing,
        random_state=settings.random_state,
    )


def load_model_with_adapter(
    model_name: str, adapter_dir: str, max_seq_length: int, load_in_4bit: bool = True
) -> tuple[Any, Any]:
    """Load the base model and attach a previously trained LoRA adapter
    (used to continue training in Stage 2, and for inference/evaluation)."""
    from peft import PeftModel

    model, tokenizer = load_base_model(model_name, max_seq_length, load_in_4bit)
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
    return model, tokenizer