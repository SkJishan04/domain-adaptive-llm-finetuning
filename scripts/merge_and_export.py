#!/usr/bin/env python
"""Merge a trained LoRA adapter into the base model weights and export a
standalone model directory (useful for deployment without a PEFT dependency
at inference time)."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import get_settings
from src.utils.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into the base model.")
    parser.add_argument(
        "--adapter-dir", type=Path, required=True, help="Path to a trained LoRA adapter."
    )
    parser.add_argument(
        "--export-dir", type=Path, required=True, help="Directory to write the merged model to."
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    from src.training.lora_utils import load_model_with_adapter

    logger.info("Loading base model + adapter from %s", args.adapter_dir)
    model, tokenizer = load_model_with_adapter(
        settings.base_model_name, str(args.adapter_dir), settings.max_seq_length
    )

    logger.info("Merging LoRA weights into base model")
    merged_model = model.merge_and_unload()

    args.export_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(args.export_dir))
    tokenizer.save_pretrained(str(args.export_dir))
    logger.info("Merged model exported to %s", args.export_dir)


if __name__ == "__main__":
    main()