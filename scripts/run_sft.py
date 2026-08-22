#!/usr/bin/env python
"""CLI entrypoint for Stage 2: Supervised instruction fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.config import get_settings
from src.training.sft_trainer import SFTConfig, SFTTrainer
from src.utils.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def load_yaml_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_config(raw: dict, settings) -> SFTConfig:
    return SFTConfig(
        base_model_name=raw["model"]["base_model_name"] or settings.base_model_name,
        max_seq_length=raw["model"]["max_seq_length"],
        load_in_4bit=raw["model"]["load_in_4bit"],
        cpt_adapter_dir=Path(settings.cpt_adapter_dir),
        instructions_path=Path(settings.domain_instructions_path),
        val_split=raw["data"]["val_split"],
        output_dir=Path(settings.sft_adapter_dir),
        num_train_epochs=raw["training"]["num_train_epochs"],
        per_device_train_batch_size=raw["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=raw["training"]["gradient_accumulation_steps"],
        learning_rate=raw["training"]["learning_rate"],
        lr_scheduler_type=raw["training"]["lr_scheduler_type"],
        warmup_ratio=raw["training"]["warmup_ratio"],
        weight_decay=raw["training"]["weight_decay"],
        logging_steps=raw["training"]["logging_steps"],
        seed=raw["training"]["seed"],
        optim=raw["training"]["optim"],
        report_to=raw["training"]["report_to"],
        packing=raw["training"]["packing"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2 instruction fine-tuning.")
    parser.add_argument("--config", type=Path, default=Path("configs/sft_config.yaml"))
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    raw_config = load_yaml_config(args.config)
    config = build_config(raw_config, settings)

    logger.info("Starting SFT run with config: %s", args.config)
    SFTTrainer(config).run()


if __name__ == "__main__":
    main()