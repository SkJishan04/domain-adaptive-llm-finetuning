"""Stage 1: Continued Pre-Training (CPT) on raw domain text via causal LM loss."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.domain_corpus import build_cpt_dataset
from src.training.lora_utils import LoraSettings, attach_lora, load_base_model
from src.utils.logging_config import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


@dataclass
class CPTConfig:
    base_model_name: str
    max_seq_length: int
    load_in_4bit: bool
    lora: LoraSettings
    corpus_dir: Path
    block_size: int
    val_split: float
    output_dir: Path
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    weight_decay: float
    logging_steps: int
    seed: int
    optim: str
    report_to: str


class CPTTrainer:
    """Trains LoRA adapters with a standard causal-LM objective over raw,
    unformatted domain text (no instruction masking — every token contributes
    to the loss)."""

    def __init__(self, config: CPTConfig) -> None:
        self.config = config

    def run(self) -> Path:
        set_seed(self.config.seed)
        logger.info("Loading base model %s for CPT", self.config.base_model_name)

        model, tokenizer = load_base_model(
            self.config.base_model_name, self.config.max_seq_length, self.config.load_in_4bit
        )
        model = attach_lora(model, self.config.lora)

        train_ds, val_ds, stats = build_cpt_dataset(
            self.config.corpus_dir, tokenizer, self.config.block_size, self.config.val_split
        )
        logger.info(
            "CPT corpus stats: documents=%d tokens=%d blocks=%d",
            stats.num_documents,
            stats.num_tokens,
            stats.num_blocks,
        )

        trainer = self._build_trainer(model, tokenizer, train_ds, val_ds)
        trainer.train()

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        logger.info("CPT adapter saved to %s", output_dir)
        return output_dir

    def _build_trainer(
        self, model: Any, tokenizer: Any, train_ds: Any, val_ds: Any
    ) -> Any:
        from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
        from unsloth import is_bf16_supported

        args = TrainingArguments(
            output_dir=str(self.config.output_dir),
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_strategy="epoch",
            eval_strategy="epoch" if val_ds is not None else "no",
            seed=self.config.seed,
            optim=self.config.optim,
            report_to=self.config.report_to,
            bf16=is_bf16_supported(),
            fp16=not is_bf16_supported(),
        )
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        return Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
        )