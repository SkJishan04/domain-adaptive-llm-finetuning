"""Stage 2: Supervised instruction fine-tuning on top of the CPT adapter.

Continues training from the Stage 1 (CPT) LoRA weights, using
response-only loss masking so the model is optimized to produce the
target completion given an instruction, rather than to model the prompt
tokens themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.instruction_dataset import RESPONSE_TEMPLATE, build_sft_dataset
from src.training.lora_utils import load_model_with_adapter
from src.utils.logging_config import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


@dataclass
class SFTConfig:
    base_model_name: str
    max_seq_length: int
    load_in_4bit: bool
    cpt_adapter_dir: Path
    instructions_path: Path
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
    packing: bool


class SFTTrainer:
    """Continues training the CPT adapter on instruction-response pairs,
    freezing nothing but relying on the (already low-rank) adapter update
    to specialize behavior without catastrophically overwriting CPT knowledge."""

    def __init__(self, config: SFTConfig) -> None:
        self.config = config

    def run(self) -> Path:
        set_seed(self.config.seed)
        logger.info(
            "Loading base model %s with CPT adapter from %s",
            self.config.base_model_name,
            self.config.cpt_adapter_dir,
        )
        model, tokenizer = load_model_with_adapter(
            self.config.base_model_name,
            str(self.config.cpt_adapter_dir),
            self.config.max_seq_length,
            self.config.load_in_4bit,
        )

        train_ds, val_ds = build_sft_dataset(self.config.instructions_path, self.config.val_split)
        logger.info(
            "SFT dataset: train=%d val=%s",
            len(train_ds),
            len(val_ds) if val_ds is not None else 0,
        )

        trainer = self._build_trainer(model, tokenizer, train_ds, val_ds)
        trainer.train()

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        logger.info("SFT adapter saved to %s", output_dir)
        return output_dir

    def _build_trainer(
        self, model: Any, tokenizer: Any, train_ds: Any, val_ds: Any
    ) -> Any:
        from trl import DataCollatorForCompletionOnlyLM, SFTConfig as TRLSFTConfig
        from trl import SFTTrainer as TRLSFTTrainer

        collator = DataCollatorForCompletionOnlyLM(
            response_template=RESPONSE_TEMPLATE, tokenizer=tokenizer
        )

        args = TRLSFTConfig(
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
            packing=self.config.packing,
            max_seq_length=self.config.max_seq_length,
            dataset_text_field="text",
        )

        return TRLSFTTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            tokenizer=tokenizer,
        )