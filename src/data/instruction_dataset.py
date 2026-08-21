"""Instruction dataset loading and prompt formatting for Stage 2 (SFT).

Instructions are stored as JSONL records with `instruction`, optional
`input`, and `output` fields, and are rendered into a single Alpaca-style
prompt template. The response template is used downstream by
`DataCollatorForCompletionOnlyLM` (TRL) so that loss is only computed on the
model's response tokens, not the prompt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROMPT_TEMPLATE_WITH_INPUT = (
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
)
PROMPT_TEMPLATE_NO_INPUT = "### Instruction:\n{instruction}\n\n### Response:\n{output}"
RESPONSE_TEMPLATE = "### Response:\n"


@dataclass(frozen=True)
class InstructionExample:
    instruction: str
    input: str
    output: str

    def format(self) -> str:
        if self.input.strip():
            return PROMPT_TEMPLATE_WITH_INPUT.format(
                instruction=self.instruction, input=self.input, output=self.output
            )
        return PROMPT_TEMPLATE_NO_INPUT.format(instruction=self.instruction, output=self.output)


def load_instruction_examples(path: Path) -> list[InstructionExample]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Instruction dataset not found: {path}")

    examples: list[InstructionExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc

            if "instruction" not in record or "output" not in record:
                raise ValueError(
                    f"Line {line_no} of {path} is missing required fields "
                    "'instruction' and/or 'output'"
                )
            examples.append(
                InstructionExample(
                    instruction=record["instruction"],
                    input=record.get("input", ""),
                    output=record["output"],
                )
            )

    if not examples:
        raise ValueError(f"No instruction examples found in {path}")
    return examples


def build_sft_dataset(path: Path, val_split: float = 0.1) -> tuple[Any, Any]:
    """Return (train_dataset, val_dataset) as HF `datasets.Dataset` with a
    single `text` column already rendered via the prompt template."""
    from datasets import Dataset

    examples = load_instruction_examples(path)
    texts = [ex.format() for ex in examples]

    n_val = max(1, int(len(texts) * val_split)) if len(texts) > 1 else 0
    val_texts = texts[:n_val]
    train_texts = texts[n_val:]

    train_ds = Dataset.from_dict({"text": train_texts})
    val_ds = Dataset.from_dict({"text": val_texts}) if val_texts else None
    return train_ds, val_ds