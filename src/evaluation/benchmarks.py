"""Benchmark loaders: domain-specific QA plus general-capability MMLU/GSM8K.

MMLU and GSM8K are loaded from the Hugging Face Hub; the domain benchmark is
a local, versioned JSONL file so the "domain-specific gain" metric is fully
reproducible and doesn't depend on an external dataset's future changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class MultipleChoiceExample:
    question: str
    choices: list[str]
    answer_index: int


@dataclass(frozen=True)
class GenerativeExample:
    question: str
    gold_answer: str


BenchmarkType = Literal["multiple_choice", "generative_exact_match"]


def load_domain_benchmark(path: Path, max_samples: int | None = None) -> list[MultipleChoiceExample]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Domain evaluation set not found: {path}")

    examples: list[MultipleChoiceExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record: dict[str, Any] = json.loads(line)
            examples.append(
                MultipleChoiceExample(
                    question=record["question"],
                    choices=record["choices"],
                    answer_index=record["answer_index"],
                )
            )
    if max_samples is not None:
        examples = examples[:max_samples]
    if not examples:
        raise ValueError(f"Domain benchmark at {path} is empty")
    return examples


def load_mmlu(
    hf_config: str = "all", split: str = "test", max_samples: int | None = 500
) -> list[MultipleChoiceExample]:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", hf_config, split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    return [
        MultipleChoiceExample(
            question=row["question"], choices=row["choices"], answer_index=row["answer"]
        )
        for row in ds
    ]


def load_gsm8k(
    hf_config: str = "main", split: str = "test", max_samples: int | None = 250
) -> list[GenerativeExample]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", hf_config, split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    return [
        GenerativeExample(question=row["question"], gold_answer=row["answer"]) for row in ds
    ]