"""Metric computation and result aggregation for the evaluation harness."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

GSM8K_ANSWER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


@dataclass
class BenchmarkResult:
    name: str
    accuracy: float
    num_examples: int
    num_correct: int


@dataclass
class EvaluationReport:
    model_label: str
    results: list[BenchmarkResult] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "model_label": self.model_label,
            "results": [
                {
                    "name": r.name,
                    "accuracy": round(r.accuracy, 4),
                    "num_examples": r.num_examples,
                    "num_correct": r.num_correct,
                }
                for r in self.results
            ],
        }

    def accuracy_for(self, name: str) -> float | None:
        for r in self.results:
            if r.name == name:
                return r.accuracy
        return None


def accuracy(num_correct: int, num_examples: int) -> float:
    if num_examples == 0:
        raise ValueError("num_examples must be > 0 to compute accuracy")
    return num_correct / num_examples


def extract_final_number(text: str) -> str | None:
    """Extract the final numeric answer from a generated GSM8K-style response.

    Prefers the canonical '#### <answer>' marker used by GSM8K gold answers;
    falls back to the last number mentioned in free-form generations.
    """
    if "####" in text:
        tail = text.split("####")[-1]
        match = GSM8K_ANSWER_PATTERN.search(tail)
        if match:
            return match.group(0).replace(",", "")

    matches = GSM8K_ANSWER_PATTERN.findall(text)
    if matches:
        return matches[-1].replace(",", "")
    return None


def exact_match_numeric(prediction: str | None, gold: str | None) -> bool:
    if prediction is None or gold is None:
        return False
    try:
        return float(prediction) == float(gold)
    except ValueError:
        return prediction.strip() == gold.strip()


def retention_rate(finetuned_score: float, base_score: float) -> float:
    """Percentage of the base model's general-capability score retained after
    fine-tuning. 100% = no forgetting; <100% = catastrophic forgetting; >100%
    indicates the fine-tuning stage also improved general capability."""
    if base_score == 0:
        raise ValueError("base_score must be non-zero to compute retention rate")
    return (finetuned_score / base_score) * 100.0