"""Evaluator: runs a loaded model against all configured benchmarks and
produces an EvaluationReport, including retention-rate comparison against a
stored base-model baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.benchmarks import (
    GenerativeExample,
    MultipleChoiceExample,
    load_domain_benchmark,
    load_gsm8k,
    load_mmlu,
)
from src.evaluation.metrics import (
    BenchmarkResult,
    EvaluationReport,
    accuracy,
    exact_match_numeric,
    extract_final_number,
    retention_rate,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class EvalHarnessConfig:
    domain_eval_path: Path
    domain_max_samples: int | None
    mmlu_config: str
    mmlu_split: str
    mmlu_max_samples: int | None
    gsm8k_config: str
    gsm8k_split: str
    gsm8k_max_samples: int | None
    gsm8k_max_new_tokens: int
    results_dir: Path


class Evaluator:
    """Wraps a `model` + `tokenizer` pair (see `src.inference.model_loader`)
    and scores it on the domain benchmark, MMLU, and GSM8K."""

    def __init__(self, model: Any, tokenizer: Any, config: EvalHarnessConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def evaluate(self, model_label: str) -> EvaluationReport:
        report = EvaluationReport(model_label=model_label)

        domain_examples = load_domain_benchmark(
            self.config.domain_eval_path, self.config.domain_max_samples
        )
        report.results.append(self._score_multiple_choice("domain", domain_examples))

        mmlu_examples = load_mmlu(
            self.config.mmlu_config, self.config.mmlu_split, self.config.mmlu_max_samples
        )
        report.results.append(self._score_multiple_choice("mmlu", mmlu_examples))

        gsm8k_examples = load_gsm8k(
            self.config.gsm8k_config, self.config.gsm8k_split, self.config.gsm8k_max_samples
        )
        report.results.append(self._score_gsm8k("gsm8k", gsm8k_examples))

        return report

    def _score_multiple_choice(
        self, name: str, examples: list[MultipleChoiceExample]
    ) -> BenchmarkResult:
        """Score via log-likelihood comparison across choices (standard MMLU-style
        evaluation), rather than free-form generation + parsing."""
        num_correct = 0
        for ex in examples:
            predicted_index = self._best_choice_by_loglikelihood(ex.question, ex.choices)
            if predicted_index == ex.answer_index:
                num_correct += 1

        acc = accuracy(num_correct, len(examples))
        logger.info("%s: accuracy=%.4f (%d/%d)", name, acc, num_correct, len(examples))
        return BenchmarkResult(
            name=name, accuracy=acc, num_examples=len(examples), num_correct=num_correct
        )

    def _score_gsm8k(self, name: str, examples: list[GenerativeExample]) -> BenchmarkResult:
        num_correct = 0
        for ex in examples:
            generated = self._generate(ex.question, self.config.gsm8k_max_new_tokens)
            prediction = extract_final_number(generated)
            gold = extract_final_number(ex.gold_answer)
            if exact_match_numeric(prediction, gold):
                num_correct += 1

        acc = accuracy(num_correct, len(examples))
        logger.info("%s: accuracy=%.4f (%d/%d)", name, acc, num_correct, len(examples))
        return BenchmarkResult(
            name=name, accuracy=acc, num_examples=len(examples), num_correct=num_correct
        )

    def _best_choice_by_loglikelihood(self, question: str, choices: list[str]) -> int:
        import torch

        best_index, best_score = 0, float("-inf")
        for i, choice in enumerate(choices):
            prompt = f"Question: {question}\nAnswer: {choice}"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
            # Lower loss == higher likelihood of this completion.
            score = -outputs.loss.item()
            if score > best_score:
                best_score, best_index = score, i
        return best_index

    def _generate(self, question: str, max_new_tokens: int) -> str:
        inputs = self.tokenizer(f"Question: {question}\nAnswer:", return_tensors="pt").to(
            self.model.device
        )
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def save_report(report: EvaluationReport, results_dir: Path) -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{report.model_label}.json"
    out_path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    return out_path


def compute_retention(finetuned: EvaluationReport, baseline: EvaluationReport) -> dict[str, float]:
    """Retention rate per general-capability benchmark (MMLU here; GSM8K is
    reported the same way if desired)."""
    retained: dict[str, float] = {}
    for name in ("mmlu", "gsm8k"):
        ft_score = finetuned.accuracy_for(name)
        base_score = baseline.accuracy_for(name)
        if ft_score is not None and base_score is not None:
            retained[name] = retention_rate(ft_score, base_score)
    return retained