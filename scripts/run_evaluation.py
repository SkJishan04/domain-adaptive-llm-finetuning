#!/usr/bin/env python
"""CLI entrypoint for the evaluation harness.

Evaluates one of {base, cpt, sft} model variants against the domain
benchmark, MMLU, and GSM8K, saves a JSON report, and (for cpt/sft) computes
retention rate against a stored base-model baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.config import get_settings
from src.evaluation.evaluator import EvalHarnessConfig, Evaluator, compute_retention, save_report
from src.evaluation.metrics import EvaluationReport
from src.inference.model_loader import get_model_loader
from src.utils.logging_config import configure_logging, get_logger

logger = get_logger(__name__)

VARIANT_ADAPTER_ATTR = {"base": None, "cpt": "cpt_adapter_dir", "sft": "sft_adapter_dir"}


def build_harness_config(raw: dict, settings) -> EvalHarnessConfig:
    return EvalHarnessConfig(
        domain_eval_path=Path(settings.domain_eval_path),
        domain_max_samples=raw["benchmarks"]["domain"]["max_samples"],
        mmlu_config=raw["benchmarks"]["mmlu"]["hf_config"],
        mmlu_split=raw["benchmarks"]["mmlu"]["split"],
        mmlu_max_samples=raw["benchmarks"]["mmlu"]["max_samples"],
        gsm8k_config=raw["benchmarks"]["gsm8k"]["hf_config"],
        gsm8k_split=raw["benchmarks"]["gsm8k"]["split"],
        gsm8k_max_samples=raw["benchmarks"]["gsm8k"]["max_samples"],
        gsm8k_max_new_tokens=raw["benchmarks"]["gsm8k"]["max_new_tokens"],
        results_dir=Path(raw["output"]["results_dir"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the evaluation harness on a model variant.")
    parser.add_argument("--variant", choices=["base", "cpt", "sft"], required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/eval_config.yaml"))
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    raw_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    harness_config = build_harness_config(raw_config, settings)

    adapter_attr = VARIANT_ADAPTER_ATTR[args.variant]
    adapter_dir = str(getattr(settings, adapter_attr)) if adapter_attr else None

    loader = get_model_loader()
    model, tokenizer = loader.load(
        base_model_name=settings.base_model_name,
        max_seq_length=settings.max_seq_length,
        adapter_dir=adapter_dir,
    )

    model_label = f"{settings.domain_name}_{args.variant}"
    evaluator = Evaluator(model, tokenizer, harness_config)
    report = evaluator.evaluate(model_label)

    out_path = save_report(report, harness_config.results_dir)
    logger.info("Saved evaluation report to %s", out_path)

    baseline_path = Path(raw_config["output"]["baseline_results_path"])
    if args.variant != "base" and baseline_path.exists():
        import json

        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_report = EvaluationReport(model_label=baseline_payload["model_label"])
        from src.evaluation.metrics import BenchmarkResult

        baseline_report.results = [
            BenchmarkResult(**r) for r in baseline_payload["results"]
        ]
        retention = compute_retention(report, baseline_report)
        logger.info("Retention vs base model: %s", retention)

        payload = report.as_dict()
        payload["retention"] = retention
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elif args.variant == "base":
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            Path(out_path).read_text(encoding="utf-8"), encoding="utf-8"
        )
        logger.info("Stored base-model results as retention baseline at %s", baseline_path)


if __name__ == "__main__":
    main()