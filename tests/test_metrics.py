"""Tests for evaluation metrics, including the retention-rate calculation
that quantifies catastrophic forgetting."""

from __future__ import annotations

import pytest

from src.evaluation.metrics import (
    BenchmarkResult,
    EvaluationReport,
    accuracy,
    exact_match_numeric,
    extract_final_number,
    retention_rate,
)


def test_accuracy_basic() -> None:
    assert accuracy(3, 4) == pytest.approx(0.75)


def test_accuracy_rejects_zero_examples() -> None:
    with pytest.raises(ValueError):
        accuracy(0, 0)


def test_extract_final_number_prefers_gsm8k_marker() -> None:
    text = "Step 1: 2+2=4. #### 4"
    assert extract_final_number(text) == "4"


def test_extract_final_number_falls_back_to_last_number() -> None:
    text = "First we compute 12, then adjust to get 42 total."
    assert extract_final_number(text) == "42"


def test_extract_final_number_returns_none_when_absent() -> None:
    assert extract_final_number("no digits here") is None


def test_exact_match_numeric_handles_formatting_differences() -> None:
    assert exact_match_numeric("42", "42.0")
    assert not exact_match_numeric("42", "43")
    assert not exact_match_numeric(None, "42")


def test_retention_rate_matches_spec_example() -> None:
    # Base MMLU 66.5%, CPT+SFT pipeline retains 65.1% -> ~97.9% retention.
    rate = retention_rate(finetuned_score=0.651, base_score=0.665)
    assert rate == pytest.approx(97.89, abs=0.05)


def test_retention_rate_rejects_zero_base_score() -> None:
    with pytest.raises(ValueError):
        retention_rate(0.5, 0.0)


def test_evaluation_report_accuracy_for_lookup() -> None:
    report = EvaluationReport(model_label="legal_sft")
    report.results.append(BenchmarkResult("domain", 0.674, 100, 67))
    report.results.append(BenchmarkResult("mmlu", 0.651, 500, 325))

    assert report.accuracy_for("mmlu") == pytest.approx(0.651)
    assert report.accuracy_for("missing") is None
    assert report.as_dict()["results"][0]["name"] == "domain"