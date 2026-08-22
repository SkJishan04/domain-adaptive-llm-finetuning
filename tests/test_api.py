"""API tests using dependency overrides so no real model is loaded."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from src.api.dependencies import provide_model_loader, provide_settings
from src.api.main import app
from src.config import Settings


class _FakeModel:
    device = "cpu"

    def generate(self, **kwargs):
        return [[0, 1, 2]]


class _FakeTokenizer:
    def __call__(self, text, return_tensors=None):
        class _Inputs(dict):
            def to(self, device):
                return self

        return _Inputs(input_ids=[[0, 1, 2]])

    def decode(self, token_ids, skip_special_tokens=True):
        return "### Response:\nThe indemnification clause allocates liability risk."


class _FakeModelLoader:
    def load(self, base_model_name, max_seq_length, adapter_dir=None, load_in_4bit=True):
        return _FakeModel(), _FakeTokenizer()


def _override_settings() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


def _override_model_loader() -> _FakeModelLoader:
    return _FakeModelLoader()


app.dependency_overrides[provide_settings] = _override_settings
app.dependency_overrides[provide_model_loader] = _override_model_loader

client = TestClient(app)


def test_health_endpoint_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_label"] == "legal"


def test_generate_endpoint_returns_response_text() -> None:
    response = client.post("/generate", json={"instruction": "What is an indemnification clause?"})

    assert response.status_code == 200
    body = response.json()
    assert body["instruction"] == "What is an indemnification clause?"
    assert "indemnification" in body["response"]


def test_generate_endpoint_rejects_empty_instruction() -> None:
    response = client.post("/generate", json={"instruction": ""})

    assert response.status_code == 422


def test_evaluation_summary_returns_404_when_no_report_exists(tmp_path, monkeypatch) -> None:
    def _settings_with_missing_output() -> Settings:
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        settings.output_dir = tmp_path
        return settings

    app.dependency_overrides[provide_settings] = _settings_with_missing_output
    try:
        response = client.get("/evaluation/summary")
        assert response.status_code == 404
    finally:
        app.dependency_overrides[provide_settings] = _override_settings


def test_evaluation_summary_returns_report_when_present(tmp_path) -> None:
    def _settings_with_report() -> Settings:
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        settings.output_dir = tmp_path
        settings.domain_name = "legal"
        return settings

    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir(parents=True)
    payload = {
        "model_label": "legal_sft",
        "results": [{"name": "domain", "accuracy": 0.674, "num_examples": 100, "num_correct": 67}],
        "retention": {"mmlu": 97.9},
    }
    (eval_dir / "legal_sft.json").write_text(json.dumps(payload), encoding="utf-8")

    app.dependency_overrides[provide_settings] = _settings_with_report
    try:
        response = client.get("/evaluation/summary")
        assert response.status_code == 200
        body = response.json()
        assert body["model_label"] == "legal_sft"
        assert body["retention"]["mmlu"] == 97.9
    finally:
        app.dependency_overrides[provide_settings] = _override_settings