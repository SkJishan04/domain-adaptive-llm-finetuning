"""API route handlers: generation and evaluation-summary endpoints."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import provide_model_loader, provide_settings
from src.api.schemas import (
    EvaluationSummaryResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
)
from src.config import Settings
from src.inference.generate import generate_response
from src.inference.model_loader import ModelLoader

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(provide_settings)) -> HealthResponse:
    return HealthResponse(status="ok", model_label=settings.domain_name)


@router.post("/generate", response_model=GenerateResponse)
def generate(
    request: GenerateRequest,
    settings: Settings = Depends(provide_settings),
    model_loader: ModelLoader = Depends(provide_model_loader),
) -> GenerateResponse:
    try:
        model, tokenizer = model_loader.load(
            base_model_name=settings.base_model_name,
            max_seq_length=settings.max_seq_length,
            adapter_dir=str(settings.sft_adapter_dir),
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as a clean 503 to the client
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc

    response_text = generate_response(
        model,
        tokenizer,
        request.instruction,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
    )
    return GenerateResponse(
        instruction=request.instruction, response=response_text, model_label=settings.domain_name
    )


@router.get("/evaluation/summary", response_model=EvaluationSummaryResponse)
def evaluation_summary(
    settings: Settings = Depends(provide_settings),
) -> EvaluationSummaryResponse:
    results_path = Path(settings.output_dir) / "evaluation" / f"{settings.domain_name}_sft.json"
    if not results_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No evaluation report found at {results_path}. Run scripts/run_evaluation.py first.",
        )

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    return EvaluationSummaryResponse(
        model_label=payload["model_label"],
        results=payload["results"],
        retention=payload.get("retention", {}),
    )