"""Pydantic request/response schemas for the serving API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    instruction: str = Field(..., min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    instruction: str
    response: str
    model_label: str


class BenchmarkScore(BaseModel):
    name: str
    accuracy: float
    num_examples: int
    num_correct: int


class EvaluationSummaryResponse(BaseModel):
    model_label: str
    results: list[BenchmarkScore]
    retention: dict[str, float] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    model_label: str | None = None