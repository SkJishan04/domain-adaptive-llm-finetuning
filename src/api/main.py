"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.config import get_settings
from src.utils.logging_config import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)

app = FastAPI(
    title="Domain-Adaptive LLM Fine-Tuning API",
    description=(
        "Serves a two-stage (CPT + SFT) domain-adapted LLM and exposes "
        "its catastrophic-forgetting evaluation results."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def on_startup() -> None:
    logger.info(
        "API starting: domain=%s base_model=%s", settings.domain_name, settings.base_model_name
    )