"""Structured logging configuration used across training, evaluation, and API."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure a single, consistent root logger for the whole application."""
    root = logging.getLogger()
    if root.handlers:
        # Avoid duplicate handlers when called multiple times (e.g. in tests).
        root.setLevel(level.upper())
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)