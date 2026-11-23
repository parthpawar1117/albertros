"""Shared LLM configuration for all agents."""

import os

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


def groq_model() -> str:
    """Groq model used by every agent; override with the GROQ_MODEL env var."""
    return os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
