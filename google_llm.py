"""
google_llm.py — Compatibility shim
════════════════════════════════════
All pipeline files import from google_llm for backwards compatibility.
This file now delegates to llm_client.py which supports both
Google Gemini and OpenRouter.

To switch provider: set keys in Cell 2 of the notebook.
"""
from llm_client import (
    call_gemini,
    call_gemini_json,
    get_delay,
    FAST_MODEL,
    SMART_MODEL,
    LLMClient,
    get_client,
    OR_AUTO_FREE,
    OR_FREE_MODELS,
)

__all__ = [
    "call_gemini",
    "call_gemini_json",
    "get_delay",
    "FAST_MODEL",
    "SMART_MODEL",
    "LLMClient",
    "get_client",
]
