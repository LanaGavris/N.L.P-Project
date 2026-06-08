"""
llm_client.py — Unified LLM Client
════════════════════════════════════
Switch between Google Gemini and OpenRouter with one variable.

Usage:
    from llm_client import LLMClient, get_client

    client = get_client()          # auto-detects from env vars
    response = client.call("...")
    json_resp = client.call_json("...")

Providers:
    Google Gemini (free tier):
        GOOGLE_API_KEY = "AIza..."
        Free: gemini-2.5-flash-lite (1500 req/day), gemini-2.5-flash (500/day)
        Get key: https://aistudio.google.com/apikey

    OpenRouter (free tier):
        OPENROUTER_API_KEY = "sk-or-..."
        Free: openrouter/free (auto-selects best available free model)
              OR specific models ending in :free
        Limits: 20 req/min, 200 req/day
        Get key: https://openrouter.ai → Sign in → Keys
        Best free models (May 2026):
          - openrouter/free              ← auto-select (recommended)
          - meta-llama/llama-4-maverick:free  (1M context, best quality)
          - deepseek/deepseek-r1:free    (strong reasoning)
          - deepseek/deepseek-v3:free    (general purpose)
          - qwen/qwen3-235b-a22b:free    (large, good quality)
          - google/gemma-3-27b-it:free   (Google, reliable)
"""

import os
import re
import json
import time
import requests
from typing import Optional


# ─── Provider constants ───────────────────────────────────────────────────

# Google Gemini
GEMINI_FAST  = "gemini-2.5-flash-lite"   # 1500 req/day free
GEMINI_SMART = "gemini-2.5-flash"        # 500  req/day free
GEMINI_URL   = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_DELAY_FAST  = 2.5   # 30 RPM → safe
GEMINI_DELAY_SMART = 4.5   # 15 RPM → safe

# OpenRouter
OR_BASE_URL  = "https://openrouter.ai/api/v1/chat/completions"
OR_AUTO_FREE = "openrouter/free"   # auto-selects best free model

# Specific free models ranked by quality (May 2026)
OR_FREE_MODELS = [
    "meta-llama/llama-4-maverick:free",   # best quality, 1M context
    "deepseek/deepseek-r1:free",          # strong reasoning
    "deepseek/deepseek-v3:free",          # general purpose
    "qwen/qwen3-235b-a22b:free",          # large model
    "google/gemma-3-27b-it:free",         # reliable fallback
]

OR_DELAY = 4.0   # 20 RPM → safe (3s minimum, 4s for buffer)


# ─── Unified client ───────────────────────────────────────────────────────

class LLMClient:
    """
    Unified interface for Gemini and OpenRouter.
    All pipeline modules use this instead of calling APIs directly.
    """

    def __init__(
        self,
        provider: str = "auto",          # "gemini" | "openrouter" | "auto"
        gemini_key: Optional[str] = None,
        openrouter_key: Optional[str] = None,
        gemini_fast_model: str = GEMINI_FAST,
        gemini_smart_model: str = GEMINI_SMART,
        openrouter_model: str = OR_AUTO_FREE,
        request_delay: Optional[float] = None,  # None = auto
        retries: int = 3,
    ):
        self.gemini_key    = gemini_key    or os.environ.get("GOOGLE_API_KEY", "")
        self.or_key        = openrouter_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.retries       = retries

        # Model config
        self.gemini_fast   = gemini_fast_model
        self.gemini_smart  = gemini_smart_model
        self.or_model      = openrouter_model

        # Determine active provider
        if provider == "auto":
            if self.or_key:
                self.provider = "openrouter"
            elif self.gemini_key:
                self.provider = "gemini"
            else:
                self.provider = "none"
        else:
            self.provider = provider

        # Delay
        if request_delay is not None:
            self.delay = request_delay
        elif self.provider == "openrouter":
            self.delay = OR_DELAY
        else:
            self.delay = GEMINI_DELAY_FAST

        self._print_config()

    def _print_config(self):
        if self.provider == "gemini":
            print(f"[LLM] Provider: Google Gemini")
            print(f"      Fast model:  {self.gemini_fast}  (1500 req/day free)")
            print(f"      Smart model: {self.gemini_smart} (500  req/day free)")
            print(f"      Delay: {self.delay}s/request")
        elif self.provider == "openrouter":
            print(f"[LLM] Provider: OpenRouter")
            print(f"      Model: {self.or_model}")
            print(f"      Limits: 20 req/min, 200 req/day (free tier)")
            print(f"      Delay: {self.delay}s/request")
            if self.or_model == OR_AUTO_FREE:
                print(f"      Auto-selecting from best available free models")
        else:
            print(f"[LLM] No API key set — mock responses will be returned")
            print(f"      Set GOOGLE_API_KEY or OPENROUTER_API_KEY")

    @property
    def is_configured(self) -> bool:
        return self.provider in ("gemini", "openrouter")

    # ─── Main call methods ─────────────────────────────────────────────────

    def call(
        self,
        prompt: str,
        system: str = "",
        use_smart_model: bool = False,   # True = use smarter model (slower/fewer quota)
        temperature: float = 0.1,
        max_tokens: int = 800,
    ) -> str:
        """Call LLM and return text response."""
        if not self.is_configured:
            return "[MOCK] No API key configured"

        if self.provider == "gemini":
            return self._call_gemini(
                prompt, system,
                model=self.gemini_smart if use_smart_model else self.gemini_fast,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            return self._call_openrouter(
                prompt, system,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    def call_json(
        self,
        prompt: str,
        system: str = "",
        use_smart_model: bool = True,
        temperature: float = 0.0,
    ) -> dict:
        """Call LLM and parse JSON response."""
        json_system = (
            system + "\nIMPORTANT: Respond ONLY with valid JSON. "
            "No markdown, no code fences, no explanation outside the JSON."
        ).strip()

        raw = self.call(
            prompt=prompt,
            system=json_system,
            use_smart_model=use_smart_model,
            temperature=temperature,
            max_tokens=400,
        )

        if raw.startswith("[MOCK]") or raw.startswith("[Error]"):
            return {"error": raw, "label": 0, "confidence": 0.5,
                    "reasoning": raw, "topic": "unknown"}

        return self._parse_json(raw)

    def sleep(self):
        """Call after each request to respect rate limits."""
        time.sleep(self.delay)

    # ─── Gemini implementation ─────────────────────────────────────────────

    def _call_gemini(
        self,
        prompt: str,
        system: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        url  = GEMINI_URL.format(model=model)
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        for attempt in range(self.retries):
            try:
                resp = requests.post(
                    url,
                    params={"key": self.gemini_key},
                    headers={"Content-Type": "application/json"},
                    json=body,
                    timeout=30,
                )
                if resp.status_code == 429:
                    wait = 20 * (attempt + 1)
                    print(f"  [Gemini] Rate limit — waiting {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                parts = (resp.json()
                         .get("candidates", [{}])[0]
                         .get("content", {})
                         .get("parts", []))
                return parts[0].get("text", "").strip() if parts else "[Error] Empty response"
            except requests.exceptions.Timeout:
                if attempt < self.retries - 1:
                    time.sleep(5)
                else:
                    return "[Error] Gemini timeout"
            except Exception as e:
                if attempt < self.retries - 1:
                    time.sleep(3)
                else:
                    return f"[Error] Gemini: {e}"
        return "[Error] Gemini: all retries exhausted"

    # ─── OpenRouter implementation ─────────────────────────────────────────

    def _call_openrouter(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.retries):
            try:
                resp = requests.post(
                    OR_BASE_URL,
                    headers={
                        "Authorization": f"Bearer {self.or_key}",
                        "Content-Type":  "application/json",
                        "HTTP-Referer":  "https://github.com/legal-contradiction-rag",
                        "X-Title":       "Legal Contradiction RAG",
                    },
                    json={
                        "model":       self.or_model,
                        "messages":    messages,
                        "temperature": temperature,
                        "max_tokens":  max_tokens,
                    },
                    timeout=45,
                )
                if resp.status_code == 429:
                    wait = 30 * (attempt + 1)
                    print(f"  [OpenRouter] Rate limit — waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if resp.status_code == 200:
                    data = resp.json()
                    # If openrouter/free returned, log which model was chosen
                    if self.or_model == OR_AUTO_FREE:
                        chosen = data.get("model", "unknown")
                        if attempt == 0 and not hasattr(self, "_logged_model"):
                            print(f"  [OpenRouter] Auto-selected: {chosen}")
                            self._logged_model = chosen
                    return (data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                            .strip())

                resp.raise_for_status()

            except requests.exceptions.Timeout:
                if attempt < self.retries - 1:
                    time.sleep(5)
                else:
                    return "[Error] OpenRouter timeout"
            except Exception as e:
                if attempt < self.retries - 1:
                    time.sleep(3)
                else:
                    return f"[Error] OpenRouter: {e}"
        return "[Error] OpenRouter: all retries exhausted"

    # ─── JSON parser ──────────────────────────────────────────────────────

    def _parse_json(self, raw: str) -> dict:
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except Exception:
                    pass
        return {
            "error": f"JSON parse failed: {cleaned[:80]}",
            "label": 0,
            "confidence": 0.5,
            "reasoning": "Could not parse response.",
            "topic": "unknown",
        }

    # ─── Provider info ────────────────────────────────────────────────────

    def info(self) -> dict:
        return {
            "provider":      self.provider,
            "configured":    self.is_configured,
            "delay_seconds": self.delay,
            "gemini_key_set":  bool(self.gemini_key),
            "or_key_set":      bool(self.or_key),
            "models": {
                "gemini_fast":  self.gemini_fast,
                "gemini_smart": self.gemini_smart,
                "openrouter":   self.or_model,
            },
        }


# ─── Factory function ─────────────────────────────────────────────────────

def get_client(
    provider: str = "auto",
    gemini_key: Optional[str] = None,
    openrouter_key: Optional[str] = None,
    openrouter_model: str = OR_AUTO_FREE,
) -> LLMClient:
    """
    Create an LLM client. Called once in your notebook Cell 2.

    Examples:
        client = get_client()                           # auto-detect
        client = get_client(provider="openrouter")      # force OpenRouter
        client = get_client(provider="gemini")          # force Gemini
        client = get_client(
            provider="openrouter",
            openrouter_model="meta-llama/llama-4-maverick:free"
        )
    """
    return LLMClient(
        provider=provider,
        gemini_key=gemini_key,
        openrouter_key=openrouter_key,
        openrouter_model=openrouter_model,
    )


# ─── Backwards compatibility shims ────────────────────────────────────────
# So that old pipeline files that still call google_llm functions
# can use the new client if google_llm.py is replaced with this.

_default_client: Optional[LLMClient] = None

def _get_default() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = get_client()
    return _default_client

def call_gemini(prompt, system="", model=None, api_key=None,
                temperature=0.1, max_tokens=800, retries=3) -> str:
    """Backwards-compatible shim."""
    c = _get_default()
    return c.call(prompt, system, temperature=temperature, max_tokens=max_tokens)

def call_gemini_json(prompt, system="", model=None, api_key=None,
                     temperature=0.0) -> dict:
    """Backwards-compatible shim."""
    c = _get_default()
    return c.call_json(prompt, system, temperature=temperature)

def get_delay(model: str = "") -> float:
    return _get_default().delay

# Constants for files that import them
FAST_MODEL  = GEMINI_FAST
SMART_MODEL = GEMINI_SMART


# ─── Quick test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== LLM Client Test ===\n")
    client = get_client()
    print(f"\nConfig: {client.info()}\n")

    if client.is_configured:
        print("Testing text call...")
        resp = client.call("Say 'API working' and nothing else.")
        print(f"Response: {resp}")

        print("\nTesting JSON call...")
        result = client.call_json(
            prompt='Return JSON: {"status": "ok", "provider": "test"}',
        )
        print(f"Parsed: {result}")
    else:
        print("No API keys set. Configure in Cell 2 of the notebook.")
