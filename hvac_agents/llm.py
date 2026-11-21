"""
llm.py - One small client used by every agent to call the language model.

Default model: Claude Haiku 5.5 (the model in the group's cost plan).
Safety and cost rules from the proposal:
  * temperature 0, token cap per call, response cache (same prompt -> no new call)
  * the LLM never does math: numbers always come from the tools
  * timeout or bad JSON -> ONE repair retry -> the caller uses a template
If there is no ANTHROPIC_API_KEY (or the `anthropic` package is missing),
the client runs in TEMPLATE mode and every agent uses its deterministic text.
"""
import hashlib
import json
import os
import re

from . import config

# USD per million tokens (Cesar's cost plan, checked 11/18/2026)
PRICE_IN, PRICE_OUT = 0.10, 0.50


class LLMUnavailable(Exception):
    pass


class LLMClient:
    def __init__(self, mode=None, fake=None):
        """
        mode: "anthropic" | "template" | None (auto: anthropic if a key exists)
        fake: optional function(system, prompt) -> str, used by the tests to
              simulate timeouts and broken JSON (TC-11, TC-12).
        """
        self.fake = fake
        self.cache = {}
        self.last_error = None          # shown on the page and in the console (never the key)
        self._use_temperature = True
        self.calls = self.tokens_in = self.tokens_out = 0
        self._client = None
        if fake is not None:
            self.mode = "fake"
        elif mode == "template":
            self.mode = "template"
        else:
            key = os.getenv("ANTHROPIC_API_KEY", "")
            try:
                import anthropic  # noqa: F401
                has_pkg = True
            except ImportError:
                has_pkg = False
            self.mode = "anthropic" if (key and has_pkg and mode != "template") else "template"
            if self.mode == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic(api_key=key, timeout=config.LLM_TIMEOUT_S)

    @property
    def available(self):
        return self.mode in ("anthropic", "fake")

    @property
    def cost_usd(self):
        return (self.tokens_in * PRICE_IN + self.tokens_out * PRICE_OUT) / 1_000_000

    def text(self, system, prompt, max_tokens=None):
        """Return model text. Raises LLMUnavailable in template mode or on failure."""
        if not self.available:
            raise LLMUnavailable("template mode")
        if not (prompt or "").strip():
            raise LLMUnavailable("empty prompt: nothing to send")      # the API refuses empty messages (400)
        key = hashlib.sha256((system + "\n" + prompt).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        self.calls += 1
        try:
            if self.mode == "fake":
                out = self.fake(system, prompt)
            else:
                args = dict(model=config.LLM_MODEL, max_tokens=max_tokens or config.LLM_MAX_TOKENS,
                            system=system, messages=[{"role": "user", "content": prompt}])
                if self._use_temperature:
                    try:
                        resp = self._client.messages.create(temperature=config.LLM_TEMPERATURE, **args)
                    except (TypeError, ValueError) as exc:
                        if "temperature" not in str(exc):
                            raise
                        # newer SDK/model versions do not accept temperature: call again without it
                        self._use_temperature = False
                        resp = self._client.messages.create(**args)
                else:
                    resp = self._client.messages.create(**args)
                out = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                self.tokens_in += resp.usage.input_tokens
                self.tokens_out += resp.usage.output_tokens
        except Exception as exc:  # timeout, network, rate limit ...
            self.last_error = explain_error(exc)
            print(f"  [LLM error] {self.last_error}")
            raise LLMUnavailable(self.last_error) from exc
        if not (out or "").strip():
            # an empty answer is a failure: do not cache it, let the caller use its template
            self.last_error = "the model returned an empty answer (template text used instead)"
            raise LLMUnavailable(self.last_error)
        self.last_error = None
        self.cache[key] = out
        return out

    def ping(self):
        """One tiny test call at start-up. Returns (ok, message)."""
        if not self.available:
            return False, "template mode (no API key)"
        try:
            self.text("Reply with the single word OK.", "test " + str(id(self)), max_tokens=5)
            return True, f"{config.LLM_MODEL} answered"
        except LLMUnavailable as exc:
            return False, str(exc)

    def json(self, system, prompt, required_keys=()):
        """Ask for JSON. One repair retry; raises LLMUnavailable if still invalid."""
        sys2 = system + "\nAnswer with ONE valid JSON object only. No other text."
        raw = self.text(sys2, prompt)
        data = _parse_json(raw, required_keys)
        if data is not None:
            return data
        repair = (f"Your previous answer was not valid JSON with keys {list(required_keys)}:\n"
                  f"{raw[:800]}\nReturn only the corrected JSON object.")
        data = _parse_json(self.text(sys2, repair), required_keys)
        if data is None:
            raise LLMUnavailable("invalid JSON after one repair retry")
        return data


def _parse_json(raw, required_keys):
    if not raw:
        return None
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or any(k not in data for k in required_keys):
        return None
    return data


def explain_error(exc):
    """Short, friendly explanation of an API error (the key itself is never printed)."""
    name = type(exc).__name__
    text = str(exc)[:300]
    hints = {
        "AuthenticationError": "the API key was rejected (401). Create a new key at console.anthropic.com > "
                               "API Keys (it starts with sk-ant-api) and put it in the .env file",
        "PermissionDeniedError": "this key has no permission for the model (403)",
        "NotFoundError": f"model '{config.LLM_MODEL}' not found (404); check HVAC_LLM_MODEL in .env",
        "RateLimitError": "rate limit reached (429); wait a minute",
        "APIConnectionError": "cannot reach the Anthropic API; check the internet connection",
        "APITimeoutError": "the API call timed out",
    }
    if "credit balance" in text.lower():
        return "the Anthropic account has no credit; add credit in console.anthropic.com > Billing"
    return f"{name}: {hints.get(name, text)}"
