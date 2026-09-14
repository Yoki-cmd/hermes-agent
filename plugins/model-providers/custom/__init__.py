"""Custom / Ollama (local) provider profile.

Covers any endpoint registered as provider="custom", including local
Ollama instances and OpenAI-compatible reasoning endpoints (GLM-5.2 on
Volcengine ARK, vLLM, llama.cpp). Key quirks:
  - ollama_num_ctx → extra_body.options.num_ctx (local context window)
  - reasoning_config disabled → top-level reasoning_effort="none"
    (Ollama /v1/chat/completions ignores think=False — ollama#14820)
    + extra_body.think = False for /api/chat and proxies
  - reasoning_config enabled + effort → top-level reasoning_effort
    (the native OpenAI-compatible format GLM/ARK expect; unset omits it
    so the endpoint's server default applies). For loopback Ollama
    endpoints the effort is first clamped onto {none,low,medium,high,max} —
    Ollama 400s on Hermes-only levels such as ultra/minimal/xhigh.
"""

from typing import Any
from urllib.parse import urlsplit

from providers import register_provider
from providers.base import ProviderProfile


# Ollama's OpenAI-compatible endpoint accepts exactly these reasoning_effort
# values and HTTP-400s on anything else (verified against Ollama 0.32.x):
#   "invalid reasoning value: 'ultra' (must be "high", "medium", "low",
#    "max", or "none")"
# Hermes' wider ladder (minimal/low/medium/high/xhigh/max/ultra) must be
# clamped before it reaches a local Ollama server — otherwise a session-level
# effort such as the global "ultra" 400s every request to the local model.
# (Local fork patch, 2026-09 — deliberately not upstreamed.)
_OLLAMA_LEGAL_EFFORTS = frozenset({"none", "low", "medium", "high", "max"})
_OLLAMA_EFFORT_CLAMP = {
    "minimal": "low",   # nearest legal level
    "xhigh": "max",     # both sit above "high"
    "ultra": "max",     # preserve the "deepest thinking" intent
}


def _is_local_ollama_base_url(base_url: str | None) -> bool:
    """True for loopback Ollama endpoints (127.0.0.1/localhost/::1 + :11434).

    Port-scoped to the Ollama default so other loopback custom endpoints
    (e.g. a local OpenAI-compatible sidecar on another port) keep receiving
    their exact effort string.
    """
    if not base_url:
        return False
    try:
        parts = urlsplit(str(base_url))
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        return False
    if host == "localhost" or host == "::1" or host.startswith("127."):
        return port == 11434
    return False


def _clamp_ollama_effort(effort: str) -> str:
    """Map a Hermes effort level onto Ollama's accepted set."""
    if effort in _OLLAMA_LEGAL_EFFORTS:
        return effort
    mapped = _OLLAMA_EFFORT_CLAMP.get(effort)
    if mapped is not None:
        return mapped
    # Unknown/future level: fall back to a legal value rather than 400-ing
    # the whole request.
    return "high"


class CustomProfile(ProviderProfile):
    """Custom/Ollama local provider — think=false and num_ctx support."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        ollama_num_ctx: int | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        # Ollama context window
        if ollama_num_ctx:
            options = extra_body.get("options", {})
            options["num_ctx"] = ollama_num_ctx
            extra_body["options"] = options

        # Reasoning / thinking control for custom OpenAI-compatible endpoints
        # (GLM-5.2 on Volcengine ARK, vLLM, Ollama, llama.cpp, …).
        #
        #   - disabled  → extra_body.think = False (Ollama's thinking-off flag)
        #   - enabled + effort set → TOP-LEVEL reasoning_effort string, the
        #     format GLM-5.2/ARK and other OpenAI-compatible reasoning APIs
        #     expect (GLM documents "high" and "max"; "max" is its default).
        #     For LOCAL OLLAMA endpoints the value is first clamped onto the
        #     server's accepted set — see _clamp_ollama_effort above.
        #   - enabled + no effort  → omit both, so the endpoint applies its own
        #     server-side default (do NOT force a level the user didn't pick).
        #
        # We deliberately do NOT emit ``think=True`` on enable: it is an
        # Ollama-only flag and thinking is already server-default-on for these
        # backends, so forcing it risks a 400 on GLM/vLLM endpoints that don't
        # recognize it. Mirrors the DeepSeek/Zai profile precedent.
        if reasoning_config and isinstance(reasoning_config, dict):
            _effort = (reasoning_config.get("effort") or "").strip().lower()
            _enabled = reasoning_config.get("enabled", True)
            if _effort == "none" or _enabled is False:
                # Ollama's /v1/chat/completions silently ignores
                # extra_body.think (only /api/chat honours it — ollama#14820)
                # but respects the top-level reasoning_effort field, so both
                # are needed to actually stop a thinking-capable model from
                # reasoning (#25758). Endpoints that recognize neither simply
                # ignore them.
                top_level["reasoning_effort"] = "none"
                extra_body["think"] = False
            elif _effort:
                if _is_local_ollama_base_url(ctx.get("base_url")):
                    _effort = _clamp_ollama_effort(_effort)
                top_level["reasoning_effort"] = _effort

        return extra_body, top_level

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Custom/Ollama: base_url is user-configured; fetch if set."""
        if not (base_url or self.base_url):
            return None
        return super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)


custom = CustomProfile(
    name="custom",
    aliases=(
        "ollama",
        "local",
        "vllm",
        "llamacpp",
        "llama.cpp",
        "llama-cpp",
    ),
    env_vars=(),  # No fixed key — custom endpoint
    base_url="",  # User-configured
    # Without this, no max_tokens is sent and Ollama falls back to its internal
    # num_predict=128, truncating responses after a few tokens (#39281). This is
    # only a floor used when the user hasn't set model.max_tokens — they can
    # override per-model — so we set it generously rather than lowballing it.
    default_max_tokens=65536,
)

register_provider(custom)
