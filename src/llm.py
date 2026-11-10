"""Minimal client for one-shot calls to a language model.

Two providers are supported: Google Gemini and Anthropic. LLM_PROVIDER picks
one ("gemini" or "anthropic", default "gemini"), and each provider has its own
key, GEMINI_API_KEY or ANTHROPIC_API_KEY, and GEMINI_MODEL optionally
overrides the default Gemini model. Every setting is read from the
environment first, then from .streamlit/secrets.toml. Every failure becomes an
LLMError with a message a user can read. Keys are never printed, logged or
included in an error.
"""

import os
import re
import time
import tomllib
from pathlib import Path

import anthropic
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

try:
    import httpx2
except ImportError:  # pragma: no cover - google-genai and anthropic depend on it
    httpx2 = None
try:
    import httpx
except ImportError:
    httpx = None

ANTHROPIC_MODEL = "claude-haiku-5-5"
# Default Gemini model: Google's current stable Flash model, free of charge on
# the Gemini free tier. The GEMINI_MODEL setting overrides it.
GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_MODEL_SETTING = "GEMINI_MODEL"
MODELS = {"gemini": GEMINI_MODEL, "anthropic": ANTHROPIC_MODEL}
PROVIDERS = tuple(MODELS)
PROVIDER_LABELS = {"gemini": "Gemini", "anthropic": "Anthropic"}
DEFAULT_PROVIDER = "gemini"
PROVIDER_SETTING = "LLM_PROVIDER"
KEY_NAMES = {"gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
PLACEHOLDER_KEYS = ("your-key-here", "your-gemini-key-here", "your-anthropic-key-here")

TIMEOUT_SECONDS = 30.0  # per attempt
# Gemini's 503 (model overloaded) is usually brief, so it alone is retried once.
GEMINI_RETRY_STATUS = 503
GEMINI_RETRY_DELAY_SECONDS = 2.0
DEFAULT_MAX_TOKENS = 4096
# Neither model is given a temperature: Haiku 5.5 rejects sampling settings,
# so both use a low reasoning setting instead, which suits short extraction.
ANTHROPIC_EFFORT = "low"
GEMINI_THINKING_LEVEL = genai_types.ThinkingLevel.LOW
SECRETS_PATH = Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml"

_HTTP_TIMEOUTS = tuple(lib.TimeoutException for lib in (httpx2, httpx) if lib is not None)
_HTTP_ERRORS = tuple(lib.HTTPError for lib in (httpx2, httpx) if lib is not None)
_FINISHED = (None, genai_types.FinishReason.STOP, genai_types.FinishReason.FINISH_REASON_UNSPECIFIED)
_BLOCKED = (
    genai_types.FinishReason.SAFETY,
    genai_types.FinishReason.RECITATION,
    genai_types.FinishReason.BLOCKLIST,
    genai_types.FinishReason.PROHIBITED_CONTENT,
    genai_types.FinishReason.SPII,
)


CODE_FENCE = re.compile(r"^```[A-Za-z]*\s*(.*?)\s*```$", re.DOTALL)


class LLMError(RuntimeError):
    """Raised when the model cannot be reached or gives no usable reply."""


def strip_code_fence(text: str) -> str:
    """Return the text inside a Markdown code fence, or the text unchanged."""
    match = CODE_FENCE.match(text.strip())
    return match.group(1) if match else text.strip()


def _read_secrets() -> dict:
    """Return the contents of secrets.toml, or {} if the file does not exist.

    Raises LLMError if the file cannot be read as TOML. The error message
    never contains the file's contents.
    """
    if not SECRETS_PATH.exists():
        return {}
    try:
        with SECRETS_PATH.open("rb") as secrets_file:
            return tomllib.load(secrets_file)
    except (OSError, tomllib.TOMLDecodeError):
        raise LLMError(f"Could not read {SECRETS_PATH.name}: check that it is valid TOML.") from None


def _setting(name: str) -> str | None:
    """Return a non-blank string setting from the environment, then secrets.toml."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        value = _read_secrets().get(name)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def get_provider() -> str:
    """Return the configured provider, "gemini" or "anthropic".

    Defaults to "gemini" when LLM_PROVIDER is not set. Raises LLMError for any
    other value; the value itself is not repeated, in case it is a key.
    """
    provider = _setting(PROVIDER_SETTING)
    if provider is None:
        return DEFAULT_PROVIDER
    provider = provider.lower()
    if provider not in PROVIDERS:
        raise LLMError(f"{PROVIDER_SETTING} must be one of: {', '.join(PROVIDERS)}.")
    return provider


def get_model(provider: str) -> str:
    """Return the model for a provider; for Gemini, the GEMINI_MODEL setting if set."""
    if provider == "gemini":
        return _setting(GEMINI_MODEL_SETTING) or GEMINI_MODEL
    return MODELS[provider]


def active_model() -> tuple[str, str]:
    """Return (provider, model) for the configured provider."""
    provider = get_provider()
    return provider, get_model(provider)


def get_api_key(provider: str | None = None) -> str | None:
    """Return the API key for a provider (default: the configured one), or None.

    Blank values and the placeholders from secrets.toml.example count as
    missing. Raises LLMError if secrets.toml is not valid TOML.
    """
    provider = get_provider() if provider is None else provider
    key = _setting(KEY_NAMES[provider])
    return None if key is None or key in PLACEHOLDER_KEYS else key


def has_api_key() -> bool:
    """Return True if the configured provider has an API key."""
    try:
        return get_api_key() is not None
    except LLMError:
        return False


def call_llm(system: str, user: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """Send one system prompt and user message to the model and return the reply text.

    Uses the configured provider's model with a TIMEOUT_SECONDS timeout per
    attempt. The only retry is one more Gemini attempt after a 503, two
    seconds later, so a caller waits at most about 62 seconds. Raises
    LLMError if there is no key, the request fails, the model declines or the
    reply is empty or cut off.
    """
    provider = get_provider()
    api_key = get_api_key(provider)
    if api_key is None:
        raise LLMError(
            f"No {PROVIDER_LABELS[provider]} API key found. Set {KEY_NAMES[provider]} "
            "or add it to .streamlit/secrets.toml."
        )
    if provider == "gemini":
        return _call_gemini(api_key, get_model("gemini"), system, user, max_tokens)
    return _call_anthropic(api_key, system, user, max_tokens)


def _timeout_error() -> LLMError:
    return LLMError(f"The model did not respond within {TIMEOUT_SECONDS:g} seconds.")


def _call_anthropic(api_key: str, system: str, user: str, max_tokens: int) -> str:
    """Return the reply text from one Anthropic Messages API call."""
    client = anthropic.Anthropic(api_key=api_key, timeout=TIMEOUT_SECONDS, max_retries=0)
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": ANTHROPIC_EFFORT},
        )
    except anthropic.AuthenticationError:
        raise LLMError("The Anthropic API key was rejected. Check that it is correct.") from None
    except anthropic.PermissionDeniedError:
        raise LLMError("The Anthropic API key does not have access to this model.") from None
    except anthropic.NotFoundError:
        raise LLMError(f"The model {ANTHROPIC_MODEL} was not found.") from None
    except anthropic.RateLimitError:
        raise LLMError("The Anthropic API rate limit was reached. Try again shortly.") from None
    except anthropic.APITimeoutError:
        raise _timeout_error() from None
    except anthropic.APIConnectionError:
        raise LLMError("Could not connect to the Anthropic API. Check the internet connection.") from None
    except anthropic.APIStatusError as error:
        raise LLMError(f"The Anthropic API returned an error (status {error.status_code}).") from None
    except anthropic.AnthropicError:
        raise LLMError("The request to the Anthropic API failed.") from None

    if response.stop_reason == "refusal":
        raise LLMError("The model declined this request.")
    if response.stop_reason == "max_tokens":
        raise LLMError("The model's reply was cut off before it finished.")
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise LLMError("The model returned an empty reply.")
    return text


def _gemini_status_error(error: genai_errors.APIError, model: str) -> LLMError:
    """Return a readable LLMError for a Gemini HTTP error status."""
    code = error.code
    # A bad key comes back as 400 INVALID_ARGUMENT with reason API_KEY_INVALID.
    if code == 401 or (code == 400 and "API_KEY_INVALID" in str(error.details)):
        return LLMError("The Gemini API key was rejected. Check that it is correct.")
    if code == 403:
        return LLMError("The Gemini API key does not have access to this model.")
    if code == 404:
        return LLMError(f"The model {model} was not found.")
    if code == 429:
        return LLMError("The Gemini API rate limit or free-tier quota was reached. Try again shortly.")
    if code == GEMINI_RETRY_STATUS:
        return LLMError(
            f"The model {model} is temporarily unavailable (status 503). Try again later, "
            f"or set {GEMINI_MODEL_SETTING} to another model."
        )
    if code in (408, 504):
        return _timeout_error()
    return LLMError(f"The Gemini API returned an error (status {code}).")


def _generate_with_retry(client, model: str, user: str, config):
    """Call generate_content, retrying once after a short wait if the reply is 503.

    The SDK's own retries are off, so this is the only retry; every other
    error, and a second 503, propagates.
    """
    try:
        return client.models.generate_content(model=model, contents=user, config=config)
    except genai_errors.APIError as error:
        if error.code != GEMINI_RETRY_STATUS:
            raise
    time.sleep(GEMINI_RETRY_DELAY_SECONDS)
    return client.models.generate_content(model=model, contents=user, config=config)


def _call_gemini(api_key: str, model: str, system: str, user: str, max_tokens: int) -> str:
    """Return the reply text from a Gemini generate_content call (one retry on 503)."""
    try:
        client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(
                timeout=int(TIMEOUT_SECONDS * 1000),  # milliseconds, per attempt
                retry_options=genai_types.HttpRetryOptions(attempts=1),
            ),
        )
        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            thinking_config=genai_types.ThinkingConfig(thinking_level=GEMINI_THINKING_LEVEL),
        )
        response = _generate_with_retry(client, model, user, config)
    except genai_errors.APIError as error:
        raise _gemini_status_error(error, model) from None
    except _HTTP_TIMEOUTS:
        raise _timeout_error() from None
    except _HTTP_ERRORS:
        raise LLMError("Could not connect to the Gemini API. Check the internet connection.") from None
    except Exception:  # anything else from the SDK, e.g. a reply it cannot parse
        raise LLMError("The request to the Gemini API failed.") from None

    feedback = response.prompt_feedback
    if feedback is not None and feedback.block_reason:
        raise LLMError("The model declined this request.")
    if not response.candidates:
        raise LLMError("The model returned an empty reply.")
    candidate = response.candidates[0]
    finish = candidate.finish_reason
    if finish == genai_types.FinishReason.MAX_TOKENS:
        raise LLMError("The model's reply was cut off before it finished.")
    if finish in _BLOCKED:
        raise LLMError("The model declined this request.")
    if finish not in _FINISHED:
        raise LLMError(f"The model stopped before finishing ({getattr(finish, 'value', finish)}).")
    parts = candidate.content.parts if candidate.content and candidate.content.parts else []
    text = "".join(part.text for part in parts if part.text and not part.thought).strip()
    if not text:
        raise LLMError("The model returned an empty reply.")
    return text
