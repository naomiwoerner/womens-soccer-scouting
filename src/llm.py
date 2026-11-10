"""Minimal client for one-shot calls to Claude.

The API key is read from the ANTHROPIC_API_KEY environment variable, or else
from .streamlit/secrets.toml. Every failure becomes an LLMError with a message
a user can read. The key is never printed, logged or included in an error.
"""

import os
import tomllib
from pathlib import Path

import anthropic

MODEL = "claude-haiku-5-5"
TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_TOKENS = 4096
# Haiku 5.5 rejects sampling settings such as temperature, so effort is the
# setting used here; low suits short extraction tasks.
EFFORT = "low"
KEY_NAME = "ANTHROPIC_API_KEY"
PLACEHOLDER_KEY = "your-key-here"
SECRETS_PATH = Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml"


class LLMError(RuntimeError):
    """Raised when Claude cannot be reached or gives no usable reply."""


def _usable(key) -> str | None:
    """Return the key if it is a real value, or None for blanks and the placeholder."""
    if not isinstance(key, str):
        return None
    key = key.strip()
    return key if key and key != PLACEHOLDER_KEY else None


def get_api_key() -> str | None:
    """Return the API key from the environment, then secrets.toml, or None.

    Raises LLMError if secrets.toml exists but cannot be read as TOML. The
    error message never contains the file's contents.
    """
    from_environment = _usable(os.environ.get(KEY_NAME))
    if from_environment:
        return from_environment
    if not SECRETS_PATH.exists():
        return None
    try:
        with SECRETS_PATH.open("rb") as secrets_file:
            secrets = tomllib.load(secrets_file)
    except (OSError, tomllib.TOMLDecodeError):
        raise LLMError(f"Could not read {SECRETS_PATH.name}: check that it is valid TOML.") from None
    return _usable(secrets.get(KEY_NAME))


def has_api_key() -> bool:
    """Return True if an API key is configured."""
    try:
        return get_api_key() is not None
    except LLMError:
        return False


def call_llm(system: str, user: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """Send one system prompt and user message to Claude and return the reply text.

    Uses MODEL with a TIMEOUT_SECONDS timeout and no automatic retries, so a
    caller waits at most about that long. Raises LLMError if there is no key,
    the request fails, the model declines, or the reply is empty or cut off.
    """
    api_key = get_api_key()
    if api_key is None:
        raise LLMError(
            f"No Anthropic API key found. Set {KEY_NAME} or add it to .streamlit/secrets.toml."
        )
    client = anthropic.Anthropic(api_key=api_key, timeout=TIMEOUT_SECONDS, max_retries=0)
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": EFFORT},
        )
    except anthropic.AuthenticationError:
        raise LLMError("The Anthropic API key was rejected. Check that it is correct.") from None
    except anthropic.PermissionDeniedError:
        raise LLMError("The Anthropic API key does not have access to this model.") from None
    except anthropic.NotFoundError:
        raise LLMError(f"The model {MODEL} was not found.") from None
    except anthropic.RateLimitError:
        raise LLMError("The Anthropic API rate limit was reached. Try again shortly.") from None
    except anthropic.APITimeoutError:
        raise LLMError(f"Claude did not respond within {TIMEOUT_SECONDS:g} seconds.") from None
    except anthropic.APIConnectionError:
        raise LLMError("Could not connect to the Anthropic API. Check the internet connection.") from None
    except anthropic.APIStatusError as error:
        raise LLMError(f"The Anthropic API returned an error (status {error.status_code}).") from None
    except anthropic.AnthropicError:
        raise LLMError("The request to the Anthropic API failed.") from None

    if response.stop_reason == "refusal":
        raise LLMError("Claude declined this request.")
    if response.stop_reason == "max_tokens":
        raise LLMError("Claude's reply was cut off before it finished.")
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise LLMError("Claude returned an empty reply.")
    return text
