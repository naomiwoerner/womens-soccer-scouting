"""Check src/llm.py and the parser's provider record without any network calls.

Covers settings (provider, keys, Gemini model, secrets.toml), every Anthropic
and Gemini failure path with simulated SDK errors and responses, the single
retry after a Gemini 503, the request settings
sent to each SDK, and that no message or log line ever contains a key. Run
from the project root with: python scripts/run_llm_check.py
"""

import io
import logging
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import anthropic  # noqa: E402
import httpx2  # noqa: E402
from google.genai import errors as genai_errors  # noqa: E402
from google.genai import types as gt  # noqa: E402

import src.llm as llm  # noqa: E402
import src.requirement_parser as parser  # noqa: E402

FAKE_GEMINI_KEY = "AIza-FAKE-GEMINI-KEY-DO-NOT-LEAK"
FAKE_ANTHROPIC_KEY = "sk-ant-FAKE-ANTHROPIC-KEY-DO-NOT-LEAK"
FAKE_KEYS = (FAKE_GEMINI_KEY, FAKE_ANTHROPIC_KEY)
SETTING_NAMES = (llm.PROVIDER_SETTING, llm.GEMINI_MODEL_SETTING, *llm.KEY_NAMES.values())

failures: list[str] = []
log_stream = io.StringIO()


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'ok' if condition else 'FAIL'}] {label}{f': {detail}' if detail else ''}")
    if not condition:
        failures.append(label)


def expect_error(label: str, call, expected: str) -> None:
    """Check that call raises LLMError containing expected, with no key and no chained context."""
    try:
        result = call()
    except llm.LLMError as error:
        message = str(error)
        clean = not any(key in message for key in FAKE_KEYS)
        hidden = error.__context__ is None or error.__suppress_context__
        check(label, expected in message and clean and hidden, message)
        return
    check(label, False, f"returned {result!r} instead of raising")


def settings(env: dict, secrets: str | None):
    """Patch the environment and secrets.toml for one check."""
    folder = tempfile.mkdtemp()
    path = Path(folder) / "secrets.toml"
    if secrets is not None:
        path.write_text(secrets)
    clean_env = {k: v for k, v in os.environ.items() if k not in SETTING_NAMES}
    return (
        mock.patch.object(llm, "SECRETS_PATH", path),
        mock.patch.dict(os.environ, {**clean_env, **env}, clear=True),
    )


def with_settings(env: dict, secrets: str | None, call):
    patch_path, patch_env = settings(env, secrets)
    with patch_path, patch_env:
        return call()


def check_settings() -> None:
    print("Settings")
    both = f'GEMINI_API_KEY = "{FAKE_GEMINI_KEY}"\nANTHROPIC_API_KEY = "{FAKE_ANTHROPIC_KEY}"\n'
    check("no settings: provider defaults to gemini", with_settings({}, None, llm.get_provider) == "gemini")
    check("no settings: no key", with_settings({}, None, llm.has_api_key) is False)
    check(
        "provider from secrets.toml",
        with_settings({}, 'LLM_PROVIDER = "anthropic"\n', llm.active_model)
        == ("anthropic", llm.ANTHROPIC_MODEL),
    )
    check(
        "environment wins over secrets.toml, case-insensitive",
        with_settings({"LLM_PROVIDER": " Gemini "}, 'LLM_PROVIDER = "anthropic"\n', llm.active_model)
        == ("gemini", llm.GEMINI_MODEL),
    )
    check(
        "blank environment value falls through to secrets.toml",
        with_settings({"LLM_PROVIDER": "  "}, 'LLM_PROVIDER = "anthropic"\n', llm.get_provider) == "anthropic",
    )
    expect_error(
        "unknown provider is rejected without echoing the value",
        lambda: with_settings({"LLM_PROVIDER": FAKE_GEMINI_KEY}, None, llm.get_provider),
        "LLM_PROVIDER must be one of: gemini, anthropic",
    )
    check(
        "each provider reads its own key",
        with_settings({}, both, lambda: (llm.get_api_key("gemini"), llm.get_api_key("anthropic")))
        == (FAKE_GEMINI_KEY, FAKE_ANTHROPIC_KEY),
    )
    check(
        "the configured provider's key is used",
        with_settings({"LLM_PROVIDER": "anthropic"}, both, llm.get_api_key) == FAKE_ANTHROPIC_KEY,
    )
    example = (PROJECT_ROOT / ".streamlit" / "secrets.toml.example").read_text()
    check(
        "secrets.toml.example placeholders count as missing",
        with_settings({}, example, lambda: (llm.get_provider(), llm.get_api_key("gemini"), llm.get_api_key("anthropic")))
        == ("gemini", None, None),
    )
    check(
        "environment key wins over secrets.toml",
        with_settings({"GEMINI_API_KEY": "env-key"}, both, llm.get_api_key) == "env-key",
    )
    expect_error(
        "invalid TOML gives a readable error without the file's contents",
        lambda: with_settings({}, f'GEMINI_API_KEY = "{FAKE_GEMINI_KEY}\n', llm.get_api_key),
        "check that it is valid TOML",
    )
    check(
        "has_api_key is False on invalid TOML",
        with_settings({}, f'GEMINI_API_KEY = "{FAKE_GEMINI_KEY}\n', llm.has_api_key) is False,
    )
    for provider, name in llm.KEY_NAMES.items():
        expect_error(
            f"call_llm without a {provider} key",
            lambda: with_settings({"LLM_PROVIDER": provider}, None, lambda: llm.call_llm("s", "u")),
            f"Set {name}",
        )


def anthropic_client(*, raises=None, reply=None):
    client = mock.MagicMock()
    if raises is not None:
        client.messages.create.side_effect = raises
    else:
        client.messages.create.return_value = reply
    return client


def run_anthropic(client):
    env = {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": FAKE_ANTHROPIC_KEY}
    with mock.patch.object(llm.anthropic, "Anthropic", return_value=client) as constructor:
        result = with_settings(env, None, lambda: llm.call_llm("system", "user"))
    return result, constructor


def check_anthropic() -> None:
    print("Anthropic path")
    request = mock.Mock()

    def response(code):
        return mock.Mock(status_code=code, headers={}, request=request)

    cases = [
        ("401", anthropic.AuthenticationError("x", response=response(401), body=None), "key was rejected"),
        ("403", anthropic.PermissionDeniedError("x", response=response(403), body=None), "does not have access"),
        ("404", anthropic.NotFoundError("x", response=response(404), body=None), "was not found"),
        ("429", anthropic.RateLimitError("x", response=response(429), body=None), "rate limit"),
        ("500", anthropic.InternalServerError("x", response=response(500), body=None), "status 500"),
        ("timeout", anthropic.APITimeoutError(request=request), "did not respond within 30 seconds"),
        ("connection", anthropic.APIConnectionError(request=request), "Could not connect to the Anthropic API"),
    ]
    for label, error, expected in cases:
        expect_error(f"anthropic {label}", lambda: run_anthropic(anthropic_client(raises=error)), expected)

    def text(value):
        return types.SimpleNamespace(type="text", text=value)

    thinking = types.SimpleNamespace(type="thinking", thinking="")
    replies = [
        ("refusal", "refusal", [], "The model declined"),
        ("cut off", "max_tokens", [text("{")], "cut off"),
        ("empty", "end_turn", [thinking, text("  ")], "empty reply"),
    ]
    for label, stop, blocks, expected in replies:
        reply = types.SimpleNamespace(stop_reason=stop, content=blocks)
        expect_error(f"anthropic {label}", lambda: run_anthropic(anthropic_client(reply=reply)), expected)

    reply = types.SimpleNamespace(stop_reason="end_turn", content=[thinking, text(' {"a": 1} ')])
    result, constructor = run_anthropic(anthropic_client(reply=reply))
    check("anthropic text reply is joined and stripped", result == '{"a": 1}')
    client_args = constructor.call_args.kwargs
    sent = constructor.return_value.messages.create.call_args.kwargs
    check(
        "anthropic client: 30 s timeout, no retries",
        client_args["timeout"] == 30.0 and client_args["max_retries"] == 0,
    )
    check(
        "anthropic request: model, effort, no temperature",
        sent["model"] == llm.ANTHROPIC_MODEL
        and sent["output_config"] == {"effort": "low"}
        and "temperature" not in sent,
        f"model {sent['model']}",
    )


def gemini_reply(*, finish="STOP", parts=None, block_reason=None, candidates=True):
    if block_reason is not None:
        return gt.GenerateContentResponse(prompt_feedback=gt.GenerateContentResponsePromptFeedback(block_reason=block_reason))
    if not candidates:
        return gt.GenerateContentResponse(candidates=[])
    content = gt.Content(role="model", parts=parts) if parts is not None else None
    return gt.GenerateContentResponse(candidates=[gt.Candidate(content=content, finish_reason=finish)])


def gemini_client(*, raises=None, reply=None):
    client = mock.MagicMock()
    if raises is not None:
        client.models.generate_content.side_effect = raises
    else:
        client.models.generate_content.return_value = reply
    return client


def run_gemini(client, extra_env=None, sleeps=None):
    """Call the Gemini path with a fake client; record any retry wait in sleeps."""
    env = {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": FAKE_GEMINI_KEY, **(extra_env or {})}
    sleeps = [] if sleeps is None else sleeps
    with (
        mock.patch.object(llm.genai, "Client", return_value=client) as constructor,
        mock.patch.object(llm.time, "sleep", side_effect=sleeps.append),
    ):
        result = with_settings(env, None, lambda: llm.call_llm("system", "user"))
    return result, constructor


def check_gemini_retry() -> None:
    print("Gemini 503 retry")
    unavailable = api_error(genai_errors.ServerError, 503, "UNAVAILABLE", "The model is overloaded.")
    ok = gemini_reply(parts=[gt.Part(text='{"a": 1}')])

    client, sleeps = gemini_client(raises=[unavailable, ok]), []
    result, _ = run_gemini(client, sleeps=sleeps)
    check(
        "503 then success: reply returned after one 2 s wait",
        result == '{"a": 1}' and client.models.generate_content.call_count == 2 and sleeps == [2.0],
        f"calls {client.models.generate_content.call_count}, waits {sleeps}",
    )

    client, sleeps = gemini_client(raises=[unavailable, unavailable]), []
    expect_error(
        "503 twice: LLMError after the second attempt",
        lambda: run_gemini(client, sleeps=sleeps),
        "temporarily unavailable (status 503)",
    )
    check(
        "503 twice: exactly two attempts and one wait",
        client.models.generate_content.call_count == 2 and sleeps == [2.0],
        f"calls {client.models.generate_content.call_count}, waits {sleeps}",
    )

    request = httpx2.Request("POST", "https://generativelanguage.googleapis.com/")
    others = [
        ("429", api_error(genai_errors.ClientError, 429, "RESOURCE_EXHAUSTED", "Quota.")),
        ("500", api_error(genai_errors.ServerError, 500, "INTERNAL", "Oops.")),
        ("504", api_error(genai_errors.ServerError, 504, "DEADLINE_EXCEEDED", "Slow.")),
        ("bad key", api_error(genai_errors.ClientError, 400, "INVALID_ARGUMENT", "Bad.", "API_KEY_INVALID")),
        ("timeout", httpx2.ReadTimeout("timed out", request=request)),
        ("connection", httpx2.ConnectError("refused", request=request)),
    ]
    for label, error in others:
        client, sleeps = gemini_client(raises=[error, ok]), []
        try:
            run_gemini(client, sleeps=sleeps)
            raised = False
        except llm.LLMError:
            raised = True
        check(
            f"{label}: no retry",
            raised and client.models.generate_content.call_count == 1 and sleeps == [],
            f"calls {client.models.generate_content.call_count}",
        )


def check_gemini_model_setting() -> None:
    print("Gemini model setting")
    check("default model when unset", with_settings({}, None, llm.active_model) == ("gemini", llm.GEMINI_MODEL))
    check(
        "GEMINI_MODEL from secrets.toml",
        with_settings({}, 'GEMINI_MODEL = "gemini-from-file"\n', llm.active_model) == ("gemini", "gemini-from-file"),
    )
    check(
        "GEMINI_MODEL from environment wins",
        with_settings({"GEMINI_MODEL": "gemini-from-env"}, 'GEMINI_MODEL = "gemini-from-file"\n', llm.active_model)
        == ("gemini", "gemini-from-env"),
    )
    check(
        "GEMINI_MODEL does not affect Anthropic",
        with_settings({"LLM_PROVIDER": "anthropic", "GEMINI_MODEL": "x"}, None, llm.active_model)
        == ("anthropic", llm.ANTHROPIC_MODEL),
    )
    ok = gemini_reply(parts=[gt.Part(text="ok")])
    client = gemini_client(reply=ok)
    run_gemini(client, extra_env={"GEMINI_MODEL": "gemini-custom"})
    sent = client.models.generate_content.call_args.kwargs["model"]
    check("request uses the configured model", sent == "gemini-custom", sent)
    expect_error(
        "404 names the configured model",
        lambda: run_gemini(
            gemini_client(raises=api_error(genai_errors.ClientError, 404, "NOT_FOUND", "No.")),
            extra_env={"GEMINI_MODEL": "gemini-custom"},
        ),
        "The model gemini-custom was not found",
    )
    with mock.patch.object(parser, "call_llm", return_value='{"position_group": "Striker", "weights": {"npxg": 3}}'):
        result = with_settings({"GEMINI_MODEL": "gemini-custom"}, None, lambda: parser.parse_requirement("a striker"))
    check("parser records the configured model", result.model == "gemini-custom", str(result.model))


def api_error(cls, code, status, message, reason=None):
    body = {"error": {"code": code, "message": message, "status": status}}
    if reason:
        body["error"]["details"] = [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": reason}]
    return cls(code, body)


def check_gemini() -> None:
    print("Gemini path")
    request = httpx2.Request("POST", "https://generativelanguage.googleapis.com/")
    cases = [
        (
            "bad key (400 API_KEY_INVALID)",
            api_error(genai_errors.ClientError, 400, "INVALID_ARGUMENT", "API key not valid.", "API_KEY_INVALID"),
            "Gemini API key was rejected",
        ),
        ("other 400", api_error(genai_errors.ClientError, 400, "INVALID_ARGUMENT", "Bad field."), "status 400"),
        ("401", api_error(genai_errors.ClientError, 401, "UNAUTHENTICATED", "No key."), "key was rejected"),
        ("403", api_error(genai_errors.ClientError, 403, "PERMISSION_DENIED", "Denied."), "does not have access"),
        ("404", api_error(genai_errors.ClientError, 404, "NOT_FOUND", "No model."), "was not found"),
        (
            "429 rate limit",
            api_error(genai_errors.ClientError, 429, "RESOURCE_EXHAUSTED", "Quota exceeded."),
            "free-tier quota was reached",
        ),
        ("500", api_error(genai_errors.ServerError, 500, "INTERNAL", "Oops."), "status 500"),
        (
            "504 deadline",
            api_error(genai_errors.ServerError, 504, "DEADLINE_EXCEEDED", "Slow."),
            "did not respond within 30 seconds",
        ),
        ("timeout", httpx2.ReadTimeout("timed out", request=request), "did not respond within 30 seconds"),
        ("connection", httpx2.ConnectError("refused", request=request), "Could not connect to the Gemini API"),
        ("unexpected SDK error", ValueError(f"bad reply {FAKE_GEMINI_KEY}"), "request to the Gemini API failed"),
    ]
    for label, error, expected in cases:
        expect_error(f"gemini {label}", lambda: run_gemini(gemini_client(raises=error)), expected)

    replies = [
        ("blocked prompt", gemini_reply(block_reason="SAFETY"), "The model declined"),
        ("safety finish", gemini_reply(finish="SAFETY", parts=[]), "The model declined"),
        ("cut off", gemini_reply(finish="MAX_TOKENS", parts=[gt.Part(text="{")]), "cut off"),
        ("no candidates", gemini_reply(candidates=False), "empty reply"),
        ("no content", gemini_reply(parts=None), "empty reply"),
        ("only thoughts", gemini_reply(parts=[gt.Part(text="thinking...", thought=True)]), "empty reply"),
        ("blank text", gemini_reply(parts=[gt.Part(text="   ")]), "empty reply"),
        ("other finish", gemini_reply(finish="OTHER", parts=[gt.Part(text="x")]), "stopped before finishing (OTHER)"),
    ]
    for label, reply, expected in replies:
        expect_error(f"gemini {label}", lambda: run_gemini(gemini_client(reply=reply)), expected)

    reply = gemini_reply(parts=[gt.Part(text="hidden", thought=True), gt.Part(text=' {"a": '), gt.Part(text="1} ")])
    result, constructor = run_gemini(gemini_client(reply=reply))
    check("gemini text parts are joined, thoughts dropped", result == '{"a": 1}', repr(result))
    client_args = constructor.call_args.kwargs
    options = client_args["http_options"]
    check(
        "gemini client: 30 s timeout (in ms), one attempt",
        options.timeout == 30_000 and options.retry_options.attempts == 1,
    )
    sent = constructor.return_value.models.generate_content.call_args.kwargs
    config = sent["config"]
    check(
        "gemini request: model, system prompt, max tokens, low thinking, no temperature",
        sent["model"] == llm.GEMINI_MODEL
        and sent["contents"] == "user"
        and config.system_instruction == "system"
        and config.max_output_tokens == llm.DEFAULT_MAX_TOKENS
        and config.thinking_config.thinking_level == gt.ThinkingLevel.LOW
        and config.temperature is None,
        f"model {sent['model']}",
    )


def check_parser_record() -> None:
    print("Parser record")
    reply = '{"position_group": "Striker", "weights": {"npxg": 3}, "min_minutes": 600, "notes": "ok"}'
    for provider in llm.PROVIDERS:
        env = {"LLM_PROVIDER": provider}
        with mock.patch.object(parser, "call_llm", return_value=reply):
            result = with_settings(env, None, lambda: parser.parse_requirement("a striker"))
        check(
            f"{provider} answer is recorded",
            (result.source, result.provider, result.model) == ("llm", provider, llm.MODELS[provider]),
            f"{result.provider}, {result.model}",
        )
    with mock.patch.object(parser, "call_llm", side_effect=llm.LLMError("The model did not respond.")):
        result = with_settings({}, None, lambda: parser.parse_requirement("a striker"))
    check(
        "fallback result records no model",
        (result.source, result.provider, result.model) == ("offline", None, None)
        and result.notes[0] == "Used the offline keyword parser. Reason: The model did not respond.",
        result.notes[0],
    )
    with mock.patch.object(parser, "call_llm", return_value="not json"):
        result = with_settings({}, None, lambda: parser.parse_requirement("a striker"))
    check(
        "invalid-reply fallback note is provider-neutral",
        result.notes[0].startswith("Used the offline keyword parser. Reason: The model's reply could not be used"),
        result.notes[0],
    )
    result = with_settings({"LLM_PROVIDER": "openai"}, None, lambda: parser.parse_requirement("a striker"))
    check("bad provider setting falls back to offline", result.source == "offline", result.notes[0])
    sources = [PROJECT_ROOT / "src" / "llm.py", PROJECT_ROOT / "src" / "requirement_parser.py"]
    messages = [
        line
        for path in sources
        for line in path.read_text().splitlines()
        if "Claude" in line and ("LLMError(" in line or "reason =" in line or "notes" in line)
    ]
    check("no user-facing message names Claude", not messages, "; ".join(messages))


def main() -> None:
    handler = logging.StreamHandler(log_stream)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    check_settings()
    check_anthropic()
    check_gemini()
    check_gemini_retry()
    check_gemini_model_setting()
    check_parser_record()
    root.removeHandler(handler)
    logs = log_stream.getvalue()
    check("no key appears in any log output", not any(key in logs for key in FAKE_KEYS))
    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        sys.exit(1)
    print("All LLM client checks passed.")


if __name__ == "__main__":
    main()
