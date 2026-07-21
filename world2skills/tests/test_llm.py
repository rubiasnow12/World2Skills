from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from world2skills.runtime.llm import (
    AzureResponsesClient,
    ChatResult,
    LLMRequestError,
    Message,
    MockLLMClient,
    ModelSettings,
    OpenAIClient,
    make_request_hash,
)


MESSAGES = [
    Message(role="system", content="Choose one primitive."),
    Message(role="user", content="Ego lane: 1"),
]


class ScriptedCreate:
    def __init__(self, outcomes: list[Any]):
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeOpenAIProvider:
    def __init__(
        self,
        *,
        chat_outcomes: list[Any] | None = None,
        response_outcomes: list[Any] | None = None,
    ):
        self.chat_create = ScriptedCreate(chat_outcomes or [])
        self.responses_create = ScriptedCreate(response_outcomes or [])
        self.close_calls = 0
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.chat_create)
        )
        self.responses = SimpleNamespace(create=self.responses_create)

    def close(self) -> None:
        self.close_calls += 1


class RecordingFactory:
    def __init__(self, provider: FakeOpenAIProvider):
        self.provider = provider
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> FakeOpenAIProvider:
        self.calls.append(kwargs)
        return self.provider


def _chat_response(reply: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]
    )


def _responses_response(reply: str | None) -> SimpleNamespace:
    return SimpleNamespace(output_text=reply)


def _status_error(status_code: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://example.test/v1/responses")
    response = httpx.Response(status_code, request=request)
    return openai.APIStatusError(
        f"status {status_code}",
        response=response,
        body=None,
    )


def _timeout_error() -> openai.APITimeoutError:
    request = httpx.Request("POST", "https://example.test/v1/responses")
    return openai.APITimeoutError(request=request)


def _connection_error() -> openai.APIConnectionError:
    request = httpx.Request("POST", "https://example.test/v1/responses")
    return openai.APIConnectionError(request=request)


def _hash_kwargs() -> dict[str, Any]:
    return {
        "client_type": "OpenAIClient",
        "model": "model-a",
        "api_type": "openai",
        "base_url": "https://example.test/v1",
        "api_version": "2026-01-01",
        "effective_settings": {
            "temperature": 0.2,
            "max_tokens": 64,
            "extra_body": {"guided": {"choice": ["a", "b"]}},
        },
        "messages": MESSAGES,
        "prompt_version": "prompt-v1",
    }


def test_request_hash_is_canonical_and_deterministic():
    kwargs = _hash_kwargs()
    reordered = {
        **kwargs,
        "effective_settings": {
            "extra_body": {"guided": {"choice": ["a", "b"]}},
            "max_tokens": 64,
            "temperature": 0.2,
        },
    }

    assert make_request_hash(**kwargs) == make_request_hash(**reordered)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("client_type", "AzureResponsesClient"),
        ("model", "model-b"),
        ("api_type", "azure"),
        ("base_url", "https://other.test/v1"),
        ("api_version", "2026-02-02"),
        (
            "effective_settings",
            {
                "temperature": 0.7,
                "max_tokens": 64,
                "extra_body": {"guided": {"choice": ["a", "b"]}},
            },
        ),
        ("messages", [Message(role="user", content="changed")]),
        ("prompt_version", "prompt-v2"),
    ],
)
def test_request_hash_covers_every_request_identity_field(
    field_name: str,
    replacement: Any,
):
    kwargs = _hash_kwargs()

    assert make_request_hash(**kwargs) != make_request_hash(
        **{**kwargs, field_name: replacement}
    )


def test_mock_client_returns_scripted_results_and_auditable_hashes():
    client = MockLLMClient(["first", "second"])

    first = client.chat(MESSAGES, prompt_version="v1")
    second = client.chat(MESSAGES, prompt_version="v1")

    assert isinstance(first, ChatResult)
    assert (first.reply, second.reply) == ("first", "second")
    assert first.request_hash == second.request_hash
    assert first.cache_hit is False
    assert first.latency_ms == 0.0


def test_openai_chat_completions_sends_effective_settings_and_disables_sdk_retry(
    tmp_path,
):
    provider = FakeOpenAIProvider(chat_outcomes=[_chat_response("ok")])
    factory = RecordingFactory(provider)
    client = OpenAIClient(
        model="chat-model",
        api_key="test-key",
        base_url="https://chat.test/v1",
        api_type="local-openai",
        api_version="compat-v1",
        use_cache=False,
        retry_delays=(),
        client_factory=factory,
    )
    settings = ModelSettings(
        temperature=0.25,
        max_tokens=77,
        extra_body={"guided_json": {"type": "object"}},
    )

    result = client.chat(MESSAGES, settings, prompt_version="pv")

    assert result.reply == "ok"
    assert factory.calls == [
        {
            "api_key": "test-key",
            "base_url": "https://chat.test/v1",
            "timeout": 600.0,
            "max_retries": 0,
        }
    ]
    assert provider.chat_create.calls == [
        {
            "model": "chat-model",
            "messages": [
                {"role": "system", "content": "Choose one primitive."},
                {"role": "user", "content": "Ego lane: 1"},
            ],
            "temperature": 0.25,
            "max_tokens": 77,
            "extra_body": {"guided_json": {"type": "object"}},
        }
    ]


def test_azure_responses_sends_effective_settings_and_endpoint_identity():
    provider = FakeOpenAIProvider(
        response_outcomes=[_responses_response("accelerate")]
    )
    factory = RecordingFactory(provider)
    client = AzureResponsesClient(
        model="gpt-5.4",
        api_key="test-key",
        base_url="http://127.0.0.1:8765/openai/",
        api_type="azure-responses",
        api_version="2026-06-01-preview",
        use_cache=False,
        retry_delays=(),
        client_factory=factory,
    )
    settings = ModelSettings(
        temperature=0.4,
        max_tokens=91,
        extra_body={"reasoning_effort": "low"},
    )

    result = client.chat(MESSAGES, settings, prompt_version="pv")

    assert result.reply == "accelerate"
    assert factory.calls == [
        {
            "api_key": "test-key",
            "base_url": "http://127.0.0.1:8765/openai/",
            "default_query": {"api-version": "2026-06-01-preview"},
            "timeout": 600.0,
            "max_retries": 0,
        }
    ]
    assert provider.responses_create.calls == [
        {
            "model": "gpt-5.4",
            "input": [
                {"role": "system", "content": "Choose one primitive."},
                {"role": "user", "content": "Ego lane: 1"},
            ],
            "temperature": 0.4,
            "max_output_tokens": 91,
            "extra_body": {"reasoning_effort": "low"},
        }
    ]
    expected_hash = make_request_hash(
        client_type="AzureResponsesClient",
        model="gpt-5.4",
        api_type="azure-responses",
        base_url="http://127.0.0.1:8765/openai/",
        api_version="2026-06-01-preview",
        effective_settings={
            "temperature": 0.4,
            "max_output_tokens": 91,
            "extra_body": {"reasoning_effort": "low"},
        },
        messages=MESSAGES,
        prompt_version="pv",
    )
    assert result.request_hash == expected_hash


def test_azure_responses_drops_seed_before_hashing_and_provider_call():
    provider = FakeOpenAIProvider(
        response_outcomes=[_responses_response("maintain-speed")]
    )
    client = AzureResponsesClient(
        model="gpt-5.4",
        api_key="test-key",
        base_url="http://127.0.0.1:8765/openai/",
        api_version="2026-06-01-preview",
        use_cache=False,
        retry_delays=(),
        client_factory=RecordingFactory(provider),
    )
    settings = ModelSettings(
        temperature=0.1,
        max_tokens=32,
        extra_body={"seed": 1234, "reasoning_effort": "low"},
    )

    result = client.chat(MESSAGES, settings, prompt_version="pv")

    effective_settings = {
        "temperature": 0.1,
        "max_output_tokens": 32,
        "extra_body": {"reasoning_effort": "low"},
    }
    assert client._effective_settings(settings) == effective_settings
    assert provider.responses_create.calls == [
        {
            "model": "gpt-5.4",
            "input": [
                {"role": "system", "content": "Choose one primitive."},
                {"role": "user", "content": "Ego lane: 1"},
            ],
            **effective_settings,
        }
    ]
    assert result.request_hash == make_request_hash(
        client_type="AzureResponsesClient",
        model="gpt-5.4",
        api_type="azure-responses",
        base_url="http://127.0.0.1:8765/openai/",
        api_version="2026-06-01-preview",
        effective_settings=effective_settings,
        messages=MESSAGES,
        prompt_version="pv",
    )


@pytest.mark.parametrize(
    "error_factory",
    [
        _timeout_error,
        _connection_error,
        lambda: _status_error(408),
        lambda: _status_error(409),
        lambda: _status_error(429),
        lambda: _status_error(500),
        lambda: _status_error(503),
    ],
)
def test_retries_only_transient_provider_errors(error_factory):
    provider = FakeOpenAIProvider(
        chat_outcomes=[error_factory(), _chat_response("recovered")]
    )
    sleeps: list[float] = []
    client = OpenAIClient(
        model="m",
        use_cache=False,
        retry_delays=(2.5,),
        sleeper=sleeps.append,
        client_factory=RecordingFactory(provider),
    )

    result = client.chat(MESSAGES, prompt_version="pv")

    assert result.reply == "recovered"
    assert len(provider.chat_create.calls) == 2
    assert sleeps == [2.5]


def test_retry_schedule_is_initial_attempt_plus_one_attempt_per_delay():
    provider = FakeOpenAIProvider(
        chat_outcomes=[
            _timeout_error(),
            _connection_error(),
            _chat_response("third attempt"),
        ]
    )
    sleeps: list[float] = []
    client = OpenAIClient(
        model="m",
        use_cache=False,
        retry_delays=(5, 10, 30),
        sleeper=sleeps.append,
        client_factory=RecordingFactory(provider),
    )

    result = client.chat(MESSAGES, prompt_version="pv")

    assert result.reply == "third attempt"
    assert len(provider.chat_create.calls) == 3
    assert sleeps == [5, 10]
    assert client.retry_delays == (5, 10, 30)


def test_non_retriable_error_fails_immediately_without_sleeping():
    original = _status_error(400)
    provider = FakeOpenAIProvider(chat_outcomes=[original])
    sleeps: list[float] = []
    ticks = iter([1.0, 1.125])
    client = OpenAIClient(
        model="m",
        use_cache=False,
        retry_delays=(5, 10, 30),
        sleeper=sleeps.append,
        clock=lambda: next(ticks),
        client_factory=RecordingFactory(provider),
    )

    with pytest.raises(LLMRequestError) as caught:
        client.chat(MESSAGES, prompt_version="pv")

    assert len(provider.chat_create.calls) == 1
    assert sleeps == []
    assert caught.value.original_exception is original
    assert caught.value.request_hash
    assert caught.value.latency_ms == pytest.approx(125.0)
    assert caught.value.__cause__ is original


def test_permanent_transient_failure_never_sleeps_after_final_attempt():
    errors = [_timeout_error(), _timeout_error(), _timeout_error()]
    provider = FakeOpenAIProvider(chat_outcomes=errors)
    sleeps: list[float] = []
    client = OpenAIClient(
        model="m",
        use_cache=False,
        retry_delays=(5, 10),
        sleeper=sleeps.append,
        client_factory=RecordingFactory(provider),
    )

    with pytest.raises(LLMRequestError) as caught:
        client.chat(MESSAGES, prompt_version="pv")

    assert len(provider.chat_create.calls) == 3
    assert sleeps == [5, 10]
    assert caught.value.original_exception is errors[-1]
    assert caught.value.request_hash
    assert caught.value.latency_ms >= 0.0


def test_identical_disk_cached_call_invokes_provider_once(tmp_path):
    provider = FakeOpenAIProvider(chat_outcomes=[_chat_response("cached")])
    client = OpenAIClient(
        model="m",
        use_cache=True,
        cache_path=tmp_path / "cache",
        retry_delays=(),
        client_factory=RecordingFactory(provider),
    )

    first = client.chat(MESSAGES, prompt_version="pv")
    second = client.chat(MESSAGES, prompt_version="pv")
    client.close()

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.reply == "cached"
    assert first.request_hash == second.request_hash
    assert len(provider.chat_create.calls) == 1


def test_prompt_messages_settings_and_version_each_create_cache_miss(tmp_path):
    provider = FakeOpenAIProvider(
        chat_outcomes=[
            _chat_response("base"),
            _chat_response("message"),
            _chat_response("settings"),
            _chat_response("version"),
        ]
    )
    client = OpenAIClient(
        model="m",
        use_cache=True,
        cache_path=tmp_path / "cache",
        retry_delays=(),
        client_factory=RecordingFactory(provider),
    )

    client.chat(MESSAGES, prompt_version="v1")
    client.chat([Message(role="user", content="different")], prompt_version="v1")
    client.chat(
        MESSAGES,
        ModelSettings(temperature=0.5),
        prompt_version="v1",
    )
    client.chat(MESSAGES, prompt_version="v2")
    client.close()

    assert len(provider.chat_create.calls) == 4


def test_successful_empty_reply_is_cached(tmp_path):
    provider = FakeOpenAIProvider(chat_outcomes=[_chat_response(None)])
    client = OpenAIClient(
        model="m",
        use_cache=True,
        cache_path=tmp_path / "cache",
        retry_delays=(),
        client_factory=RecordingFactory(provider),
    )

    first = client.chat(MESSAGES, prompt_version="pv")
    second = client.chat(MESSAGES, prompt_version="pv")
    client.close()

    assert first.reply == second.reply == ""
    assert second.cache_hit is True
    assert len(provider.chat_create.calls) == 1


def test_failed_request_is_not_cached(tmp_path):
    provider = FakeOpenAIProvider(
        chat_outcomes=[_timeout_error(), _chat_response("recovered")]
    )
    client = OpenAIClient(
        model="m",
        use_cache=True,
        cache_path=tmp_path / "cache",
        retry_delays=(),
        client_factory=RecordingFactory(provider),
    )

    with pytest.raises(LLMRequestError):
        client.chat(MESSAGES, prompt_version="pv")
    recovered = client.chat(MESSAGES, prompt_version="pv")
    cached = client.chat(MESSAGES, prompt_version="pv")
    client.close()

    assert recovered.reply == "recovered"
    assert recovered.cache_hit is False
    assert cached.cache_hit is True
    assert len(provider.chat_create.calls) == 2


def test_close_is_idempotent(tmp_path):
    provider = FakeOpenAIProvider(chat_outcomes=[])
    client = OpenAIClient(
        model="m",
        use_cache=True,
        cache_path=tmp_path / "cache",
        retry_delays=(),
        client_factory=RecordingFactory(provider),
    )

    client.close()
    client.close()

    assert provider.close_calls == 1
