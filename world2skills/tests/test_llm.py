from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time
from types import SimpleNamespace
from typing import Any

from diskcache import Lock as DiskCacheLock
import httpx
import openai
import pytest

import world2skills.runtime.llm as llm_module
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


class BlockingCreate:
    def __init__(self, outcome: Any):
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        self.started.set()
        if not self.release.wait(timeout=2):
            raise TimeoutError("test did not release provider call")
        return self.outcome


class WaitForConcurrentCreate:
    def __init__(self, outcome: Any):
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []
        self._calls_lock = threading.Lock()
        self._second_call_started = threading.Event()

    def __call__(self, **kwargs: Any) -> Any:
        with self._calls_lock:
            self.calls.append(kwargs)
            call_count = len(self.calls)
            if call_count == 2:
                self._second_call_started.set()
        if call_count == 1:
            self._second_call_started.wait(timeout=0.5)
        return self.outcome


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


def test_request_hash_treats_equivalent_base_urls_as_same_endpoint():
    kwargs = _hash_kwargs()

    without_slash = make_request_hash(**kwargs)
    with_extra_slashes = make_request_hash(
        **{**kwargs, "base_url": "https://example.test/v1///"}
    )

    assert without_slash == with_extra_slashes


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
            "base_url": "https://chat.test/v1/",
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


def test_provider_receives_canonical_base_url_and_endpoint_fields_are_read_only():
    provider = FakeOpenAIProvider(chat_outcomes=[_chat_response("ok")])
    factory = RecordingFactory(provider)
    client = OpenAIClient(
        model="m",
        base_url="https://chat.test/v1///",
        api_type="local-openai",
        api_version="compat-v1",
        use_cache=False,
        retry_delays=(),
        client_factory=factory,
    )

    result = client.chat(MESSAGES, prompt_version="pv")

    assert factory.calls[0]["base_url"] == "https://chat.test/v1/"
    assert client.base_url == "https://chat.test/v1/"
    assert client.api_type == "local-openai"
    assert client.api_version == "compat-v1"
    assert result.request_hash == make_request_hash(
        client_type="OpenAIClient",
        model="m",
        api_type="local-openai",
        base_url="https://chat.test/v1/",
        api_version="compat-v1",
        effective_settings={"temperature": 0.0},
        messages=MESSAGES,
        prompt_version="pv",
    )
    assert {"base_url", "api_type", "api_version"}.isdisjoint(vars(client))
    for field_name in ("base_url", "api_type", "api_version"):
        with pytest.raises(AttributeError):
            setattr(client, field_name, "changed")
    client.close()


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


def test_chat_after_close_raises_clear_runtime_error(tmp_path):
    provider = FakeOpenAIProvider(chat_outcomes=[_chat_response("unexpected")])
    client = OpenAIClient(
        model="m",
        use_cache=True,
        cache_path=tmp_path / "cache",
        retry_delays=(),
        client_factory=RecordingFactory(provider),
    )

    client.close()

    assert client.closed is True
    with pytest.raises(RuntimeError, match="LLM client is closed"):
        client.chat(MESSAGES, prompt_version="pv")
    assert provider.chat_create.calls == []


def test_close_waits_for_in_flight_chat_before_closing_provider():
    blocking_create = BlockingCreate(_chat_response("done"))
    provider = FakeOpenAIProvider(chat_outcomes=[])
    provider.chat_create = blocking_create
    provider.chat.completions.create = blocking_create
    client = OpenAIClient(
        model="m",
        use_cache=False,
        retry_delays=(),
        client_factory=RecordingFactory(provider),
    )
    close_started = threading.Event()
    close_finished = threading.Event()

    def close_client() -> None:
        close_started.set()
        client.close()
        close_finished.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        chat_future = pool.submit(client.chat, MESSAGES, None, "pv")
        assert blocking_create.started.wait(timeout=1)
        close_future = pool.submit(close_client)
        assert close_started.wait(timeout=1)
        try:
            assert not close_finished.wait(timeout=0.1)
        finally:
            blocking_create.release.set()
        assert chat_future.result(timeout=1).reply == "done"
        close_future.result(timeout=1)

    assert close_finished.is_set()
    assert provider.close_calls == 1
    assert client.closed is True


def test_concurrent_identical_cache_misses_invoke_provider_once(tmp_path):
    concurrent_create = WaitForConcurrentCreate(_chat_response("shared"))
    provider = FakeOpenAIProvider(chat_outcomes=[])
    provider.chat_create = concurrent_create
    provider.chat.completions.create = concurrent_create
    factory = RecordingFactory(provider)
    clients = [
        OpenAIClient(
            model="m",
            base_url="https://chat.test/v1",
            use_cache=True,
            cache_path=tmp_path / "cache",
            retry_delays=(),
            client_factory=factory,
        ),
        OpenAIClient(
            model="m",
            base_url="https://chat.test/v1/",
            use_cache=True,
            cache_path=tmp_path / "cache",
            retry_delays=(),
            client_factory=factory,
        ),
    ]
    start = threading.Barrier(3)

    def invoke(client: OpenAIClient) -> ChatResult:
        start.wait(timeout=1)
        return client.chat(MESSAGES, prompt_version="pv")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(invoke, client) for client in clients]
        start.wait(timeout=1)
        results = [future.result(timeout=2) for future in futures]

    for client in clients:
        client.close()

    assert [result.reply for result in results] == ["shared", "shared"]
    assert sorted(result.cache_hit for result in results) == [False, True]
    assert results[0].request_hash == results[1].request_hash
    assert len(concurrent_create.calls) == 1


def test_lock_lease_covers_timeout_retry_and_safety_budget():
    provider = FakeOpenAIProvider(chat_outcomes=[])
    client = OpenAIClient(
        model="m",
        use_cache=False,
        timeout=2.0,
        retry_delays=(1.0, 4.0),
        lock_safety_margin=3.0,
        client_factory=RecordingFactory(provider),
    )

    assert client.request_timeout == 2.0
    assert client.lock_safety_margin == 3.0
    assert client.lock_lease_seconds == pytest.approx(
        3 * 2.0 + 1.0 + 4.0 + 3.0
    )
    client.close()


def test_expired_stale_request_lock_recovers_with_finite_lease(
    tmp_path,
    monkeypatch,
):
    provider = FakeOpenAIProvider(chat_outcomes=[_chat_response("recovered")])
    client = OpenAIClient(
        model="m",
        base_url="https://chat.test/v1",
        use_cache=True,
        cache_path=tmp_path / "cache",
        timeout=0.02,
        retry_delays=(),
        lock_safety_margin=0.03,
        client_factory=RecordingFactory(provider),
    )
    request_hash = make_request_hash(
        client_type="OpenAIClient",
        model="m",
        api_type="openai",
        base_url="https://chat.test/v1/",
        api_version="",
        effective_settings={"temperature": 0.0},
        messages=MESSAGES,
        prompt_version="pv",
    )
    lock_key = ("world2skills.llm.request", request_hash)
    assert client._cache is not None
    stale_lock = DiskCacheLock(
        client._cache,
        lock_key,
        expire=client.lock_lease_seconds,
    )
    stale_lock.acquire()

    observed_expirations: list[float | None] = []
    real_lock = llm_module.Lock

    def recording_lock(cache, key, expire=None, tag=None):
        observed_expirations.append(expire)
        return real_lock(cache, key, expire=expire, tag=tag)

    monkeypatch.setattr(llm_module, "Lock", recording_lock)
    started_at = time.perf_counter()

    result = client.chat(MESSAGES, prompt_version="pv")

    elapsed = time.perf_counter() - started_at
    client.close()
    assert result.reply == "recovered"
    assert elapsed < 1.0
    assert observed_expirations == [client.lock_lease_seconds]
    assert len(provider.chat_create.calls) == 1


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
