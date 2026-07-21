"""Auditable OpenAI-compatible LLM clients with deterministic disk caching."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from diskcache import Cache
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
)


DEFAULT_RETRY_DELAYS = (5, 10, 30)
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_AZURE_PROXY_URL = "http://127.0.0.1:8765/openai/"
DEFAULT_AZURE_API_VERSION = "2025-04-01-preview"


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass
class ModelSettings:
    temperature: float = 0.0
    max_tokens: int | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)

    def for_chat_completions(self) -> dict[str, Any]:
        settings: dict[str, Any] = {"temperature": self.temperature}
        if self.max_tokens is not None:
            settings["max_tokens"] = self.max_tokens
        if self.extra_body:
            settings["extra_body"] = copy.deepcopy(self.extra_body)
        return settings

    def for_responses(self) -> dict[str, Any]:
        settings: dict[str, Any] = {"temperature": self.temperature}
        if self.max_tokens is not None:
            settings["max_output_tokens"] = self.max_tokens
        if self.extra_body:
            extra_body = copy.deepcopy(self.extra_body)
            extra_body.pop("seed", None)
            if extra_body:
                settings["extra_body"] = extra_body
        return settings


@dataclass(frozen=True)
class ChatResult:
    reply: str
    request_hash: str
    cache_hit: bool
    latency_ms: float


class LLMRequestError(RuntimeError):
    """Provider failure with enough request identity for trajectory auditing."""

    def __init__(
        self,
        *,
        request_hash: str,
        latency_ms: float,
        original_exception: Exception,
    ):
        super().__init__(
            f"LLM request failed with "
            f"{type(original_exception).__name__}: {original_exception}"
        )
        self.request_hash = request_hash
        self.latency_ms = latency_ms
        self.original_exception = original_exception


def _message_payload(messages: Sequence[Message]) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def make_request_hash(
    *,
    client_type: str,
    model: str,
    api_type: str,
    base_url: str,
    api_version: str,
    effective_settings: Mapping[str, Any],
    messages: Sequence[Message],
    prompt_version: str,
) -> str:
    """Hash the complete effective request using canonical JSON."""

    payload = {
        "client_type": client_type,
        "model": model,
        "endpoint_identity": {
            "api_type": api_type,
            "base_url": base_url,
            "api_version": api_version,
        },
        "effective_settings": effective_settings,
        "messages": _message_payload(messages),
        "prompt_version": prompt_version,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class LLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: Sequence[Message],
        settings: ModelSettings | None = None,
        prompt_version: str = "",
    ) -> ChatResult:
        """Return one non-streaming model reply."""

    def close(self) -> None:
        """Release client resources."""


class MockLLMClient(LLMClient):
    """Return scripted replies without network or disk access."""

    def __init__(
        self,
        replies: Sequence[str] | Callable[[Sequence[Message]], str],
    ):
        if not callable(replies) and not replies:
            raise ValueError("MockLLMClient requires at least one reply")
        self._replies = replies
        self._index = 0

    def chat(
        self,
        messages: Sequence[Message],
        settings: ModelSettings | None = None,
        prompt_version: str = "",
    ) -> ChatResult:
        settings = settings or ModelSettings()
        if callable(self._replies):
            reply = self._replies(messages)
        else:
            reply = self._replies[self._index % len(self._replies)]
            self._index += 1
        effective_settings = settings.for_chat_completions()
        request_hash = make_request_hash(
            client_type=type(self).__name__,
            model="mock",
            api_type="mock",
            base_url="mock://local",
            api_version="",
            effective_settings=effective_settings,
            messages=messages,
            prompt_version=prompt_version,
        )
        return ChatResult(
            reply=reply,
            request_hash=request_hash,
            cache_hit=False,
            latency_ms=0.0,
        )


class _CachedOpenAIClient(LLMClient):
    def __init__(
        self,
        *,
        model: str,
        api_type: str,
        base_url: str,
        api_version: str,
        use_cache: bool,
        cache_path: str | os.PathLike[str] | None,
        retry_delays: Sequence[float],
        sleeper: Callable[[float], None],
        clock: Callable[[], float],
    ):
        self.model = model
        self.api_type = api_type
        self.base_url = base_url
        self.api_version = api_version
        self.retry_delays = tuple(retry_delays)
        self._sleeper = sleeper
        self._clock = clock
        self._cache: Cache | None = None
        self._provider_closed = False
        if use_cache:
            cache_dir = Path(
                cache_path
                or Path.home() / ".cache" / "world2skills" / "llm"
            ).expanduser()
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache = Cache(str(cache_dir))

    @abstractmethod
    def _effective_settings(self, settings: ModelSettings) -> dict[str, Any]:
        pass

    @abstractmethod
    def _raw_call(
        self,
        messages: list[dict[str, str]],
        effective_settings: Mapping[str, Any],
    ) -> str:
        pass

    def _is_retryable(self, error: Exception) -> bool:
        if isinstance(error, (APITimeoutError, APIConnectionError)):
            return True
        if isinstance(error, APIStatusError):
            status_code = error.status_code
            return (
                status_code in {408, 409, 429}
                or 500 <= status_code <= 599
            )
        return False

    def _call_with_retry(
        self,
        messages: list[dict[str, str]],
        effective_settings: Mapping[str, Any],
    ) -> str:
        for attempt in range(len(self.retry_delays) + 1):
            try:
                return self._raw_call(messages, effective_settings)
            except Exception as error:
                final_attempt = attempt == len(self.retry_delays)
                if final_attempt or not self._is_retryable(error):
                    raise
                self._sleeper(self.retry_delays[attempt])
        raise AssertionError("retry loop exhausted without returning or raising")

    def chat(
        self,
        messages: Sequence[Message],
        settings: ModelSettings | None = None,
        prompt_version: str = "",
    ) -> ChatResult:
        settings = settings or ModelSettings()
        message_list = list(messages)
        effective_settings = self._effective_settings(settings)
        request_hash = make_request_hash(
            client_type=type(self).__name__,
            model=self.model,
            api_type=self.api_type,
            base_url=self.base_url,
            api_version=self.api_version,
            effective_settings=effective_settings,
            messages=message_list,
            prompt_version=prompt_version,
        )
        if self._cache is not None and request_hash in self._cache:
            return ChatResult(
                reply=self._cache[request_hash],
                request_hash=request_hash,
                cache_hit=True,
                latency_ms=0.0,
            )

        started_at = self._clock()
        try:
            reply = self._call_with_retry(
                _message_payload(message_list),
                effective_settings,
            )
        except Exception as error:
            latency_ms = max(0.0, (self._clock() - started_at) * 1000.0)
            request_error = LLMRequestError(
                request_hash=request_hash,
                latency_ms=latency_ms,
                original_exception=error,
            )
            raise request_error from error

        latency_ms = max(0.0, (self._clock() - started_at) * 1000.0)
        if self._cache is not None:
            self._cache[request_hash] = reply
        return ChatResult(
            reply=reply,
            request_hash=request_hash,
            cache_hit=False,
            latency_ms=latency_ms,
        )

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()
            self._cache = None
        provider = getattr(self, "_client", None)
        provider_close = getattr(provider, "close", None)
        if not self._provider_closed and callable(provider_close):
            provider_close()
            self._provider_closed = True


class OpenAIClient(_CachedOpenAIClient):
    """OpenAI-compatible Chat Completions client."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        api_type: str = "openai",
        api_version: str = "",
        use_cache: bool = True,
        cache_path: str | os.PathLike[str] | None = None,
        timeout: float = 600.0,
        retry_delays: Sequence[float] = DEFAULT_RETRY_DELAYS,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
        client_factory: Callable[..., Any] = OpenAI,
    ):
        resolved_base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or DEFAULT_OPENAI_BASE_URL
        )
        super().__init__(
            model=model,
            api_type=api_type,
            base_url=resolved_base_url,
            api_version=api_version,
            use_cache=use_cache,
            cache_path=cache_path,
            retry_delays=retry_delays,
            sleeper=sleeper,
            clock=clock,
        )
        self._client = client_factory(
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "EMPTY",
            base_url=resolved_base_url,
            timeout=timeout,
            max_retries=0,
        )

    def _effective_settings(self, settings: ModelSettings) -> dict[str, Any]:
        return settings.for_chat_completions()

    def _raw_call(
        self,
        messages: list[dict[str, str]],
        effective_settings: Mapping[str, Any],
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            **effective_settings,
        )
        return response.choices[0].message.content or ""


class AzureResponsesClient(_CachedOpenAIClient):
    """Responses API client for the local Azure proxy used by GPT-5.4."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        api_type: str = "azure-responses",
        api_version: str | None = None,
        use_cache: bool = True,
        cache_path: str | os.PathLike[str] | None = None,
        timeout: float = 600.0,
        retry_delays: Sequence[float] = DEFAULT_RETRY_DELAYS,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
        client_factory: Callable[..., Any] = OpenAI,
    ):
        resolved_base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or DEFAULT_AZURE_PROXY_URL
        )
        resolved_api_version = (
            api_version
            or os.getenv("OPENAI_API_VERSION")
            or DEFAULT_AZURE_API_VERSION
        )
        super().__init__(
            model=model,
            api_type=api_type,
            base_url=resolved_base_url,
            api_version=resolved_api_version,
            use_cache=use_cache,
            cache_path=cache_path,
            retry_delays=retry_delays,
            sleeper=sleeper,
            clock=clock,
        )
        self._client = client_factory(
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "EMPTY",
            base_url=resolved_base_url,
            default_query={"api-version": resolved_api_version},
            timeout=timeout,
            max_retries=0,
        )

    def _effective_settings(self, settings: ModelSettings) -> dict[str, Any]:
        return settings.for_responses()

    def _raw_call(
        self,
        messages: list[dict[str, str]],
        effective_settings: Mapping[str, Any],
    ) -> str:
        response = self._client.responses.create(
            model=self.model,
            input=messages,
            **effective_settings,
        )
        return response.output_text or ""
