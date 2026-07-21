"""Strict, auditable LLM executor for discrete skill primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import re
import time
from typing import Any

from .llm import (
    LLMClient,
    LLMRequestError,
    Message,
    ModelSettings,
    make_request_hash,
)
from .types import DecisionResult, Grounding, ObservationContext, SkillCard


_FENCED_JSON = re.compile(
    r"\A```json[ \t]*(?:\r?\n)?(?P<body>\{.*\})(?:\r?\n)?```\Z",
    re.DOTALL,
)
_UNAVAILABLE_CLIENT_IDENTITY = "<unavailable>"


class NoAvailablePrimitiveError(RuntimeError):
    """Raised when the environment offers no primitive supported by the skill."""


class _JSONObjectPairs(list):
    """Marker used to distinguish JSON objects from JSON arrays."""


class LLMSkillExecutor:
    """Choose one skill primitive and map it to a discrete backend action."""

    def __init__(
        self,
        skill_card: SkillCard,
        grounding: Grounding,
        llm_client: LLMClient,
        prompt_version: str,
        name_to_index: Mapping[str, int],
    ):
        self.skill_card = deepcopy(skill_card)
        self.grounding = deepcopy(grounding)
        self.llm = llm_client
        self.prompt_version = prompt_version
        self._primitives = tuple(self.skill_card.primitives)
        self._primitive_map = dict(self.grounding.primitive_map)
        self._name_to_index = dict(name_to_index)
        self._validate_action_contract()

    def _validate_action_contract(self) -> None:
        if len(set(self._primitives)) != len(self._primitives):
            raise ValueError("skill primitives must be unique")

        missing_primitives = [
            primitive
            for primitive in self._primitives
            if primitive not in self._primitive_map
        ]
        if missing_primitives:
            raise ValueError(
                "primitive_map is missing skill primitives: "
                + ", ".join(missing_primitives)
            )

        backend_actions = [
            self._primitive_map[primitive] for primitive in self._primitives
        ]
        if any(not isinstance(name, str) or not name for name in backend_actions):
            raise ValueError("primitive_map backend actions must be non-empty strings")

        if any(not isinstance(name, str) or not name for name in self._name_to_index):
            raise ValueError("backend action names must be non-empty strings")
        indices = list(self._name_to_index.values())
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in indices
        ):
            raise ValueError("action indices must be non-negative integers")
        if len(set(indices)) != len(indices):
            raise ValueError("duplicate action indices are not allowed")

        missing_backend_actions = [
            name
            for name in dict.fromkeys(backend_actions)
            if name not in self._name_to_index
        ]
        if missing_backend_actions:
            raise ValueError(
                "backend actions missing from name_to_index: "
                + ", ".join(missing_backend_actions)
            )

    def _filter_available(
        self,
        available_primitives: Sequence[str] | None,
    ) -> list[str]:
        supplied = {
            primitive
            for primitive in available_primitives or ()
            if isinstance(primitive, str)
        }
        return [primitive for primitive in self._primitives if primitive in supplied]

    def _require_available(
        self,
        available_primitives: Sequence[str] | None,
    ) -> list[str]:
        allowed = self._filter_available(available_primitives)
        if not allowed:
            raise NoAvailablePrimitiveError("no skill primitive is currently available")
        return allowed

    def build_messages(
        self,
        obs_text: str,
        context: ObservationContext | Sequence[str],
    ) -> list[Message]:
        """Build a complete skill prompt for the current allowed action set."""

        available_primitives = (
            context.available_primitives
            if isinstance(context, ObservationContext)
            else context
        )
        allowed = self._require_available(available_primitives)
        card = self.skill_card
        safety = "\n".join(f"- {constraint}" for constraint in card.safety_constraints)
        system = (
            "You are an autonomous-driving policy. Choose exactly one abstract "
            "driving primitive for the current step.\n"
            "Primitive semantics: maintain-speed holds speed; accelerate speeds "
            "up; decelerate slows down; change-lane-left and "
            "change-lane-right move one adjacent lane.\n"
            "Safety constraints:\n"
            f"{safety}\n"
            'Respond with ONLY one JSON object: {"primitive": '
            '"<allowed primitive>"} and nothing else.'
        )
        user = "\n\n".join(
            [
                f"# Skill: {card.name}\n{card.description}",
                f"# Skill instructions\n{card.skill_md_body.rstrip()}",
                "# Parameters\n" + _json_block(card.parameters),
                "# Preconditions\n" + _list_block(card.preconditions),
                "# Execution graph\n" + _json_block(card.execution),
                "# Effects\n" + _list_block(card.effects),
                "# Success criteria\n" + _list_block(card.success_criteria),
                "# Failure criteria\n" + _list_block(card.failure_criteria),
                "# Safety constraints\n" + _list_block(card.safety_constraints),
                "# Failure modes\n" + _list_block(card.failure_modes),
                f"# Termination\n{card.termination}",
                f"# Current observation\n{obs_text}",
                "# Allowed primitives this step\n"
                + json.dumps(allowed, ensure_ascii=False),
                'Return only {"primitive": "<allowed primitive>"}.',
            ]
        )
        return [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]

    def _parse_primitive(self, reply: Any) -> tuple[str | None, str | None]:
        if not isinstance(reply, str):
            return None, "response must be a string"
        text = reply.strip()
        fenced = _FENCED_JSON.fullmatch(text)
        payload = fenced.group("body") if fenced else text
        try:
            parsed = json.loads(payload, object_pairs_hook=_JSONObjectPairs)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None, "response is not an accepted JSON object"

        if not isinstance(parsed, _JSONObjectPairs):
            return None, "response must be a JSON object"
        if len(parsed) != 1 or parsed[0][0] != "primitive":
            return None, "response must contain exactly one primitive key"

        primitive = parsed[0][1]
        if not isinstance(primitive, str):
            return None, "primitive must be a string"
        if primitive not in self._primitives:
            return None, f"unknown skill primitive: {primitive}"
        return primitive, None

    @staticmethod
    def _fallback(allowed: list[str]) -> str:
        if "maintain-speed" in allowed:
            return "maintain-speed"
        return allowed[0]

    def decide(
        self,
        obs_text: str,
        context: ObservationContext | None,
        available_primitives: Sequence[str],
    ) -> DecisionResult:
        """Call the LLM, validate its choice, and return an executable action."""

        allowed = self._filter_available(available_primitives)
        if context is not None:
            context_allowed = self._filter_available(context.available_primitives)
            if context_allowed != allowed:
                raise ValueError(
                    "context available_primitives do not match explicit availability"
                )
        if not allowed:
            raise NoAvailablePrimitiveError("no skill primitive is currently available")
        messages = self.build_messages(obs_text, allowed)
        settings = ModelSettings()
        status = "ok"
        fallback_reason: str | None = None
        raw_response = ""
        request_hash = make_request_hash(
            client_type=type(self.llm).__name__,
            model=_client_identity(self.llm, "model"),
            api_type=_client_identity(self.llm, "api_type"),
            base_url=_client_identity(self.llm, "base_url"),
            api_version=_client_identity(self.llm, "api_version"),
            effective_settings=settings.for_chat_completions(),
            messages=messages,
            prompt_version=self.prompt_version,
        )
        cache_hit = False
        latency_ms = 0.0
        started = time.perf_counter()

        try:
            result = self.llm.chat(
                messages,
                settings,
                self.prompt_version,
            )
            raw_response = _normalize_reply(result.reply)
            if _is_nonempty_string(result.request_hash):
                request_hash = result.request_hash
            cache_hit = result.cache_hit
            latency_ms = result.latency_ms
        except LLMRequestError as error:
            status = "llm_error_fallback"
            if _is_nonempty_string(error.request_hash):
                request_hash = error.request_hash
            latency_ms = error.latency_ms
            fallback_reason = _describe_exception(error.original_exception)
        except Exception as error:
            status = "llm_error_fallback"
            latency_ms = (time.perf_counter() - started) * 1000
            fallback_reason = _describe_exception(error)

        if status == "ok":
            primitive, parse_error = self._parse_primitive(raw_response)
            if primitive is None:
                status = "parse_fallback"
                fallback_reason = parse_error
                primitive = self._fallback(allowed)
            elif primitive not in allowed:
                status = "unavailable_fallback"
                fallback_reason = f"{primitive} is not available in the current state"
                primitive = self._fallback(allowed)
        else:
            primitive = self._fallback(allowed)

        backend_action = self._primitive_map[primitive]
        return DecisionResult(
            primitive=primitive,
            backend_action=backend_action,
            action_index=self._name_to_index[backend_action],
            request_hash=request_hash,
            raw_response=raw_response,
            cache_hit=cache_hit,
            latency_ms=latency_ms,
            decision_status=status,
            fallback_reason=fallback_reason,
            available_primitives=allowed.copy(),
        )

    def available_primitives(self, env: Any) -> list[str]:
        """Return environment-available primitives in stable skill order."""

        available_indices = set(env.unwrapped.action_type.get_available_actions())
        return [
            primitive
            for primitive in self._primitives
            if self._name_to_index[self._primitive_map[primitive]] in available_indices
        ]


def _json_block(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )


def _list_block(values: Sequence[str]) -> str:
    return "\n".join(f"- {value}" for value in values)


def _describe_exception(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _client_identity(client: Any, attribute: str) -> str:
    try:
        value = getattr(client, attribute, "")
        return "" if value is None else str(value)
    except Exception:
        return _UNAVAILABLE_CLIENT_IDENTITY


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _normalize_reply(reply: Any) -> str:
    if isinstance(reply, str):
        return reply
    try:
        return json.dumps(
            reply,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return repr(reply)
