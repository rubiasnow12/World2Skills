from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from world2skills.runtime.executor import (
    LLMSkillExecutor,
    NoAvailablePrimitiveError,
)
from world2skills.runtime.llm import (
    LLMRequestError,
    MockLLMClient,
    ModelSettings,
    make_request_hash,
)
from world2skills.runtime.skill_loader import load_skill, select_grounding
from world2skills.runtime.types import ObservationContext


NAME_TO_INDEX = {
    "LANE_LEFT": 0,
    "IDLE": 1,
    "LANE_RIGHT": 2,
    "FASTER": 3,
    "SLOWER": 4,
}
ALL_PRIMITIVES = [
    "maintain-speed",
    "accelerate",
    "decelerate",
    "change-lane-left",
    "change-lane-right",
]


def _skill_and_grounding():
    card = load_skill("lane-change-overtake")
    grounding = select_grounding(card, "highway-env")
    return card, grounding


def _executor(
    reply: Any = '{"primitive": "maintain-speed"}',
    *,
    client: Any | None = None,
    name_to_index: dict[str, int] | None = None,
) -> LLMSkillExecutor:
    card, grounding = _skill_and_grounding()
    return LLMSkillExecutor(
        card,
        grounding,
        client or MockLLMClient([reply]),
        "lco-v1",
        NAME_TO_INDEX if name_to_index is None else name_to_index,
    )


def _observation_context(
    available_primitives: list[str],
) -> ObservationContext:
    return ObservationContext(
        available_primitives=available_primitives,
        ego_lane=1,
        prev_primitive=None,
        target_ahead=True,
        target_gap_m=20.0,
        target_rel_speed_mps=-5.0,
        left_lane_exists=True,
        right_lane_exists=True,
        left_front_gap_m=30.0,
        left_rear_gap_m=25.0,
        left_rear_closing_speed_mps=0.0,
        right_front_gap_m=30.0,
        right_rear_gap_m=25.0,
        right_rear_closing_speed_mps=0.0,
    )


def test_valid_json_primitive_maps_to_backend_index():
    result = _executor('{"primitive": "accelerate"}').decide(
        "observation",
        None,
        ALL_PRIMITIVES,
    )

    assert result.primitive == "accelerate"
    assert result.backend_action == "FASTER"
    assert result.action_index == 3
    assert result.decision_status == "ok"
    assert result.fallback_reason is None
    assert result.request_hash
    assert result.raw_response == '{"primitive": "accelerate"}'


def test_exact_json_fence_is_accepted():
    result = _executor(
        '```json\n{"primitive": "decelerate"}\n```'
    ).decide("observation", None, ALL_PRIMITIVES)

    assert result.primitive == "decelerate"
    assert result.action_index == 4
    assert result.decision_status == "ok"


@pytest.mark.parametrize(
    "reply",
    [
        "do not accelerate, maintain-speed please",
        '{"primitive": "teleport"}',
        '{"primitive": "accelerate", "reason": "clear"}',
        'prefix ```json\n{"primitive": "accelerate"}\n```',
        '```json\n{"primitive": "accelerate"}\n``` suffix',
        '```\n{"primitive": "accelerate"}\n```',
        '{"primitive": 3}',
        '["accelerate"]',
    ],
)
def test_malformed_unknown_or_non_exact_reply_uses_parse_fallback(reply: str):
    result = _executor(reply).decide(
        "observation",
        None,
        ALL_PRIMITIVES,
    )

    assert result.decision_status == "parse_fallback"
    assert result.primitive == "maintain-speed"
    assert result.backend_action == "IDLE"
    assert result.action_index == 1
    assert result.fallback_reason


@pytest.mark.parametrize(
    ("reply", "expected_raw_response"),
    [
        (None, "null"),
        ({"z": 1, "a": 2}, '{"a":2,"z":1}'),
    ],
)
def test_non_string_reply_is_normalized_before_parse_fallback(
    reply: Any,
    expected_raw_response: str,
):
    result = _executor(reply).decide(
        "observation",
        None,
        ALL_PRIMITIVES,
    )

    assert result.decision_status == "parse_fallback"
    assert result.primitive == "maintain-speed"
    assert result.raw_response == expected_raw_response
    assert isinstance(result.raw_response, str)
    assert result.fallback_reason


def test_valid_but_unavailable_primitive_uses_unavailable_fallback():
    result = _executor('{"primitive": "change-lane-left"}').decide(
        "observation",
        None,
        ["decelerate", "accelerate", "maintain-speed"],
    )

    assert result.decision_status == "unavailable_fallback"
    assert result.primitive == "maintain-speed"
    assert "change-lane-left" in result.fallback_reason


def test_fallback_is_stable_skill_order_when_maintain_speed_is_unavailable():
    first = _executor("invalid").decide(
        "observation",
        None,
        ["decelerate", "accelerate"],
    )
    second = _executor("invalid").decide(
        "observation",
        None,
        ["accelerate", "decelerate"],
    )

    assert first.primitive == "accelerate"
    assert second.primitive == "accelerate"
    assert first.available_primitives == ["accelerate", "decelerate"]
    assert second.available_primitives == ["accelerate", "decelerate"]


def test_unknown_and_duplicate_supplied_primitives_are_filtered():
    supplied = [
        "teleport",
        "decelerate",
        "accelerate",
        "decelerate",
        "unknown",
    ]

    result = _executor('{"primitive": "accelerate"}').decide(
        "observation",
        None,
        supplied,
    )
    supplied.append("maintain-speed")

    assert result.available_primitives == ["accelerate", "decelerate"]
    assert result.decision_status == "ok"


def test_empty_allowed_intersection_raises_without_calling_llm():
    calls = 0

    def reply(_messages):
        nonlocal calls
        calls += 1
        return '{"primitive": "accelerate"}'

    executor = _executor(client=MockLLMClient(reply))

    with pytest.raises(
        NoAvailablePrimitiveError,
        match="no skill primitive is currently available",
    ):
        executor.decide("observation", None, ["teleport"])

    assert calls == 0


def test_prompt_contains_full_skill_contract_and_filtered_allowed_list():
    executor = _executor()

    messages = executor.build_messages(
        "Ego: lane=1 speed=20.0 m/s",
        ["teleport", "decelerate", "accelerate"],
    )

    assert [message.role for message in messages] == ["system", "user"]
    system = messages[0].content
    user = messages[1].content
    assert 'ONLY one JSON object: {"primitive": "<allowed primitive>"}' in system
    assert "no overlap between ego bounding box and any other vehicle" in system
    assert "# Lane Change Overtake" in user
    assert "## Procedure" in user
    assert "# Parameters" in user
    assert '"target_speed"' in user
    assert "# Preconditions" in user
    assert "ego.lane has a slower lead vehicle" in user
    assert "# Execution graph" in user
    assert '"id": "check-lead"' in user
    assert '"action": "decelerate"' in user
    assert "# Effects" in user
    assert "ego ahead of the previously blocking lead vehicle" in user
    assert "# Success criteria" in user
    assert "ego overtook the lead vehicle" in user
    assert "# Failure criteria" in user
    assert "collision == true" in user
    assert "# Safety constraints" in user
    assert "lane change only into a lane that exists" in user
    assert "# Failure modes" in user
    assert "oscillating between lanes without net progress" in user
    assert "# Termination" in user
    assert "ego past the lead vehicle at target speed OR overtake aborted" in user
    assert "# Current observation" in user
    assert "Ego: lane=1 speed=20.0 m/s" in user
    assert (
        '# Allowed primitives this step\n["accelerate", "decelerate"]'
        in user
    )


def test_build_messages_accepts_approved_observation_context_interface():
    context = _observation_context(
        ["teleport", "decelerate", "accelerate"]
    )

    messages = _executor().build_messages("observation", context)

    assert (
        '# Allowed primitives this step\n["accelerate", "decelerate"]'
        in messages[1].content
    )


def test_decide_accepts_matching_filtered_context_and_explicit_availability():
    context = _observation_context(
        ["unknown", "decelerate", "accelerate", "decelerate"]
    )

    result = _executor('{"primitive": "accelerate"}').decide(
        "observation",
        context,
        ["accelerate", "teleport", "decelerate"],
    )

    assert result.decision_status == "ok"
    assert result.available_primitives == ["accelerate", "decelerate"]


def test_decide_rejects_context_availability_mismatch_before_llm_call():
    calls = 0

    def reply(_messages):
        nonlocal calls
        calls += 1
        return '{"primitive": "accelerate"}'

    executor = _executor(client=MockLLMClient(reply))
    context = _observation_context(["maintain-speed", "accelerate"])

    with pytest.raises(
        ValueError,
        match="context available_primitives do not match explicit availability",
    ):
        executor.decide(
            "observation",
            context,
            ["maintain-speed", "decelerate"],
        )

    assert calls == 0


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda primitive_map, name_to_index: primitive_map.pop(
                "change-lane-left"
            ),
            "primitive_map is missing skill primitives",
        ),
        (
            lambda primitive_map, name_to_index: name_to_index.pop("FASTER"),
            "backend actions missing from name_to_index",
        ),
        (
            lambda primitive_map, name_to_index: name_to_index.update(
                {"SLOWER": 3}
            ),
            "duplicate action indices",
        ),
        (
            lambda primitive_map, name_to_index: name_to_index.update(
                {"FASTER": "3"}
            ),
            "action indices must be non-negative integers",
        ),
        (
            lambda primitive_map, name_to_index: name_to_index.update(
                {"FASTER": True}
            ),
            "action indices must be non-negative integers",
        ),
        (
            lambda primitive_map, name_to_index: name_to_index.update(
                {"FASTER": -1}
            ),
            "action indices must be non-negative integers",
        ),
    ],
)
def test_constructor_rejects_invalid_action_contract(mutator, message: str):
    card, grounding = _skill_and_grounding()
    name_to_index = NAME_TO_INDEX.copy()
    mutator(grounding.primitive_map, name_to_index)

    with pytest.raises(ValueError, match=message):
        LLMSkillExecutor(
            card,
            grounding,
            MockLLMClient(['{"primitive": "accelerate"}']),
            "lco-v1",
            name_to_index,
        )


def test_constructor_copies_action_mappings():
    card, grounding = _skill_and_grounding()
    name_to_index = NAME_TO_INDEX.copy()
    executor = LLMSkillExecutor(
        card,
        grounding,
        MockLLMClient(['{"primitive": "accelerate"}']),
        "lco-v1",
        name_to_index,
    )

    grounding.primitive_map["accelerate"] = "IDLE"
    name_to_index["FASTER"] = 99
    result = executor.decide("observation", None, ["accelerate"])

    assert result.backend_action == "FASTER"
    assert result.action_index == 3


def test_constructor_allows_many_primitives_to_share_one_backend_action():
    card, grounding = _skill_and_grounding()
    grounding.primitive_map["change-lane-left"] = "IDLE"
    executor = LLMSkillExecutor(
        card,
        grounding,
        MockLLMClient(['{"primitive": "change-lane-left"}']),
        "lco-v1",
        NAME_TO_INDEX,
    )

    result = executor.decide(
        "observation",
        None,
        ["change-lane-left"],
    )

    assert result.backend_action == "IDLE"
    assert result.action_index == 1


def test_constructor_ignores_unrelated_extra_primitive_mapping():
    card, grounding = _skill_and_grounding()
    grounding.primitive_map["runtime-diagnostic"] = "NOT_AN_ENV_ACTION"
    executor = LLMSkillExecutor(
        card,
        grounding,
        MockLLMClient(['{"primitive": "accelerate"}']),
        "lco-v1",
        NAME_TO_INDEX,
    )

    result = executor.decide("observation", None, ["accelerate"])

    assert result.backend_action == "FASTER"
    assert result.action_index == 3


def test_constructor_deep_snapshots_skill_card_and_grounding():
    card, grounding = _skill_and_grounding()
    executor = LLMSkillExecutor(
        card,
        grounding,
        MockLLMClient(['{"primitive": "accelerate"}']),
        "lco-v1",
        NAME_TO_INDEX,
    )

    card.skill_md_body = "MUTATED SKILL BODY"
    card.effects[0] = "MUTATED EFFECT"
    card.parameters["target_speed"]["default"] = 999
    card.primitives[:] = ["teleport"]
    grounding.action["type"] = "MUTATED ACTION TYPE"
    grounding.primitive_map["accelerate"] = "IDLE"

    messages = executor.build_messages("observation", ["accelerate"])
    result = executor.decide("observation", None, ["accelerate"])
    prompt = "\n".join(message.content for message in messages)

    assert "# Lane Change Overtake" in prompt
    assert "ego ahead of the previously blocking lead vehicle" in prompt
    assert '"default": 25' in prompt
    assert "MUTATED" not in prompt
    assert result.decision_status == "ok"
    assert result.backend_action == "FASTER"
    assert result.action_index == 3
    assert executor.skill_card is not card
    assert executor.grounding is not grounding
    assert executor.grounding.action["type"] == "DiscreteMetaAction"


class RaisingClient:
    def __init__(
        self,
        error: Exception,
        *,
        model: str = "failing-model",
        api_type: str = "test-api",
        base_url: str = "https://executor.test/v1",
        api_version: str = "2026-07-21",
    ):
        self.error = error
        self.model = model
        self.api_type = api_type
        self.base_url = base_url
        self.api_version = api_version

    def chat(self, *_args, **_kwargs):
        raise self.error


def test_llm_request_error_preserves_trace_identity_and_original_error():
    original = TimeoutError("provider timed out")
    error = LLMRequestError(
        request_hash="request-abc",
        latency_ms=321.5,
        original_exception=original,
    )

    result = _executor(client=RaisingClient(error)).decide(
        "observation",
        None,
        ["accelerate", "maintain-speed"],
    )

    assert result.decision_status == "llm_error_fallback"
    assert result.primitive == "maintain-speed"
    assert result.request_hash == "request-abc"
    assert result.latency_ms == 321.5
    assert result.raw_response == ""
    assert result.cache_hit is False
    assert result.fallback_reason == "TimeoutError: provider timed out"


def test_generic_client_error_uses_local_audit_hash_and_fallback():
    client = RaisingClient(RuntimeError("provider exploded"))
    executor = _executor(client=client)
    allowed = ["accelerate"]
    messages = executor.build_messages("observation", allowed)
    expected_hash = make_request_hash(
        client_type="RaisingClient",
        model=client.model,
        api_type=client.api_type,
        base_url=client.base_url,
        api_version=client.api_version,
        effective_settings=ModelSettings().for_chat_completions(),
        messages=messages,
        prompt_version="lco-v1",
    )

    result = executor.decide("observation", None, allowed)

    assert result.decision_status == "llm_error_fallback"
    assert result.primitive == "accelerate"
    assert result.request_hash == expected_hash
    assert len(result.request_hash) == 64
    assert result.raw_response == ""
    assert result.cache_hit is False
    assert result.latency_ms > 0
    assert result.fallback_reason == "RuntimeError: provider exploded"


class FakeActionType:
    def get_available_actions(self):
        return [4, 99, 3, 0, 3]


def test_available_primitives_maps_env_indices_in_skill_order():
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(action_type=FakeActionType())
    )

    available = _executor().available_primitives(env)

    assert available == [
        "change-lane-left",
        "accelerate",
        "decelerate",
    ]
