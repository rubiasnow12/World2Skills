import json
from dataclasses import asdict

from world2skills.runtime.types import (
    DecisionResult,
    EpisodeResult,
    Grounding,
    SkillCard,
    StepRecord,
)


def test_grounding_and_skill_card_json_contract():
    grounding = Grounding(
        backend="highway-env",
        backend_version=">=1.8",
        environment="highway-v0",
        observation={"type": "Kinematics"},
        action={"type": "DiscreteMetaAction"},
        primitive_map={"accelerate": "FASTER"},
    )
    skill = SkillCard(
        name="lane-change-overtake",
        description="Overtake a slower lead vehicle.",
        skill_md_body="# Lane Change Overtake",
        parameters={"target_speed": {"default": 25}},
        interface={"actions": []},
        execution={"entry": "check-lead"},
        preconditions=["a slower lead vehicle exists"],
        effects=["ego is ahead of the blocking vehicle"],
        success_criteria=["ego overtook the lead vehicle"],
        failure_criteria=["collision == true"],
        safety_constraints=["lane change only into an existing lane"],
        failure_modes=["changing into an occupied gap"],
        termination="ego passed the lead vehicle",
        groundings=[grounding],
        primitives=["accelerate"],
    )

    grounding_payload = asdict(grounding)
    skill_payload = asdict(skill)

    assert grounding_payload["backend_version"] == ">=1.8"
    assert skill_payload["success_criteria"] == ["ego overtook the lead vehicle"]
    assert skill_payload["failure_criteria"] == ["collision == true"]
    assert skill_payload["safety_constraints"] == [
        "lane change only into an existing lane"
    ]
    assert skill_payload["effects"] == ["ego is ahead of the blocking vehicle"]
    assert skill_payload["failure_modes"] == ["changing into an occupied gap"]
    assert skill_payload["termination"] == "ego passed the lead vehicle"
    assert skill_payload["groundings"][0]["backend_version"] == ">=1.8"
    json.dumps(grounding_payload)
    json.dumps(skill_payload)


def test_steprecord_from_decision_and_json():
    dr = DecisionResult(
        primitive="accelerate",
        backend_action="FASTER",
        action_index=3,
        request_hash="abc",
        raw_response='{"primitive": "accelerate"}',
        cache_hit=False,
        latency_ms=12.0,
        decision_status="ok",
        fallback_reason=None,
        available_primitives=["accelerate", "maintain-speed"],
    )
    rec = StepRecord.from_decision(
        dr,
        t=0,
        obs_summary="ego...",
        reward=1.0,
        crashed=False,
    )
    assert rec.t == 0 and rec.primitive == "accelerate" and rec.action_index == 3
    assert rec.reward == 1.0 and rec.crashed is False
    json.dumps(asdict(rec))


def test_steprecord_copies_available_primitives():
    dr = DecisionResult(
        primitive="accelerate",
        backend_action="FASTER",
        action_index=3,
        request_hash="abc",
        raw_response='{"primitive": "accelerate"}',
        cache_hit=False,
        latency_ms=12.0,
        decision_status="ok",
        fallback_reason=None,
        available_primitives=["accelerate", "maintain-speed"],
    )

    rec = StepRecord.from_decision(
        dr,
        t=0,
        obs_summary="ego...",
        reward=1.0,
        crashed=False,
    )
    dr.available_primitives.append("decelerate")

    assert rec.available_primitives == ["accelerate", "maintain-speed"]


def test_episode_result_defaults_json():
    er = EpisodeResult(
        seed=0,
        status="ok",
        success=True,
        success_reason="ok",
        episode_return=1.0,
        crashed=False,
        steps=3,
        mean_speed=20.0,
        parse_failures=0,
        unavailable_action_attempts=0,
        llm_errors=0,
        terminated=False,
        truncated=False,
        max_steps_reached=False,
        scenario_completed=True,
        termination_reason="success",
        target_initially_ahead=True,
        lane_change_completed_step=1,
        overtake_step=2,
    )
    assert er.exception_type is None and er.step_records == []
    json.dumps(asdict(er))
