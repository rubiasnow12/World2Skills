"""Shared dataclasses for the World2Skills experiment loop.

Pure data holders (no behavior beyond one convenience constructor). All are
``dataclasses.asdict``-serializable so results and traces can be written as JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Grounding:
    backend: str
    backend_version: str
    environment: str
    observation: dict[str, Any]
    action: dict[str, Any]
    primitive_map: dict[str, str]


@dataclass
class SkillCard:
    name: str
    description: str
    skill_md_body: str
    parameters: dict[str, Any]
    interface: dict[str, Any]
    execution: dict[str, Any]
    preconditions: list[str]
    success_criteria: list[str]
    failure_criteria: list[str]
    safety_constraints: list[str]
    groundings: list[Grounding]
    primitives: list[str]


@dataclass
class ObservationContext:
    available_primitives: list[str]
    ego_lane: int
    prev_primitive: str | None
    target_ahead: bool
    target_gap_m: float
    target_rel_speed_mps: float
    left_lane_exists: bool
    right_lane_exists: bool
    left_front_gap_m: float | None
    left_rear_gap_m: float | None
    left_rear_closing_speed_mps: float | None
    right_front_gap_m: float | None
    right_rear_gap_m: float | None
    right_rear_closing_speed_mps: float | None


@dataclass
class DecisionResult:
    """Executor output available before ``env.step()``."""

    primitive: str
    backend_action: str
    action_index: int
    request_hash: str
    raw_response: str
    cache_hit: bool
    latency_ms: float
    decision_status: str
    fallback_reason: str | None
    available_primitives: list[str]


@dataclass
class StepRecord:
    """A decision plus its outcome, assembled after ``env.step()``."""

    t: int
    obs_summary: str
    primitive: str
    backend_action: str
    action_index: int
    reward: float
    crashed: bool
    request_hash: str
    raw_response: str
    cache_hit: bool
    latency_ms: float
    decision_status: str
    fallback_reason: str | None
    available_primitives: list[str]

    @classmethod
    def from_decision(
        cls,
        dr: DecisionResult,
        *,
        t: int,
        obs_summary: str,
        reward: float,
        crashed: bool,
    ) -> StepRecord:
        return cls(
            t=t,
            obs_summary=obs_summary,
            primitive=dr.primitive,
            backend_action=dr.backend_action,
            action_index=dr.action_index,
            reward=reward,
            crashed=crashed,
            request_hash=dr.request_hash,
            raw_response=dr.raw_response,
            cache_hit=dr.cache_hit,
            latency_ms=dr.latency_ms,
            decision_status=dr.decision_status,
            fallback_reason=dr.fallback_reason,
            available_primitives=dr.available_primitives.copy(),
        )


@dataclass
class EpisodeResult:
    seed: int
    status: str
    success: bool
    success_reason: str
    episode_return: float
    crashed: bool
    steps: int
    mean_speed: float
    parse_failures: int
    unavailable_action_attempts: int
    llm_errors: int
    terminated: bool
    truncated: bool
    max_steps_reached: bool
    scenario_completed: bool
    termination_reason: str
    target_initially_ahead: bool
    lane_change_completed_step: int | None
    overtake_step: int | None
    exception_type: str | None = None
    step_records: list[StepRecord] = field(default_factory=list)


@dataclass
class BatchResult:
    schema_version: str
    model: str
    prompt_version: str
    skill: str
    environment: str
    seeds: list[int]
    requested_episodes: int
    completed_episodes: int
    error_episodes: int
    successful_episodes: int
    success_rate: float
    collision_rate: float
    mean_return: float
    results: list[dict[str, Any]]
