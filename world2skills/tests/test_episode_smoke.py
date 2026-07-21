from __future__ import annotations

from collections.abc import Mapping
import re
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("highway_env")

from world2skills.evaluation.episode import run_episode
from world2skills.runtime.env_factory import make_env
from world2skills.runtime.executor import LLMSkillExecutor
from world2skills.runtime.llm import Message, MockLLMClient
from world2skills.runtime.scenario import LaneChangeOvertakeScenario
from world2skills.runtime.skill_loader import load_skill, select_grounding
from world2skills.runtime.types import DecisionResult, ObservationContext


_EGO_LANE = re.compile(r"Ego: .* lane=(?P<lane>\d+)")
_FEATURES = ["presence", "x", "y", "vx", "vy"]
_OBSERVATION = np.array(
    [
        [1.0, 0.0, 0.0, 20.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)
_DEFAULT_SCENARIO_RESULT = object()


def _scripted_overtake(messages: list[Message]) -> str:
    prompt = messages[-1].content
    lane_match = _EGO_LANE.search(prompt)
    if lane_match is None:
        raise AssertionError("prompt omitted ego lane")
    if "Previous primitive: none" in prompt:
        primitive = "change-lane-left"
    elif int(lane_match.group("lane")) == 1:
        primitive = "maintain-speed"
    else:
        primitive = "accelerate"
    return f'{{"primitive": "{primitive}"}}'


@pytest.mark.parametrize("seed", range(5))
def test_scripted_episode_completes_overtake_for_m1_seeds(seed: int) -> None:
    card = load_skill("lane-change-overtake")
    grounding = select_grounding(card, "highway-env")
    scenario = LaneChangeOvertakeScenario(card)
    env, name_to_index = make_env(
        grounding,
        scenario.configure(),
        seed=seed,
    )
    executor = LLMSkillExecutor(
        card,
        grounding,
        MockLLMClient(_scripted_overtake),
        "lco-v1",
        name_to_index,
    )

    result = run_episode(
        env,
        executor,
        scenario,
        seed=seed,
        max_steps=40,
    )

    assert result.status == "ok"
    assert result.success is True, result.success_reason
    assert result.scenario_completed is True
    assert result.termination_reason == "success"
    assert result.crashed is False
    assert result.terminated is False
    assert result.truncated is False
    assert result.target_initially_ahead is True
    assert result.lane_change_completed_step is not None
    assert result.overtake_step is not None
    assert result.lane_change_completed_step < result.overtake_step
    assert result.lead_initial_gap_m == pytest.approx(scenario.spawn_gap)
    assert result.steps == len(result.step_records)
    assert result.steps > 0
    for record in result.step_records:
        assert "Target:" in record.obs_summary
        assert "Left lane:" in record.obs_summary
        assert "Right lane:" in record.obs_summary
        assert "Available primitives:" in record.obs_summary


class _ResetError(RuntimeError):
    pass


class _UpdateError(RuntimeError):
    pass


class _CloseError(RuntimeError):
    pass


def _context(available_primitives: list[str]) -> ObservationContext:
    return ObservationContext(
        available_primitives=available_primitives.copy(),
        ego_lane=1,
        prev_primitive=None,
        target_ahead=True,
        target_gap_m=30.0,
        target_rel_speed_mps=-5.0,
        left_lane_exists=True,
        right_lane_exists=True,
        left_front_gap_m=None,
        left_rear_gap_m=None,
        left_rear_closing_speed_mps=None,
        right_front_gap_m=None,
        right_rear_gap_m=None,
        right_rear_closing_speed_mps=None,
    )


def _decision(
    status: str = "ok",
    *,
    primitive: str = "maintain-speed",
) -> DecisionResult:
    action_index = 1 if primitive == "maintain-speed" else 3
    return DecisionResult(
        primitive=primitive,
        backend_action="IDLE" if action_index == 1 else "FASTER",
        action_index=action_index,
        request_hash=f"hash-{status}",
        raw_response=f'{{"primitive": "{primitive}"}}',
        cache_hit=False,
        latency_ms=1.5,
        decision_status=status,
        fallback_reason=None if status == "ok" else status,
        available_primitives=["maintain-speed", "accelerate"],
    )


class _FakeExecutor:
    def __init__(self, decisions: list[DecisionResult]) -> None:
        self.grounding = SimpleNamespace(observation={"features": _FEATURES.copy()})
        self._decisions = decisions
        self._index = 0
        self.decide_calls: list[tuple[str, ObservationContext, list[str]]] = []

    def available_primitives(self, env: Any) -> list[str]:
        del env
        return ["maintain-speed", "accelerate"]

    def decide(
        self,
        obs_text: str,
        context: ObservationContext,
        available_primitives: list[str],
    ) -> DecisionResult:
        assert context.available_primitives == available_primitives
        self.decide_calls.append((obs_text, context, available_primitives.copy()))
        decision = self._decisions[min(self._index, len(self._decisions) - 1)]
        self._index += 1
        return decision


class _FakeEnv:
    def __init__(
        self,
        transitions: list[tuple[np.ndarray, Any, Any, Any, Any]],
        *,
        close_error: Exception | None = None,
    ) -> None:
        self._transitions = transitions
        self._index = 0
        self.close_error = close_error
        self.close_calls = 0
        self.unwrapped = SimpleNamespace(vehicle=SimpleNamespace(crashed=False))

    def step(
        self,
        action_index: int,
    ) -> tuple[np.ndarray, Any, Any, Any, Any]:
        del action_index
        transition = self._transitions[self._index]
        self._index += 1
        info = transition[4]
        if isinstance(info, Mapping):
            raw_crashed = info.get("crashed", False)
            if isinstance(raw_crashed, (bool, np.bool_)):
                self.unwrapped.vehicle.crashed = bool(raw_crashed)
        return transition

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class _FakeScenario:
    def __init__(
        self,
        *,
        success_after_update: bool = False,
        reset_error: Exception | None = None,
        update_error: Exception | None = None,
        terminal_result: Any = _DEFAULT_SCENARIO_RESULT,
        evaluate_result: Any = _DEFAULT_SCENARIO_RESULT,
    ) -> None:
        self.success_after_update = success_after_update
        self.reset_error = reset_error
        self.update_error = update_error
        self.terminal_result = terminal_result
        self.evaluate_result = evaluate_result
        self.evaluate_calls = 0
        self.target_initially_ahead = True
        self.lane_change_completed_step: int | None = None
        self.overtake_step: int | None = None
        self.lead_initial_gap_m: float | None = 30.0
        self._success = False

    def reset(
        self,
        env: Any,
        seed: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del env, seed
        if self.reset_error is not None:
            raise self.reset_error
        return _OBSERVATION.copy(), {}

    def build_context(
        self,
        env: Any,
        prev_primitive: str | None,
        available_primitives: list[str],
    ) -> ObservationContext:
        del env, prev_primitive
        return _context(available_primitives)

    def update(self, env: Any, t: int) -> None:
        del env
        if self.update_error is not None:
            raise self.update_error
        if self.success_after_update:
            self.lane_change_completed_step = t
            self.overtake_step = t + 1
            self._success = True

    def is_terminal(self, env: Any) -> Any:
        del env
        if self.terminal_result is not _DEFAULT_SCENARIO_RESULT:
            return self.terminal_result
        return self._success, "success" if self._success else ""

    def evaluate(self, env: Any) -> Any:
        del env
        self.evaluate_calls += 1
        if self.evaluate_result is not _DEFAULT_SCENARIO_RESULT:
            return self.evaluate_result
        return (True, "scripted success") if self._success else (False, "not complete")


def _transition(
    *,
    reward: Any = 1.0,
    terminated: Any = False,
    truncated: Any = False,
    speed: float = 20.0,
    crashed: Any = False,
) -> tuple[np.ndarray, Any, Any, Any, dict[str, Any]]:
    return (
        _OBSERVATION.copy(),
        reward,
        terminated,
        truncated,
        {"speed": speed, "crashed": crashed},
    )


@pytest.mark.parametrize(
    (
        "scenario_success",
        "terminated",
        "truncated",
        "crashed",
        "expected_reason",
        "expected_completed",
        "expected_success",
    ),
    [
        (True, False, False, False, "success", True, True),
        (True, False, True, False, "success+truncated", True, True),
        (True, True, False, False, "success+terminated", True, True),
        (False, True, False, True, "crashed", False, False),
        (False, True, False, False, "terminated", False, False),
        (False, False, True, False, "truncated", False, False),
    ],
)
def test_episode_termination_reason_precedence(
    scenario_success: bool,
    terminated: bool,
    truncated: bool,
    crashed: bool,
    expected_reason: str,
    expected_completed: bool,
    expected_success: bool,
) -> None:
    env = _FakeEnv(
        [
            _transition(
                terminated=terminated,
                truncated=truncated,
                crashed=crashed,
            )
        ]
    )
    scenario = _FakeScenario(success_after_update=scenario_success)

    result = run_episode(
        env,
        _FakeExecutor([_decision()]),
        scenario,
        seed=0,
        max_steps=2,
    )

    assert result.status == "ok"
    assert result.success is expected_success
    assert result.scenario_completed is expected_completed
    assert result.terminated is terminated
    assert result.truncated is truncated
    assert result.crashed is crashed
    assert result.termination_reason == expected_reason
    assert env.close_calls == 1


def test_episode_counts_each_fallback_status() -> None:
    env = _FakeEnv(
        [
            _transition(),
            _transition(),
            _transition(truncated=True),
        ]
    )
    executor = _FakeExecutor(
        [
            _decision("parse_fallback"),
            _decision("unavailable_fallback"),
            _decision("llm_error_fallback"),
        ]
    )

    result = run_episode(
        env,
        executor,
        _FakeScenario(),
        seed=0,
        max_steps=5,
    )

    assert result.parse_failures == 1
    assert result.unavailable_action_attempts == 1
    assert result.llm_errors == 1
    assert result.steps == 3
    assert result.termination_reason == "truncated"


def test_error_after_step_preserves_partial_metrics_and_record() -> None:
    env = _FakeEnv(
        [
            _transition(
                reward=2.5,
                terminated=True,
                speed=12.5,
                crashed=True,
            )
        ]
    )
    scenario = _FakeScenario(update_error=_UpdateError("update failed"))

    result = run_episode(
        env,
        _FakeExecutor([_decision("parse_fallback")]),
        scenario,
        seed=7,
        max_steps=3,
    )

    assert result.status == "error"
    assert result.success is False
    assert result.termination_reason == "error"
    assert result.exception_type == "_UpdateError"
    assert result.exception_message == "update failed"
    assert result.episode_return == pytest.approx(2.5)
    assert result.mean_speed == pytest.approx(12.5)
    assert result.crashed is True
    assert result.terminated is True
    assert result.truncated is False
    assert result.parse_failures == 1
    assert result.steps == 1
    assert len(result.step_records) == 1
    assert result.step_records[0].reward == pytest.approx(2.5)
    assert result.step_records[0].crashed is True
    assert "Available primitives:" in result.step_records[0].obs_summary
    assert env.close_calls == 1


def test_non_mapping_info_preserves_completed_step_accounting() -> None:
    env = _FakeEnv(
        [
            (
                _OBSERVATION.copy(),
                2.25,
                True,
                False,
                ["not", "a", "mapping"],
            )
        ]
    )
    env.unwrapped.vehicle.crashed = True

    result = run_episode(
        env,
        _FakeExecutor([_decision("parse_fallback")]),
        _FakeScenario(),
        seed=0,
        max_steps=2,
    )

    assert result.status == "error"
    assert result.exception_type == "TypeError"
    assert result.exception_message == "env.step info must be a mapping"
    assert result.episode_return == pytest.approx(2.25)
    assert result.mean_speed == 0.0
    assert result.crashed is True
    assert result.terminated is True
    assert result.truncated is False
    assert result.parse_failures == 1
    assert result.steps == 1
    assert result.step_records[0].reward == pytest.approx(2.25)
    assert result.step_records[0].crashed is True
    assert "Target:" in result.step_records[0].obs_summary
    assert env.close_calls == 1


def test_missing_speed_preserves_completed_step_accounting() -> None:
    env = _FakeEnv(
        [
            (
                _OBSERVATION.copy(),
                1.75,
                False,
                True,
                {"crashed": False},
            )
        ]
    )

    result = run_episode(
        env,
        _FakeExecutor([_decision("llm_error_fallback")]),
        _FakeScenario(),
        seed=0,
        max_steps=2,
    )

    assert result.status == "error"
    assert result.exception_type == "ValueError"
    assert result.exception_message == "step info must contain speed"
    assert result.episode_return == pytest.approx(1.75)
    assert result.mean_speed == 0.0
    assert result.crashed is False
    assert result.terminated is False
    assert result.truncated is True
    assert result.llm_errors == 1
    assert result.steps == 1
    assert result.step_records[0].reward == pytest.approx(1.75)
    assert result.step_records[0].crashed is False
    assert "Available primitives:" in result.step_records[0].obs_summary
    assert env.close_calls == 1


@pytest.mark.parametrize(
    "reward",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-inf"),
        pytest.param(float("-inf"), id="negative-inf"),
        pytest.param(True, id="bool"),
        pytest.param(np.bool_(True), id="numpy-bool"),
    ],
)
def test_invalid_reward_preserves_step_without_counting_return(
    reward: Any,
) -> None:
    env = _FakeEnv(
        [
            _transition(
                reward=reward,
                terminated=True,
                speed=18.0,
                crashed=False,
            )
        ]
    )

    result = run_episode(
        env,
        _FakeExecutor([_decision("parse_fallback")]),
        _FakeScenario(),
        seed=0,
        max_steps=2,
    )

    assert result.status == "error"
    assert result.exception_type == "ValueError"
    assert result.exception_message == ("env.step reward must be a finite real number")
    assert result.episode_return == 0.0
    assert result.mean_speed == pytest.approx(18.0)
    assert result.terminated is True
    assert result.truncated is False
    assert result.parse_failures == 1
    assert result.steps == 1
    assert result.step_records[0].reward is None
    assert result.step_records[0].crashed is False
    assert env.close_calls == 1


@pytest.mark.parametrize(
    ("terminated", "truncated", "expected_terminated", "expected_truncated"),
    [
        ("false", np.bool_(True), False, True),
        (np.bool_(True), "false", True, False),
    ],
)
def test_string_step_flags_are_rejected_without_coercion(
    terminated: Any,
    truncated: Any,
    expected_terminated: bool,
    expected_truncated: bool,
) -> None:
    env = _FakeEnv(
        [
            _transition(
                reward=1.25,
                terminated=terminated,
                truncated=truncated,
                speed=19.0,
            )
        ]
    )

    result = run_episode(
        env,
        _FakeExecutor([_decision("unavailable_fallback")]),
        _FakeScenario(),
        seed=0,
        max_steps=2,
    )

    assert result.status == "error"
    assert result.exception_type == "TypeError"
    assert "must be bool" in result.exception_message
    assert result.episode_return == pytest.approx(1.25)
    assert result.mean_speed == pytest.approx(19.0)
    assert result.terminated is expected_terminated
    assert result.truncated is expected_truncated
    assert result.unavailable_action_attempts == 1
    assert result.steps == 1
    assert result.step_records[0].reward == pytest.approx(1.25)
    assert env.close_calls == 1


def test_numpy_bool_step_flags_and_crash_are_accepted() -> None:
    env = _FakeEnv(
        [
            _transition(
                reward=np.float64(1.5),
                terminated=np.bool_(False),
                truncated=np.bool_(True),
                crashed=np.bool_(False),
            )
        ]
    )

    result = run_episode(
        env,
        _FakeExecutor([_decision()]),
        _FakeScenario(),
        seed=0,
        max_steps=2,
    )

    assert result.status == "ok"
    assert result.episode_return == pytest.approx(1.5)
    assert result.terminated is False
    assert result.truncated is True
    assert result.crashed is False
    assert result.termination_reason == "truncated"
    assert env.close_calls == 1


def test_string_info_crashed_is_rejected_without_coercion() -> None:
    env = _FakeEnv(
        [
            _transition(
                reward=1.5,
                terminated=True,
                speed=17.0,
                crashed="false",
            )
        ]
    )
    env.unwrapped.vehicle.crashed = True

    result = run_episode(
        env,
        _FakeExecutor([_decision("llm_error_fallback")]),
        _FakeScenario(),
        seed=0,
        max_steps=2,
    )

    assert result.status == "error"
    assert result.exception_type == "TypeError"
    assert result.exception_message == "step info crashed must be bool"
    assert result.episode_return == pytest.approx(1.5)
    assert result.mean_speed == pytest.approx(17.0)
    assert result.terminated is True
    assert result.crashed is True
    assert result.llm_errors == 1
    assert result.steps == 1
    assert result.step_records[0].reward == pytest.approx(1.5)
    assert result.step_records[0].crashed is True
    assert env.close_calls == 1


@pytest.mark.parametrize(
    "terminal_result",
    [
        pytest.param([False, ""], id="list"),
        pytest.param((False,), id="short-tuple"),
        pytest.param((0, ""), id="non-bool"),
        pytest.param((False, None), id="non-string-reason"),
    ],
)
def test_malformed_scenario_terminal_result_preserves_partial_step(
    terminal_result: Any,
) -> None:
    env = _FakeEnv([_transition(reward=2.0, speed=16.0)])
    scenario = _FakeScenario(terminal_result=terminal_result)

    result = run_episode(
        env,
        _FakeExecutor([_decision()]),
        scenario,
        seed=0,
        max_steps=2,
    )

    assert result.status == "error"
    assert result.exception_type == "TypeError"
    assert "scenario.is_terminal" in result.exception_message
    assert result.episode_return == pytest.approx(2.0)
    assert result.mean_speed == pytest.approx(16.0)
    assert result.steps == 1
    assert scenario.evaluate_calls == 0
    assert env.close_calls == 1


@pytest.mark.parametrize(
    "evaluate_result",
    [
        pytest.param([False, ""], id="list"),
        pytest.param((False,), id="short-tuple"),
        pytest.param((np.bool_(False), ""), id="numpy-bool"),
        pytest.param((False, None), id="non-string-reason"),
    ],
)
def test_malformed_scenario_evaluate_result_preserves_partial_step(
    evaluate_result: Any,
) -> None:
    env = _FakeEnv([_transition(reward=2.0, truncated=True)])
    scenario = _FakeScenario(evaluate_result=evaluate_result)

    result = run_episode(
        env,
        _FakeExecutor([_decision()]),
        scenario,
        seed=0,
        max_steps=2,
    )

    assert result.status == "error"
    assert result.exception_type == "TypeError"
    assert "scenario.evaluate" in result.exception_message
    assert result.episode_return == pytest.approx(2.0)
    assert result.truncated is True
    assert result.steps == 1
    assert scenario.evaluate_calls == 1
    assert env.close_calls == 1


@pytest.mark.parametrize(
    ("terminal_result", "evaluate_result", "expected_completed"),
    [
        ((True, "success"), (False, "lost"), True),
        ((False, ""), (True, "success"), False),
    ],
)
def test_scenario_completion_must_match_final_success(
    terminal_result: tuple[bool, str],
    evaluate_result: tuple[bool, str],
    expected_completed: bool,
) -> None:
    env = _FakeEnv([_transition()])
    scenario = _FakeScenario(
        terminal_result=terminal_result,
        evaluate_result=evaluate_result,
    )

    result = run_episode(
        env,
        _FakeExecutor([_decision()]),
        scenario,
        seed=0,
        max_steps=1,
    )

    assert result.status == "error"
    assert result.exception_type == "ValueError"
    assert result.exception_message == (
        "scenario completion disagrees with final success"
    )
    assert result.scenario_completed is expected_completed
    assert result.steps == 1
    assert env.close_calls == 1


def test_reset_error_returns_error_result_and_closes() -> None:
    env = _FakeEnv([])
    scenario = _FakeScenario(reset_error=_ResetError("reset failed"))

    result = run_episode(
        env,
        _FakeExecutor([_decision()]),
        scenario,
        seed=3,
        max_steps=2,
    )

    assert result.status == "error"
    assert result.seed == 3
    assert result.steps == 0
    assert result.exception_type == "_ResetError"
    assert result.exception_message == "reset failed"
    assert result.lead_initial_gap_m == pytest.approx(30.0)
    assert env.close_calls == 1


def test_close_error_does_not_mask_primary_error() -> None:
    env = _FakeEnv(
        [],
        close_error=_CloseError("close failed"),
    )
    scenario = _FakeScenario(reset_error=_ResetError("reset failed"))

    result = run_episode(
        env,
        _FakeExecutor([_decision()]),
        scenario,
        seed=0,
        max_steps=2,
    )

    assert result.status == "error"
    assert result.exception_type == "_ResetError"
    assert result.exception_message == "reset failed"
    assert result.cleanup_error == "_CloseError: close failed"
    assert env.close_calls == 1


def test_close_error_does_not_replace_normal_result() -> None:
    env = _FakeEnv(
        [_transition()],
        close_error=_CloseError("close failed"),
    )

    result = run_episode(
        env,
        _FakeExecutor([_decision()]),
        _FakeScenario(),
        seed=0,
        max_steps=1,
    )

    assert result.status == "ok"
    assert result.termination_reason == "max_steps"
    assert result.cleanup_error == "_CloseError: close failed"
    assert env.close_calls == 1


def test_invalid_arguments_preserve_primary_error_when_close_fails() -> None:
    env = _FakeEnv(
        [],
        close_error=_CloseError("close failed"),
    )

    with pytest.raises(
        ValueError,
        match="seed must be a nonnegative integer",
    ) as caught:
        run_episode(
            env,
            _FakeExecutor([_decision()]),
            _FakeScenario(),
            seed=-1,
            max_steps=1,
        )

    assert caught.value.__notes__ == ["cleanup_error: _CloseError: close failed"]
    assert env.close_calls == 1


def test_max_steps_sets_explicit_stop_state() -> None:
    env = _FakeEnv([_transition(), _transition()])

    result = run_episode(
        env,
        _FakeExecutor([_decision()]),
        _FakeScenario(),
        seed=0,
        max_steps=2,
    )

    assert result.status == "ok"
    assert result.success is False
    assert result.steps == 2
    assert result.max_steps_reached is True
    assert result.termination_reason == "max_steps"
    assert result.terminated is False
    assert result.truncated is False
    assert result.scenario_completed is False
    assert result.mean_speed == pytest.approx(20.0)
    assert env.close_calls == 1


@pytest.mark.parametrize(
    ("seed", "max_steps"),
    [
        (-1, 1),
        (True, 1),
        (0.5, 1),
        (0, 0),
        (0, -1),
        (0, True),
        (0, 1.5),
    ],
)
def test_invalid_episode_arguments_raise_and_still_close(
    seed: Any,
    max_steps: Any,
) -> None:
    env = _FakeEnv([])

    with pytest.raises((TypeError, ValueError)):
        run_episode(
            env,
            _FakeExecutor([_decision()]),
            _FakeScenario(),
            seed=seed,
            max_steps=max_steps,
        )

    assert env.close_calls == 1
