"""Run one auditable skill-execution episode."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
from typing import Any

import numpy as np

from ..runtime.obs_render import render
from ..runtime.types import EpisodeResult, StepRecord


_PARSE_FALLBACK = "parse_fallback"
_UNAVAILABLE_FALLBACK = "unavailable_fallback"
_LLM_ERROR_FALLBACK = "llm_error_fallback"


def _validate_arguments(seed: object, max_steps: object) -> None:
    if type(seed) is not int:
        raise TypeError("seed must be a nonnegative integer")
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(max_steps) is not int:
        raise TypeError("max_steps must be a positive integer")
    if max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _observed_speed(info: Mapping[str, Any]) -> float:
    if "speed" not in info:
        raise ValueError("step info must contain speed")
    speed = info["speed"]
    if (
        isinstance(speed, bool)
        or not isinstance(speed, Real)
        or not math.isfinite(speed)
    ):
        raise ValueError("step info speed must be a finite number")
    return float(speed)


def _validated_reward(value: object) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(
            "env.step reward must be a finite real number"
        )
    return float(value)


def _validated_bool(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be bool")
    return bool(value)


def _best_effort_env_crashed(env: Any) -> bool:
    try:
        value = getattr(env.unwrapped.vehicle, "crashed", False)
    except Exception:
        return False
    return (
        bool(value)
        if isinstance(value, (bool, np.bool_))
        else False
    )


def _validated_scenario_result(
    value: object,
    source: str,
) -> tuple[bool, str]:
    if type(value) is not tuple or len(value) != 2:
        raise TypeError(
            f"{source} must return an exact 2-tuple (bool, str)"
        )
    outcome, reason = value
    if type(outcome) is not bool or type(reason) is not str:
        raise TypeError(
            f"{source} must return an exact 2-tuple (bool, str)"
        )
    return outcome, reason


def _cleanup_error(env: Any) -> str | None:
    try:
        env.close()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _termination_reason(
    *,
    scenario_completed: bool,
    terminated: bool,
    truncated: bool,
    crashed: bool,
) -> str:
    if scenario_completed and terminated:
        return "success+terminated"
    if scenario_completed and truncated:
        return "success+truncated"
    if scenario_completed:
        return "success"
    if terminated:
        return "crashed" if crashed else "terminated"
    if truncated:
        return "truncated"
    return ""


def _scenario_value(
    scenario: Any,
    name: str,
    default: Any,
) -> Any:
    try:
        return getattr(scenario, name, default)
    except Exception:
        return default


def _episode_result(
    *,
    seed: int,
    status: str,
    success: bool,
    success_reason: str,
    episode_return: float,
    crashed: bool,
    speeds: Sequence[float],
    parse_failures: int,
    unavailable_action_attempts: int,
    llm_errors: int,
    terminated: bool,
    truncated: bool,
    max_steps_reached: bool,
    scenario_completed: bool,
    termination_reason: str,
    scenario: Any,
    records: list[StepRecord],
    exception: Exception | None = None,
) -> EpisodeResult:
    return EpisodeResult(
        seed=seed,
        status=status,
        success=success,
        success_reason=success_reason,
        episode_return=episode_return,
        crashed=crashed,
        steps=len(records),
        mean_speed=_mean(speeds),
        parse_failures=parse_failures,
        unavailable_action_attempts=unavailable_action_attempts,
        llm_errors=llm_errors,
        terminated=terminated,
        truncated=truncated,
        max_steps_reached=max_steps_reached,
        scenario_completed=scenario_completed,
        termination_reason=termination_reason,
        target_initially_ahead=bool(
            _scenario_value(
                scenario,
                "target_initially_ahead",
                False,
            )
        ),
        lane_change_completed_step=_scenario_value(
            scenario,
            "lane_change_completed_step",
            None,
        ),
        overtake_step=_scenario_value(
            scenario,
            "overtake_step",
            None,
        ),
        lead_initial_gap_m=_scenario_value(
            scenario,
            "lead_initial_gap_m",
            None,
        ),
        exception_type=type(exception).__name__ if exception else None,
        exception_message=str(exception) if exception else None,
        step_records=records,
    )


def _run_episode(
    env: Any,
    executor: Any,
    scenario: Any,
    seed: int,
    max_steps: int,
) -> EpisodeResult:
    records: list[StepRecord] = []
    speeds: list[float] = []
    episode_return = 0.0
    crashed = False
    parse_failures = 0
    unavailable_action_attempts = 0
    llm_errors = 0
    terminated = False
    truncated = False
    max_steps_reached = False
    scenario_completed = False
    termination_reason = ""
    prev_primitive: str | None = None

    try:
        feature_names = executor.grounding.observation["features"]
        obs, _ = scenario.reset(env, seed)

        for t in range(max_steps):
            available_primitives = executor.available_primitives(env)
            context = scenario.build_context(
                env,
                prev_primitive,
                available_primitives,
            )
            obs_text = render(obs, context, feature_names)
            decision = executor.decide(
                obs_text,
                context,
                available_primitives,
            )
            if decision.decision_status == _PARSE_FALLBACK:
                parse_failures += 1
            elif decision.decision_status == _UNAVAILABLE_FALLBACK:
                unavailable_action_attempts += 1
            elif decision.decision_status == _LLM_ERROR_FALLBACK:
                llm_errors += 1

            (
                obs,
                reward,
                raw_terminated,
                raw_truncated,
                info,
            ) = env.step(decision.action_index)
            record = StepRecord.from_decision(
                decision,
                t=t,
                obs_summary=obs_text,
                reward=None,
                crashed=_best_effort_env_crashed(env),
            )
            records.append(record)

            validation_error: Exception | None = None
            try:
                validated_reward = _validated_reward(reward)
            except Exception as exc:
                validation_error = exc
            else:
                record.reward = validated_reward
                episode_return += validated_reward

            try:
                terminated = _validated_bool(
                    raw_terminated,
                    "env.step terminated",
                )
            except Exception as exc:
                if validation_error is None:
                    validation_error = exc

            try:
                truncated = _validated_bool(
                    raw_truncated,
                    "env.step truncated",
                )
            except Exception as exc:
                if validation_error is None:
                    validation_error = exc

            if not isinstance(info, Mapping):
                if validation_error is None:
                    validation_error = TypeError(
                        "env.step info must be a mapping"
                    )
            else:
                if "crashed" in info:
                    try:
                        record.crashed = _validated_bool(
                            info["crashed"],
                            "step info crashed",
                        )
                    except Exception as exc:
                        if validation_error is None:
                            validation_error = exc
                try:
                    speeds.append(_observed_speed(info))
                except Exception as exc:
                    if validation_error is None:
                        validation_error = exc

            crashed = crashed or record.crashed
            if validation_error is not None:
                raise validation_error

            prev_primitive = decision.primitive
            scenario.update(env, t)
            transition_success, _ = _validated_scenario_result(
                scenario.is_terminal(env),
                "scenario.is_terminal",
            )
            scenario_completed = scenario_completed or bool(
                transition_success
            )
            termination_reason = _termination_reason(
                scenario_completed=scenario_completed,
                terminated=bool(terminated),
                truncated=bool(truncated),
                crashed=crashed,
            )
            if termination_reason:
                break
        else:
            max_steps_reached = True
            termination_reason = "max_steps"

        success, success_reason = _validated_scenario_result(
            scenario.evaluate(env),
            "scenario.evaluate",
        )
        if scenario_completed != success:
            raise ValueError(
                "scenario completion disagrees with final success"
            )
        return _episode_result(
            seed=seed,
            status="ok",
            success=bool(success),
            success_reason=str(success_reason),
            episode_return=episode_return,
            crashed=crashed,
            speeds=speeds,
            parse_failures=parse_failures,
            unavailable_action_attempts=unavailable_action_attempts,
            llm_errors=llm_errors,
            terminated=bool(terminated),
            truncated=bool(truncated),
            max_steps_reached=max_steps_reached,
            scenario_completed=scenario_completed,
            termination_reason=termination_reason,
            scenario=scenario,
            records=records,
        )
    except Exception as exc:  # noqa: BLE001 - isolate one failed episode
        return _episode_result(
            seed=seed,
            status="error",
            success=False,
            success_reason="",
            episode_return=episode_return,
            crashed=crashed,
            speeds=speeds,
            parse_failures=parse_failures,
            unavailable_action_attempts=unavailable_action_attempts,
            llm_errors=llm_errors,
            terminated=bool(terminated),
            truncated=bool(truncated),
            max_steps_reached=max_steps_reached,
            scenario_completed=scenario_completed,
            termination_reason="error",
            scenario=scenario,
            records=records,
            exception=exc,
        )


def run_episode(
    env: Any,
    executor: Any,
    scenario: Any,
    seed: int,
    max_steps: int,
) -> EpisodeResult:
    """Run one episode and close its environment exactly once."""

    try:
        _validate_arguments(seed, max_steps)
        result = _run_episode(
            env,
            executor,
            scenario,
            seed,
            max_steps,
        )
    except Exception as exc:
        cleanup_error = _cleanup_error(env)
        if cleanup_error is not None:
            exc.add_note(f"cleanup_error: {cleanup_error}")
        raise

    result.cleanup_error = _cleanup_error(env)
    return result
