"""Render validated Kinematics observations as compact deterministic text."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np

from .types import ObservationContext


_REQUIRED_FEATURES = ("presence", "x", "y", "vx", "vy")


def _measurement(value: float | None, unit: str) -> str:
    return "none" if value is None else f"{value:.1f} {unit}"


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    )


def _validate_context(context: ObservationContext) -> None:
    if not _is_finite_number(context.target_gap_m) or context.target_gap_m < 0:
        raise ValueError("target_gap_m must be finite and nonnegative")
    if not _is_finite_number(context.target_rel_speed_mps):
        raise ValueError("target_rel_speed_mps must be finite")

    for lane in ("left", "right"):
        exists = getattr(context, f"{lane}_lane_exists")
        metric_names = (
            f"{lane}_front_gap_m",
            f"{lane}_rear_gap_m",
            f"{lane}_rear_closing_speed_mps",
        )
        if not exists and any(
            getattr(context, name) is not None for name in metric_names
        ):
            raise ValueError(
                f"{lane} lane metrics must be None when {lane}_lane_exists is false"
            )

    for name in (
        "left_front_gap_m",
        "left_rear_gap_m",
        "right_front_gap_m",
        "right_rear_gap_m",
    ):
        value = getattr(context, name)
        if value is not None and (not _is_finite_number(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative")

    for name in (
        "left_rear_closing_speed_mps",
        "right_rear_closing_speed_mps",
    ):
        value = getattr(context, name)
        if value is not None and not _is_finite_number(value):
            raise ValueError(f"{name} must be finite")

    if not isinstance(context.available_primitives, list) or any(
        not isinstance(primitive, str) or not primitive.strip()
        for primitive in context.available_primitives
    ):
        raise ValueError("available_primitives must contain only non-empty strings")
    if context.prev_primitive is not None and not isinstance(
        context.prev_primitive, str
    ):
        raise ValueError("prev_primitive must be None or a string")


def _validate_observation(
    observation: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, dict[str, int]]:
    if not isinstance(observation, np.ndarray) or observation.ndim != 2:
        raise ValueError("observation must be a 2-D numpy array")
    if observation.shape[0] < 1:
        raise ValueError("observation must contain at least one row")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names must be unique")
    if observation.shape[1] != len(feature_names):
        raise ValueError(
            "observation column count must equal the number of unique feature_names"
        )

    indexes = {name: index for index, name in enumerate(feature_names)}
    missing = [name for name in _REQUIRED_FEATURES if name not in indexes]
    if missing:
        raise ValueError(f"missing required feature(s): {', '.join(missing)}")

    try:
        values = observation.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("observation values must be numeric") from exc

    presence = values[:, indexes["presence"]]
    if not np.isfinite(presence).all():
        raise ValueError("presence values must be finite")
    if not np.isin(presence, (0.0, 1.0)).all():
        raise ValueError("presence values must be exactly 0 or 1")
    if presence[0] <= 0:
        raise ValueError("ego presence must be positive")
    if not np.isfinite(values[presence == 1]).all():
        raise ValueError("present rows must contain only finite values")

    return values, indexes


def render(
    observation: np.ndarray,
    context: ObservationContext,
    feature_names: list[str],
) -> str:
    """Return an ego-relative observation and scenario summary for the LLM."""

    values, indexes = _validate_observation(observation, feature_names)
    _validate_context(context)
    ego = values[0]
    ego_speed = math.hypot(ego[indexes["vx"]], ego[indexes["vy"]])

    neighbors = [row for row in values[1:] if row[indexes["presence"]] == 1]
    lines = [
        f"Ego: position=(0.0, 0.0) lane={context.ego_lane} speed={ego_speed:.1f} m/s",
        f"Neighbors: count={len(neighbors)} frame=ego-relative",
    ]
    for number, row in enumerate(neighbors, start=1):
        lines.append(
            f"  #{number} dx={row[indexes['x']]:.1f} m "
            f"dy={row[indexes['y']]:.1f} m "
            f"dvx={row[indexes['vx']]:.1f} m/s "
            f"dvy={row[indexes['vy']]:.1f} m/s"
        )

    target_relation = "ahead" if context.target_ahead else "behind"
    previous_primitive = context.prev_primitive or "none"
    available_primitives = ", ".join(context.available_primitives) or "none"
    lines.extend(
        [
            f"Target: relation={target_relation} gap={context.target_gap_m:.1f} m "
            f"rel_speed={context.target_rel_speed_mps:.1f} m/s",
            "Sign conventions: target rel_speed = target_speed - ego_speed; "
            "negative means target slower. rear_closing_speed = "
            "rear_speed - ego_speed; positive means rear closing.",
            f"Left lane: exists={str(context.left_lane_exists).lower()} "
            f"front_gap={_measurement(context.left_front_gap_m, 'm')} "
            f"rear_gap={_measurement(context.left_rear_gap_m, 'm')} "
            "rear_closing_speed="
            f"{_measurement(context.left_rear_closing_speed_mps, 'm/s')}",
            f"Right lane: exists={str(context.right_lane_exists).lower()} "
            f"front_gap={_measurement(context.right_front_gap_m, 'm')} "
            f"rear_gap={_measurement(context.right_rear_gap_m, 'm')} "
            "rear_closing_speed="
            f"{_measurement(context.right_rear_closing_speed_mps, 'm/s')}",
            f"Previous primitive: {previous_primitive}",
            f"Available primitives: {available_primitives}",
        ]
    )
    return "\n".join(lines)
