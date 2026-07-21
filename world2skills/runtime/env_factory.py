"""Create and validate highway-env environments from skill groundings."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from importlib import metadata
import math
from numbers import Real
from typing import Any

import numpy as np
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from .types import Grounding


_BACKEND = "highway-env"
_ACTION_TYPE = "DiscreteMetaAction"
_OBSERVATION_TYPE = "Kinematics"
_DEFAULT_OBS_VEHICLES_COUNT = 8
_RENDERER_REQUIRED_FEATURES = ("presence", "x", "y", "vx", "vy")
_RESERVED_SCENARIO_KEYS = frozenset({"observation", "action"})
_POSITIVE_NUMBER_FIELDS = (
    "duration",
    "policy_frequency",
    "simulation_frequency",
)


def _validate_grounding(grounding: Grounding) -> list[str]:
    if grounding.backend != _BACKEND:
        raise ValueError(
            f"grounding backend must be {_BACKEND!r}, got {grounding.backend!r}"
        )
    if grounding.action.get("type") != _ACTION_TYPE:
        raise ValueError(
            f"grounding action type must be {_ACTION_TYPE!r}, "
            f"got {grounding.action.get('type')!r}"
        )
    if grounding.observation.get("type") != _OBSERVATION_TYPE:
        raise ValueError(
            f"grounding observation type must be {_OBSERVATION_TYPE!r}, "
            f"got {grounding.observation.get('type')!r}"
        )

    features = grounding.observation.get("features")
    if (
        not isinstance(features, list)
        or not features
        or any(not isinstance(feature, str) or not feature for feature in features)
        or len(set(features)) != len(features)
    ):
        raise ValueError(
            "grounding Kinematics features must be a non-empty list of "
            "unique, non-empty strings"
        )

    missing_features = [
        feature
        for feature in _RENDERER_REQUIRED_FEATURES
        if feature not in features
    ]
    if missing_features:
        raise ValueError(
            "grounding Kinematics features are missing renderer-required "
            f"features: {missing_features}"
        )

    if (
        not isinstance(grounding.backend_version, str)
        or not grounding.backend_version.strip()
    ):
        raise ValueError("grounding backend_version must be a non-empty string")

    try:
        specifier = SpecifierSet(grounding.backend_version)
    except InvalidSpecifier as exc:
        raise ValueError(
            f"invalid backend version specifier: {grounding.backend_version!r}"
        ) from exc

    installed_text = metadata.version(_BACKEND)
    try:
        installed = Version(installed_text)
    except InvalidVersion as exc:
        raise ValueError(
            f"installed {_BACKEND} version is invalid: {installed_text!r}"
        ) from exc
    if installed not in specifier:
        raise ValueError(
            f"installed {_BACKEND} {installed} does not satisfy "
            f"{grounding.backend_version!r}"
        )

    return features.copy()


def _resolved_config(
    scenario_config: dict[str, Any],
    features: list[str],
    default_config: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(scenario_config, dict):
        raise TypeError("scenario_config must be a dict")

    config = deepcopy(scenario_config)
    reserved = sorted(_RESERVED_SCENARIO_KEYS.intersection(config))
    if reserved:
        raise ValueError(
            f"scenario_config cannot override reserved keys: {reserved}"
        )

    obs_vehicles_count = config.pop(
        "obs_vehicles_count",
        _DEFAULT_OBS_VEHICLES_COUNT,
    )
    if type(obs_vehicles_count) is not int or obs_vehicles_count <= 0:
        raise ValueError("obs_vehicles_count must be a positive integer")

    unknown = sorted(set(config) - set(default_config))
    if unknown:
        raise ValueError(f"unknown scenario_config keys: {unknown}")

    _validate_core_scenario_values(config, default_config)

    config["observation"] = {
        "type": _OBSERVATION_TYPE,
        "features": features,
        "vehicles_count": obs_vehicles_count,
        "normalize": False,
        "absolute": False,
        "see_behind": True,
        "order": "sorted",
    }
    config["action"] = {"type": _ACTION_TYPE}
    return config


def _complete_resolved_config(
    env: Any,
    scenario_config: dict[str, Any],
    features: list[str],
) -> dict[str, Any]:
    default_config = env.unwrapped.default_config()
    if not isinstance(default_config, Mapping):
        raise ValueError("highway-env default_config() must return a mapping")
    resolved = dict(deepcopy(default_config))
    resolved.update(
        _resolved_config(
            scenario_config,
            features,
            default_config,
        )
    )
    if resolved.get("offscreen_rendering") is None:
        resolved["offscreen_rendering"] = env.render_mode != "human"
    return resolved


def _make_unconfigured_env(grounding: Grounding) -> Any:
    import gymnasium
    import highway_env  # noqa: F401 - import registers highway-env IDs

    return gymnasium.make(
        grounding.environment,
        render_mode=None,
    )


def _validate_core_scenario_values(
    config: Mapping[str, Any],
    default_config: Mapping[str, Any],
) -> None:
    if "lanes_count" in config:
        lanes_count = config["lanes_count"]
        if type(lanes_count) is not int or lanes_count <= 0:
            raise ValueError("lanes_count must be a positive integer")

    if "vehicles_count" in config:
        vehicles_count = config["vehicles_count"]
        if type(vehicles_count) is not int or vehicles_count < 0:
            raise ValueError("vehicles_count must be a nonnegative integer")

    if "controlled_vehicles" in config:
        controlled_vehicles = config["controlled_vehicles"]
        if type(controlled_vehicles) is not int or controlled_vehicles <= 0:
            raise ValueError("controlled_vehicles must be a positive integer")

    for field in _POSITIVE_NUMBER_FIELDS:
        if field not in config:
            continue
        value = config[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{field} must be a positive finite number")

    policy_frequency = config.get(
        "policy_frequency",
        default_config.get("policy_frequency"),
    )
    simulation_frequency = config.get(
        "simulation_frequency",
        default_config.get("simulation_frequency"),
    )
    if (
        policy_frequency is not None
        and simulation_frequency is not None
        and simulation_frequency < policy_frequency
    ):
        raise ValueError(
            "simulation_frequency must be greater than or equal to "
            "policy_frequency"
        )


def _invert_action_map(env: Any) -> dict[str, int]:
    import gymnasium

    if not isinstance(env.action_space, gymnasium.spaces.Discrete):
        raise ValueError(
            "highway-env action space must be gymnasium.spaces.Discrete"
        )

    actions = getattr(env.unwrapped.action_type, "actions", None)
    if not isinstance(actions, Mapping) or not actions:
        raise ValueError(
            "DiscreteMetaAction.actions must be a non-empty index-to-label mapping"
        )

    labels: list[str] = []
    indexes: list[int] = []
    for index, label in actions.items():
        if type(index) is not int:
            raise ValueError(
                "DiscreteMetaAction.actions indexes must be integers"
            )
        if not isinstance(label, str) or not label.strip():
            raise ValueError(
                "DiscreteMetaAction.actions labels must be non-empty strings"
            )
        indexes.append(index)
        labels.append(label)

    if len(set(labels)) != len(labels):
        raise ValueError("DiscreteMetaAction.actions labels must be unique")
    expected_indexes = set(range(env.action_space.n))
    if set(indexes) != expected_indexes:
        raise ValueError(
            "DiscreteMetaAction.actions indexes must exactly cover the "
            "discrete action space"
        )

    return {label: index for index, label in actions.items()}


def _validate_backend_actions(
    grounding: Grounding,
    name_to_index: Mapping[str, int],
) -> None:
    backend_labels = list(grounding.primitive_map.values())
    if any(
        not isinstance(label, str) or not label.strip()
        for label in backend_labels
    ):
        raise ValueError(
            "grounding primitive_map backend labels must be non-empty strings"
        )

    missing = sorted(set(backend_labels) - set(name_to_index))
    if missing:
        raise ValueError(
            "grounding primitive_map references unavailable backend action "
            f"labels: {missing}"
        )


def resolve_env_config(
    grounding: Grounding,
    scenario_config: dict[str, Any],
) -> dict[str, Any]:
    """Return the complete validated config without retaining a probe env."""

    features = _validate_grounding(grounding)
    env = None
    try:
        env = _make_unconfigured_env(grounding)
        return _complete_resolved_config(
            env,
            scenario_config,
            features,
        )
    finally:
        if env is not None:
            env.close()


def make_env(
    grounding: Grounding,
    scenario_config: dict[str, Any],
    seed: int,
) -> tuple[Any, dict[str, int]]:
    """Return a configured, reset environment and its action-name index map."""

    features = _validate_grounding(grounding)

    env = None
    try:
        env = _make_unconfigured_env(grounding)
        config = _complete_resolved_config(
            env,
            scenario_config,
            features,
        )
        env.unwrapped.configure(config)
        obs, _ = env.reset(seed=seed)
        # This seeds only the action space created by the factory's initial reset.
        env.action_space.seed(seed)

        expected_shape = (
            config["observation"]["vehicles_count"],
            len(features),
        )
        if not isinstance(obs, np.ndarray):
            raise ValueError("Kinematics reset observation must be a numpy ndarray")
        if obs.shape != expected_shape:
            raise ValueError(
                "Kinematics reset observation has shape "
                f"{obs.shape}, expected {expected_shape}"
            )

        name_to_index = _invert_action_map(env)
        _validate_backend_actions(grounding, name_to_index)
        return env, name_to_index
    except Exception:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        raise
