"""Create and validate highway-env environments from skill groundings."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from importlib import metadata
from typing import Any

import numpy as np
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from .types import Grounding


_BACKEND = "highway-env"
_ACTION_TYPE = "DiscreteMetaAction"
_OBSERVATION_TYPE = "Kinematics"
_DEFAULT_OBS_VEHICLES_COUNT = 8


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
) -> dict[str, Any]:
    if not isinstance(scenario_config, dict):
        raise TypeError("scenario_config must be a dict")

    config = deepcopy(scenario_config)
    obs_vehicles_count = config.pop(
        "obs_vehicles_count",
        _DEFAULT_OBS_VEHICLES_COUNT,
    )
    if type(obs_vehicles_count) is not int or obs_vehicles_count <= 0:
        raise ValueError("obs_vehicles_count must be a positive integer")

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


def make_env(
    grounding: Grounding,
    scenario_config: dict[str, Any],
    seed: int,
) -> tuple[Any, dict[str, int]]:
    """Return a configured, reset environment and its action-name index map."""

    features = _validate_grounding(grounding)
    config = _resolved_config(scenario_config, features)

    import gymnasium
    import highway_env  # noqa: F401 - import registers highway-env IDs

    env = None
    try:
        env = gymnasium.make(
            grounding.environment,
            config=config,
            render_mode=None,
        )
        obs, _ = env.reset(seed=seed)
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
