from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from importlib.metadata import version
from typing import Any

from gymnasium.spaces import Discrete
import numpy as np
from packaging.specifiers import SpecifierSet
from packaging.version import Version
import pytest

pytest.importorskip("highway_env")

from world2skills.runtime.env_factory import make_env
from world2skills.runtime.skill_loader import load_skill, select_grounding
from world2skills.runtime.types import Grounding


@pytest.fixture
def grounding() -> Grounding:
    return deepcopy(
        select_grounding(load_skill("lane-change-overtake"), "highway-env")
    )


@pytest.fixture
def scenario_config() -> dict[str, Any]:
    return {
        "lanes_count": 4,
        "vehicles_count": 10,
        "duration": 40,
        "policy_frequency": 1,
        "simulation_frequency": 15,
        "obs_vehicles_count": 8,
    }


def test_make_env_returns_resolved_kinematics_config_and_action_map(
    grounding: Grounding,
    scenario_config: dict[str, Any],
) -> None:
    env, name_to_index = make_env(grounding, scenario_config, seed=0)
    try:
        observation_config = env.unwrapped.config["observation"]
        action_config = env.unwrapped.config["action"]
        obs = env.unwrapped.observation_type.observe()

        assert env.render_mode is None
        assert observation_config == {
            "type": "Kinematics",
            "features": grounding.observation["features"],
            "vehicles_count": 8,
            "normalize": False,
            "absolute": False,
            "see_behind": True,
            "order": "sorted",
        }
        assert action_config == {"type": "DiscreteMetaAction"}
        for key in (
            "lanes_count",
            "vehicles_count",
            "duration",
            "policy_frequency",
            "simulation_frequency",
        ):
            assert env.unwrapped.config[key] == scenario_config[key]

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (8, len(grounding.observation["features"]))
        assert isinstance(env.action_space, Discrete)
        assert env.action_space.n == 5
        assert name_to_index == {
            label: index
            for index, label in env.unwrapped.action_type.actions.items()
        }
        assert name_to_index == {
            "LANE_LEFT": 0,
            "IDLE": 1,
            "LANE_RIGHT": 2,
            "FASTER": 3,
            "SLOWER": 4,
        }

        installed = Version(version("highway-env"))
        assert installed in SpecifierSet(grounding.backend_version)
    finally:
        env.close()


def test_same_seed_produces_same_initial_observation_and_action_samples(
    grounding: Grounding,
    scenario_config: dict[str, Any],
) -> None:
    env_a, _ = make_env(grounding, scenario_config, seed=17)
    env_b, _ = make_env(grounding, scenario_config, seed=17)
    try:
        obs_a = env_a.unwrapped.observation_type.observe()
        obs_b = env_b.unwrapped.observation_type.observe()
        np.testing.assert_array_equal(obs_a, obs_b)
        assert [env_a.action_space.sample() for _ in range(5)] == [
            env_b.action_space.sample() for _ in range(5)
        ]
    finally:
        env_a.close()
        env_b.close()


def test_make_env_does_not_mutate_inputs(
    grounding: Grounding,
    scenario_config: dict[str, Any],
) -> None:
    grounding_before = deepcopy(grounding)
    config_before = deepcopy(scenario_config)

    env, _ = make_env(grounding, scenario_config, seed=0)
    try:
        assert grounding == grounding_before
        assert scenario_config == config_before
    finally:
        env.close()


@pytest.mark.parametrize(
    ("updated_grounding", "message"),
    [
        ({"backend": "other-backend"}, "backend"),
        ({"action": {"type": "ContinuousAction"}}, "DiscreteMetaAction"),
        (
            {
                "observation": {
                    "type": "OccupancyGrid",
                    "features": ["presence", "x"],
                }
            },
            "Kinematics",
        ),
        ({"backend_version": ">=999"}, "does not satisfy"),
        ({"backend_version": "not a specifier"}, "invalid backend version"),
    ],
)
def test_make_env_rejects_invalid_grounding_contract(
    grounding: Grounding,
    scenario_config: dict[str, Any],
    updated_grounding: dict[str, Any],
    message: str,
) -> None:
    invalid = replace(grounding, **updated_grounding)

    with pytest.raises(ValueError, match=message):
        make_env(invalid, scenario_config, seed=0)


@pytest.mark.parametrize("backend_version", ["", "   ", None, 1])
def test_make_env_requires_non_empty_string_backend_version(
    grounding: Grounding,
    scenario_config: dict[str, Any],
    backend_version: object,
) -> None:
    invalid = replace(grounding, backend_version=backend_version)

    with pytest.raises(ValueError, match="non-empty string"):
        make_env(invalid, scenario_config, seed=0)


@pytest.mark.parametrize("missing_feature", ["presence", "x", "y", "vx", "vy"])
def test_make_env_requires_all_renderer_kinematics_features(
    grounding: Grounding,
    scenario_config: dict[str, Any],
    missing_feature: str,
) -> None:
    features = [
        feature
        for feature in grounding.observation["features"]
        if feature != missing_feature
    ]
    invalid = replace(
        grounding,
        observation={**grounding.observation, "features": features},
    )

    with pytest.raises(ValueError, match=missing_feature):
        make_env(invalid, scenario_config, seed=0)


@pytest.mark.parametrize("value", [0, -1, 1.5, True, None, "8"])
def test_make_env_rejects_non_positive_integer_observation_count(
    grounding: Grounding,
    scenario_config: dict[str, Any],
    value: object,
) -> None:
    scenario_config["obs_vehicles_count"] = value

    with pytest.raises(ValueError, match="obs_vehicles_count"):
        make_env(grounding, scenario_config, seed=0)


def test_make_env_rejects_missing_backend_action_label(
    grounding: Grounding,
    scenario_config: dict[str, Any],
) -> None:
    grounding.primitive_map["accelerate"] = "NOT_A_HIGHWAY_ACTION"

    with pytest.raises(ValueError, match="NOT_A_HIGHWAY_ACTION"):
        make_env(grounding, scenario_config, seed=0)


def test_make_env_closes_created_env_when_initialization_fails(
    monkeypatch: pytest.MonkeyPatch,
    grounding: Grounding,
    scenario_config: dict[str, Any],
) -> None:
    import gymnasium

    class FailingEnv:
        def __init__(self) -> None:
            self.closed = False

        def reset(self, *, seed: int) -> None:
            raise RuntimeError(f"reset failed for seed {seed}")

        def close(self) -> None:
            self.closed = True

    fake_env = FailingEnv()
    make_kwargs: dict[str, Any] = {}

    def fake_make(environment: str, **kwargs: Any) -> FailingEnv:
        assert environment == grounding.environment
        make_kwargs.update(kwargs)
        return fake_env

    monkeypatch.setattr(gymnasium, "make", fake_make)

    with pytest.raises(RuntimeError, match="reset failed"):
        make_env(grounding, scenario_config, seed=23)

    assert fake_env.closed is True
    assert make_kwargs["render_mode"] is None
    assert make_kwargs["config"]["observation"]["normalize"] is False
    assert make_kwargs["config"]["action"] == {"type": "DiscreteMetaAction"}
