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

from world2skills.runtime.env_factory import (
    build_custom_env_config,
    make_env,
    resolve_env_config,
)
from world2skills.runtime.skill_loader import load_skill, select_grounding
from world2skills.runtime.types import Grounding


@pytest.fixture
def grounding() -> Grounding:
    return deepcopy(select_grounding(load_skill("lane-change-overtake"), "highway-env"))


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


class _TrackingEnv:
    def __init__(
        self,
        reset_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.unwrapped = self
        self.render_mode = None
        self.closed = False
        self.configured_with: dict[str, Any] | None = None
        self.reset_error = reset_error
        self.close_error = close_error

    def default_config(self) -> dict[str, Any]:
        return {
            "observation": {},
            "action": {},
            "lanes_count": 4,
            "vehicles_count": 50,
            "duration": 40,
            "policy_frequency": 1,
            "simulation_frequency": 15,
            "controlled_vehicles": 1,
        }

    def configure(self, config: dict[str, Any]) -> None:
        self.configured_with = deepcopy(config)

    def reset(self, *, seed: int) -> None:
        if self.reset_error is not None:
            raise self.reset_error
        raise AssertionError(f"reset should not be called for seed {seed}")

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


def _patch_make(
    monkeypatch: pytest.MonkeyPatch,
    grounding: Grounding,
    *,
    reset_error: Exception | None = None,
    close_error: Exception | None = None,
) -> tuple[_TrackingEnv, dict[str, Any]]:
    import gymnasium

    fake_env = _TrackingEnv(reset_error, close_error)
    make_kwargs: dict[str, Any] = {}

    def fake_make(environment: str, **kwargs: Any) -> _TrackingEnv:
        assert environment == grounding.environment
        make_kwargs.update(kwargs)
        return fake_env

    monkeypatch.setattr(gymnasium, "make", fake_make)
    return fake_env, make_kwargs


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
            label: index for index, label in env.unwrapped.action_type.actions.items()
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


def test_resolve_env_config_equals_real_make_env_config(
    grounding: Grounding,
    scenario_config: dict[str, Any],
) -> None:
    config_before = deepcopy(scenario_config)

    resolved = resolve_env_config(grounding, scenario_config)
    env, _ = make_env(grounding, scenario_config, seed=0)
    try:
        assert resolved == env.unwrapped.config
        assert resolved is not env.unwrapped.config
        assert (
            resolved["controlled_vehicles"]
            == (env.unwrapped.default_config()["controlled_vehicles"])
        )
        assert scenario_config == config_before
    finally:
        env.close()


def test_build_custom_env_config_is_pure_and_forces_backend_settings(
    grounding: Grounding,
    scenario_config: dict[str, Any],
) -> None:
    config_before = deepcopy(scenario_config)

    custom = build_custom_env_config(grounding, scenario_config)

    assert scenario_config == config_before
    assert "obs_vehicles_count" not in custom
    assert "controlled_vehicles" not in custom
    assert custom["observation"] == {
        "type": "Kinematics",
        "features": grounding.observation["features"],
        "vehicles_count": 8,
        "normalize": False,
        "absolute": False,
        "see_behind": True,
        "order": "sorted",
    }
    assert custom["action"] == {"type": "DiscreteMetaAction"}
    for key in (
        "lanes_count",
        "vehicles_count",
        "duration",
        "policy_frequency",
        "simulation_frequency",
    ):
        assert custom[key] == scenario_config[key]


def test_resolve_env_config_always_closes_probe_env(
    monkeypatch: pytest.MonkeyPatch,
    grounding: Grounding,
    scenario_config: dict[str, Any],
) -> None:
    fake_env, make_kwargs = _patch_make(monkeypatch, grounding)

    resolved = resolve_env_config(grounding, scenario_config)

    assert fake_env.closed is True
    assert fake_env.configured_with is None
    assert make_kwargs == {"render_mode": None}
    assert resolved["controlled_vehicles"] == 1
    assert resolved["vehicles_count"] == scenario_config["vehicles_count"]
    assert resolved["observation"]["vehicles_count"] == 8


def test_resolve_env_config_closes_probe_when_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
    grounding: Grounding,
    scenario_config: dict[str, Any],
) -> None:
    scenario_config["vehcles_count"] = 10
    fake_env, _ = _patch_make(monkeypatch, grounding)

    with pytest.raises(ValueError, match="vehcles_count"):
        resolve_env_config(grounding, scenario_config)

    assert fake_env.closed is True
    assert fake_env.configured_with is None


def test_resolve_env_config_preserves_primary_error_and_adds_cleanup_note(
    monkeypatch: pytest.MonkeyPatch,
    grounding: Grounding,
    scenario_config: dict[str, Any],
) -> None:
    scenario_config["vehcles_count"] = 10
    fake_env, _ = _patch_make(
        monkeypatch,
        grounding,
        close_error=RuntimeError("probe close failed"),
    )

    with pytest.raises(ValueError, match="vehcles_count") as captured:
        resolve_env_config(grounding, scenario_config)

    assert fake_env.closed is True
    assert captured.value.__notes__ == [
        "cleanup_error: RuntimeError: probe close failed"
    ]


def test_resolve_env_config_raises_close_error_after_success(
    monkeypatch: pytest.MonkeyPatch,
    grounding: Grounding,
    scenario_config: dict[str, Any],
) -> None:
    fake_env, _ = _patch_make(
        monkeypatch,
        grounding,
        close_error=RuntimeError("probe close failed"),
    )

    with pytest.raises(RuntimeError, match="probe close failed") as captured:
        resolve_env_config(grounding, scenario_config)

    assert fake_env.closed is True
    assert getattr(captured.value, "__notes__", []) == []


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
    monkeypatch: pytest.MonkeyPatch,
    grounding: Grounding,
    scenario_config: dict[str, Any],
    value: object,
) -> None:
    scenario_config["obs_vehicles_count"] = value
    fake_env, _ = _patch_make(monkeypatch, grounding)

    with pytest.raises(ValueError, match="obs_vehicles_count"):
        make_env(grounding, scenario_config, seed=0)

    assert fake_env.closed is True
    assert fake_env.configured_with is None


def test_make_env_rejects_unknown_scenario_key_and_closes_env(
    monkeypatch: pytest.MonkeyPatch,
    grounding: Grounding,
    scenario_config: dict[str, Any],
) -> None:
    scenario_config["vehcles_count"] = scenario_config["vehicles_count"]
    config_before = deepcopy(scenario_config)
    fake_env, _ = _patch_make(monkeypatch, grounding)

    with pytest.raises(ValueError, match="vehcles_count"):
        make_env(grounding, scenario_config, seed=0)

    assert fake_env.closed is True
    assert fake_env.configured_with is None
    assert scenario_config == config_before


@pytest.mark.parametrize("reserved_key", ["observation", "action"])
def test_make_env_rejects_caller_supplied_reserved_config_and_closes_env(
    monkeypatch: pytest.MonkeyPatch,
    grounding: Grounding,
    scenario_config: dict[str, Any],
    reserved_key: str,
) -> None:
    scenario_config[reserved_key] = {"type": "CallerOverride"}
    fake_env, _ = _patch_make(monkeypatch, grounding)

    with pytest.raises(ValueError, match=reserved_key):
        make_env(grounding, scenario_config, seed=0)

    assert fake_env.closed is True
    assert fake_env.configured_with is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lanes_count", 0),
        ("lanes_count", -1),
        ("lanes_count", True),
        ("lanes_count", 2.0),
        ("vehicles_count", -1),
        ("vehicles_count", True),
        ("vehicles_count", 1.0),
        ("duration", 0),
        ("duration", -1),
        ("duration", True),
        ("duration", float("nan")),
        ("duration", float("inf")),
        ("policy_frequency", 0),
        ("policy_frequency", -1),
        ("policy_frequency", True),
        ("policy_frequency", float("nan")),
        ("simulation_frequency", 0),
        ("simulation_frequency", -1),
        ("simulation_frequency", True),
        ("simulation_frequency", float("inf")),
        ("controlled_vehicles", 0),
        ("controlled_vehicles", -1),
        ("controlled_vehicles", True),
        ("controlled_vehicles", 1.0),
    ],
)
def test_make_env_rejects_invalid_core_scenario_value_and_closes_env(
    monkeypatch: pytest.MonkeyPatch,
    grounding: Grounding,
    scenario_config: dict[str, Any],
    field: str,
    value: object,
) -> None:
    scenario_config[field] = value
    fake_env, _ = _patch_make(monkeypatch, grounding)

    with pytest.raises(ValueError, match=field):
        make_env(grounding, scenario_config, seed=0)

    assert fake_env.closed is True
    assert fake_env.configured_with is None


def test_make_env_allows_single_lane_and_retains_config(
    grounding: Grounding,
    scenario_config: dict[str, Any],
) -> None:
    scenario_config["lanes_count"] = 1

    env, _ = make_env(grounding, scenario_config, seed=0)
    try:
        assert env.unwrapped.config["lanes_count"] == 1
    finally:
        env.close()


def test_make_env_rejects_simulation_frequency_below_policy_frequency(
    monkeypatch: pytest.MonkeyPatch,
    grounding: Grounding,
    scenario_config: dict[str, Any],
) -> None:
    scenario_config["policy_frequency"] = 10
    scenario_config["simulation_frequency"] = 5
    fake_env, _ = _patch_make(monkeypatch, grounding)

    with pytest.raises(
        ValueError,
        match="simulation_frequency.*policy_frequency",
    ):
        make_env(grounding, scenario_config, seed=0)

    assert fake_env.closed is True
    assert fake_env.configured_with is None


def test_make_env_allows_zero_other_vehicles(
    grounding: Grounding,
    scenario_config: dict[str, Any],
) -> None:
    scenario_config["vehicles_count"] = 0

    env, _ = make_env(grounding, scenario_config, seed=0)
    env.close()


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
    fake_env, make_kwargs = _patch_make(
        monkeypatch,
        grounding,
        reset_error=RuntimeError("reset failed"),
    )

    with pytest.raises(RuntimeError, match="reset failed"):
        make_env(grounding, scenario_config, seed=23)

    assert fake_env.closed is True
    assert make_kwargs == {"render_mode": None}
    assert fake_env.configured_with is not None
    assert fake_env.configured_with["observation"]["normalize"] is False
    assert fake_env.configured_with["action"] == {"type": "DiscreteMetaAction"}


def test_make_env_preserves_primary_error_and_adds_cleanup_note(
    monkeypatch: pytest.MonkeyPatch,
    grounding: Grounding,
    scenario_config: dict[str, Any],
) -> None:
    fake_env, _ = _patch_make(
        monkeypatch,
        grounding,
        reset_error=RuntimeError("reset failed"),
        close_error=RuntimeError("env close failed"),
    )

    with pytest.raises(RuntimeError, match="reset failed") as captured:
        make_env(grounding, scenario_config, seed=23)

    assert fake_env.closed is True
    assert captured.value.__notes__ == ["cleanup_error: RuntimeError: env close failed"]
