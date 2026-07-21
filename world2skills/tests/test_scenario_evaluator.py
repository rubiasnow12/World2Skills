from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

from gymnasium.spaces import Discrete
import numpy as np
import pytest

pytest.importorskip("highway_env")

from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle

from world2skills.runtime.env_factory import make_env
from world2skills.runtime.scenario import (
    LaneChangeOvertakeScenario,
    ScenarioSetupError,
    judge_success,
)
from world2skills.runtime.skill_loader import load_skill, select_grounding


def _make_scenario_env(
    seed: int,
    **scenario_kwargs: float,
) -> tuple[LaneChangeOvertakeScenario, object, dict[str, int]]:
    card = load_skill("lane-change-overtake")
    grounding = select_grounding(card, "highway-env")
    scenario = LaneChangeOvertakeScenario(card, **scenario_kwargs)
    env, name_to_index = make_env(grounding, scenario.configure(), seed)
    return scenario, env, name_to_index


def _longitudinal(lane: object, position: object) -> float:
    return float(lane.local_coordinates(position)[0])


def _geometry_snapshot(env: object) -> tuple[object, ...]:
    u = env.unwrapped
    rows = []
    for vehicle in u.road.vehicles:
        lane = u.road.network.get_lane(vehicle.lane_index)
        rows.append(
            (
                type(vehicle).__name__,
                vehicle.lane_index,
                round(_longitudinal(lane, vehicle.position), 8),
                round(float(vehicle.speed), 8),
                bool(getattr(vehicle, "enable_lane_change", False)),
                vehicle is u.vehicle,
            )
        )
    return tuple(sorted(rows, key=repr))


def test_judge_success_requires_lane_change_before_overtake() -> None:
    success, reason = judge_success(
        target_initially_ahead=True,
        lane_change_completed_step=5,
        overtake_step=9,
        collision=False,
    )

    assert success is True
    assert reason == "lane change at step 5, then overtook at step 9"


def test_judge_success_rejects_collision() -> None:
    success, reason = judge_success(True, 5, 9, collision=True)

    assert success is False
    assert reason == "collision"


def test_judge_success_requires_target_initially_ahead() -> None:
    success, reason = judge_success(False, 5, 9, collision=False)

    assert success is False
    assert reason == "target was not initially ahead"


def test_judge_success_requires_completed_lane_change() -> None:
    success, reason = judge_success(True, None, 9, collision=False)

    assert success is False
    assert reason == "lane change not completed"


def test_judge_success_requires_overtake() -> None:
    success, reason = judge_success(True, 5, None, collision=False)

    assert success is False
    assert reason == "target not overtaken"


@pytest.mark.parametrize(
    ("lane_change_step", "overtake_step"),
    [(9, 5), (5, 5)],
)
def test_judge_success_requires_strict_causal_order(
    lane_change_step: int,
    overtake_step: int,
) -> None:
    success, reason = judge_success(
        True,
        lane_change_step,
        overtake_step,
        collision=False,
    )

    assert success is False
    assert reason == "overtake did not occur after lane change"


def test_configure_and_snapshot_expose_resolved_m1_parameters() -> None:
    card = load_skill("lane-change-overtake")
    scenario = LaneChangeOvertakeScenario(
        card,
        success_margin=7.0,
        lead_speed_ratio=0.5,
        target_lane_front_clearance_m=24.0,
        target_lane_rear_clearance_m=22.0,
        spawn_gap=36.0,
    )

    assert scenario.configure() == {
        "lanes_count": 4,
        "vehicles_count": 30,
        "controlled_vehicles": 1,
        "initial_lane_id": 1,
        "duration": 40,
        "policy_frequency": 1,
        "simulation_frequency": 15,
        "obs_vehicles_count": 8,
    }
    assert scenario.config_snapshot() == {
        "success_margin": 7.0,
        "lead_speed_ratio": 0.5,
        "target_lane_front_clearance_m": 24.0,
        "target_lane_rear_clearance_m": 22.0,
        "spawn_gap": 36.0,
        "min_lane_gap_m": 20.0,
        "minimum_spawn_clearance_m": 25.0,
        "target_speed_mps": 25.0,
        "environment_config": scenario.configure(),
    }


@pytest.mark.parametrize(
    "value",
    [0, -1, None, True, "20", float("nan"), float("inf")],
)
def test_scenario_requires_positive_skill_min_lane_gap(value: object) -> None:
    card = deepcopy(load_skill("lane-change-overtake"))
    card.parameters["min_lane_gap"]["default"] = value

    with pytest.raises(ValueError, match="min_lane_gap.default"):
        LaneChangeOvertakeScenario(card)


def test_scenario_rejects_spawn_gap_below_minimum_bumper_clearance() -> None:
    card = load_skill("lane-change-overtake")
    minimum = 20.0 + Vehicle.LENGTH

    with pytest.raises(ValueError, match=r"spawn_gap.*25\.0"):
        LaneChangeOvertakeScenario(card, spawn_gap=minimum - 0.01)


def test_scenario_accepts_exact_minimum_bumper_clearance() -> None:
    card = load_skill("lane-change-overtake")
    minimum = 20.0 + Vehicle.LENGTH

    scenario = LaneChangeOvertakeScenario(card, spawn_gap=minimum)

    assert scenario.min_lane_gap_m == 20.0
    assert scenario.minimum_spawn_clearance_m == 25.0
    assert scenario.spawn_gap == scenario.minimum_spawn_clearance_m


def test_reset_revalidates_spawn_gap_with_actual_vehicle_lengths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = load_skill("lane-change-overtake")
    minimum = 20.0 + Vehicle.LENGTH
    scenario = LaneChangeOvertakeScenario(card, spawn_gap=minimum)
    grounding = select_grounding(card, "highway-env")
    env, _ = make_env(grounding, scenario.configure(), seed=0)
    monkeypatch.setattr(IDMVehicle, "LENGTH", 9.0)
    try:
        with pytest.raises(
            ScenarioSetupError,
            match="actual vehicle lengths",
        ):
            scenario.reset(env, seed=0)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"success_margin": 0.0}, "success_margin"),
        ({"lead_speed_ratio": 0.0}, "lead_speed_ratio"),
        ({"lead_speed_ratio": 1.0}, "lead_speed_ratio"),
        ({"target_lane_front_clearance_m": -1.0}, "front_clearance"),
        ({"target_lane_rear_clearance_m": -1.0}, "rear_clearance"),
        ({"spawn_gap": 0.0}, "spawn_gap"),
    ],
)
def test_scenario_rejects_invalid_parameters(
    kwargs: dict[str, float],
    message: str,
) -> None:
    card = load_skill("lane-change-overtake")

    with pytest.raises(ValueError, match=message):
        LaneChangeOvertakeScenario(card, **kwargs)


@pytest.mark.parametrize("seed", range(5))
def test_reset_establishes_exact_comparable_geometry_and_reseeds_actions(
    seed: int,
) -> None:
    scenario, env, _ = _make_scenario_env(seed)
    try:
        obs, info = scenario.reset(env, seed)
        u = env.unwrapped
        ego = u.vehicle
        initial_lane = u.road.network.get_lane(scenario.initial_ego_lane)
        target_lane = u.road.network.get_lane(scenario.target_lane)
        expected_actions = Discrete(env.action_space.n, seed=seed)

        assert obs.shape == u.observation_type.space().shape
        assert info["scenario_seed"] == seed
        assert tuple(info["initial_ego_lane"]) == scenario.initial_ego_lane
        assert tuple(info["target_lane"]) == scenario.target_lane
        assert info["lead_initial_gap_m"] == pytest.approx(
            scenario.spawn_gap,
            abs=1e-8,
        )
        assert scenario.initial_ego_lane[2] == 1
        assert scenario.target_lane[2] == 0
        assert scenario.env_unwrapped is u
        assert scenario.road is u.road
        assert scenario.ego_vehicle is ego
        assert scenario.reference_lane is initial_lane
        assert scenario.initial_lead_vehicle in u.road.vehicles
        assert isinstance(scenario.initial_lead_vehicle, IDMVehicle)
        assert scenario.initial_lead_vehicle.lane_index == scenario.initial_ego_lane
        assert scenario.initial_lead_vehicle.enable_lane_change is False
        assert scenario.initial_lead_vehicle.speed == pytest.approx(15.0)
        assert scenario.initial_lead_vehicle.target_speed == pytest.approx(15.0)
        assert scenario.lead_initial_gap_m == pytest.approx(
            scenario.spawn_gap,
            abs=1e-8,
        )
        assert (
            _longitudinal(initial_lane, scenario.initial_lead_vehicle.position)
            - _longitudinal(initial_lane, ego.position)
        ) == pytest.approx(scenario.spawn_gap, abs=1e-8)
        assert scenario.target_initially_ahead is True

        initial_lane_traffic = [
            vehicle
            for vehicle in u.road.vehicles
            if initial_lane.on_lane(vehicle.position)
        ]
        target_lane_traffic = [
            vehicle
            for vehicle in u.road.vehicles
            if target_lane.on_lane(vehicle.position)
        ]
        assert initial_lane_traffic == [ego, scenario.initial_lead_vehicle]
        assert target_lane_traffic == []
        for vehicle in u.road.vehicles:
            if vehicle is not ego and hasattr(vehicle, "enable_lane_change"):
                assert vehicle.enable_lane_change is False

        assert [env.action_space.sample() for _ in range(8)] == [
            expected_actions.sample() for _ in range(8)
        ]
        scenario.validate_preconditions(env)
    finally:
        env.close()


def test_repeated_same_seed_reproduces_full_scene_and_observation() -> None:
    scenario, env, _ = _make_scenario_env(seed=3)
    try:
        first_obs, first_info = scenario.reset(env, seed=3)
        first_geometry = _geometry_snapshot(env)
        first_actions = [env.action_space.sample() for _ in range(6)]

        second_obs, second_info = scenario.reset(env, seed=3)
        second_geometry = _geometry_snapshot(env)
        second_actions = [env.action_space.sample() for _ in range(6)]

        np.testing.assert_array_equal(first_obs, second_obs)
        assert first_info == second_info
        assert first_geometry == second_geometry
        assert first_actions == second_actions
    finally:
        env.close()


def test_reset_returns_observation_refreshed_after_scene_edits() -> None:
    scenario, env, _ = _make_scenario_env(seed=0)
    try:
        obs, _ = scenario.reset(env, seed=0)
        current_obs = env.unwrapped.observation_type.observe()
        features = select_grounding(
            load_skill("lane-change-overtake"),
            "highway-env",
        ).observation["features"]
        indexes = {feature: index for index, feature in enumerate(features)}

        np.testing.assert_array_equal(obs, current_obs)
        present_neighbors = obs[1:][obs[1:, indexes["presence"]] == 1]
        assert any(
            row[indexes["x"]] == pytest.approx(scenario.spawn_gap, abs=1e-6)
            and row[indexes["y"]] == pytest.approx(0.0, abs=1e-6)
            for row in present_neighbors
        )
    finally:
        env.close()


def _remove_bound_lead(scenario: LaneChangeOvertakeScenario, env: object) -> None:
    env.unwrapped.road.vehicles.remove(scenario.initial_lead_vehicle)


def _move_bound_lead_behind(
    scenario: LaneChangeOvertakeScenario,
    env: object,
) -> None:
    u = env.unwrapped
    lane = u.road.network.get_lane(scenario.initial_ego_lane)
    ego_s = _longitudinal(lane, u.vehicle.position)
    scenario.initial_lead_vehicle.position = lane.position(ego_s - 5.0, 0)
    scenario.initial_lead_vehicle.on_state_update()


def _make_bound_lead_too_fast(
    scenario: LaneChangeOvertakeScenario,
    env: object,
) -> None:
    del env
    scenario.initial_lead_vehicle.speed = scenario.target_speed_mps


def _block_target_corridor(
    scenario: LaneChangeOvertakeScenario,
    env: object,
) -> None:
    u = env.unwrapped
    reference_lane = u.road.network.get_lane(scenario.initial_ego_lane)
    ego_s = _longitudinal(reference_lane, u.vehicle.position)
    blocker = IDMVehicle.make_on_lane(
        u.road,
        scenario.target_lane,
        ego_s + scenario.spawn_gap + scenario.success_margin,
        speed=float(u.vehicle.speed),
    )
    blocker.enable_lane_change = False
    u.road.vehicles.append(blocker)


def _make_target_lane_unreachable(
    scenario: LaneChangeOvertakeScenario,
    env: object,
) -> None:
    env.unwrapped.road.network.get_lane(scenario.target_lane).forbidden = True


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_remove_bound_lead, "bound lead.*road"),
        (_move_bound_lead_behind, "ahead"),
        (_make_bound_lead_too_fast, "slower"),
        (_block_target_corridor, "target lane.*clear"),
        (_make_target_lane_unreachable, "reachable"),
    ],
)
def test_validate_preconditions_reports_broken_scene_contract(
    mutate: Callable[[LaneChangeOvertakeScenario, object], None],
    message: str,
) -> None:
    scenario, env, _ = _make_scenario_env(seed=0)
    try:
        scenario.reset(env, seed=0)
        mutate(scenario, env)

        with pytest.raises(ScenarioSetupError, match=message):
            scenario.validate_preconditions(env)
    finally:
        env.close()


def test_reset_rejects_environment_with_fewer_than_two_lanes() -> None:
    card = load_skill("lane-change-overtake")
    grounding = select_grounding(card, "highway-env")
    scenario = LaneChangeOvertakeScenario(card)
    config = scenario.configure()
    config.update({"lanes_count": 1, "initial_lane_id": 0})
    env, _ = make_env(grounding, config, seed=0)
    try:
        with pytest.raises(ScenarioSetupError, match="at least 2 lanes"):
            scenario.reset(env, seed=0)
    finally:
        env.close()


def _invoke_env_coupled_method(
    scenario: LaneChangeOvertakeScenario,
    env: object,
    method: str,
) -> object:
    if method == "validate_preconditions":
        return scenario.validate_preconditions(env)
    if method == "update":
        return scenario.update(env, t=0)
    if method == "build_context":
        return scenario.build_context(
            env,
            prev_primitive=None,
            available_primitives=["maintain-speed"],
        )
    if method == "is_terminal":
        return scenario.is_terminal(env)
    raise AssertionError(f"unknown method: {method}")


@pytest.mark.parametrize(
    "method",
    ["validate_preconditions", "update", "build_context", "is_terminal"],
)
def test_env_coupled_methods_reject_a_different_environment(method: str) -> None:
    scenario, bound_env, _ = _make_scenario_env(seed=0)
    _, other_env, _ = _make_scenario_env(seed=1)
    try:
        scenario.reset(bound_env, seed=0)

        with pytest.raises(ScenarioSetupError, match="different environment"):
            _invoke_env_coupled_method(scenario, other_env, method)
    finally:
        bound_env.close()
        other_env.close()


@pytest.mark.parametrize(
    "method",
    ["validate_preconditions", "update", "build_context", "is_terminal"],
)
def test_env_coupled_methods_reject_external_environment_reset(
    method: str,
) -> None:
    scenario, env, _ = _make_scenario_env(seed=0)
    try:
        scenario.reset(env, seed=0)
        env.reset(seed=0)

        with pytest.raises(ScenarioSetupError, match="externally reset"):
            _invoke_env_coupled_method(scenario, env, method)
    finally:
        env.close()


@pytest.mark.parametrize(
    "method",
    ["validate_preconditions", "update", "build_context", "is_terminal"],
)
def test_env_coupled_methods_require_bound_lead_membership(
    method: str,
) -> None:
    scenario, env, _ = _make_scenario_env(seed=0)
    try:
        scenario.reset(env, seed=0)
        env.unwrapped.road.vehicles.remove(scenario.initial_lead_vehicle)

        with pytest.raises(ScenarioSetupError, match="bound lead.*road"):
            _invoke_env_coupled_method(scenario, env, method)
    finally:
        env.close()


def test_build_context_reports_stable_target_relation_and_side_gap_signs() -> None:
    scenario, env, _ = _make_scenario_env(seed=0)
    try:
        scenario.reset(env, seed=0)
        u = env.unwrapped
        ego = u.vehicle
        u.road.vehicles = [ego, scenario.initial_lead_vehicle]
        right_lane = (
            scenario.initial_ego_lane[0],
            scenario.initial_ego_lane[1],
            scenario.initial_ego_lane[2] + 1,
        )
        reference_lane = u.road.network.get_lane(scenario.initial_ego_lane)
        ego_s = _longitudinal(reference_lane, ego.position)
        front = IDMVehicle.make_on_lane(
            u.road,
            right_lane,
            ego_s + 40.0,
            speed=10.0,
        )
        rear = IDMVehicle.make_on_lane(
            u.road,
            right_lane,
            ego_s - 15.0,
            speed=float(ego.speed) + 5.0,
        )
        front.enable_lane_change = rear.enable_lane_change = False
        u.road.vehicles.extend([front, rear])
        primitives = ["change-lane-left", "maintain-speed"]

        context = scenario.build_context(
            env,
            prev_primitive="maintain-speed",
            available_primitives=primitives,
        )
        primitives.append("accelerate")

        assert context.available_primitives == [
            "change-lane-left",
            "maintain-speed",
        ]
        assert context.ego_lane == 1
        assert context.prev_primitive == "maintain-speed"
        assert context.target_ahead is True
        assert context.target_gap_m == pytest.approx(scenario.spawn_gap)
        assert context.target_rel_speed_mps == pytest.approx(-10.0)
        assert context.left_lane_exists is True
        assert context.left_front_gap_m is None
        assert context.left_rear_gap_m is None
        assert context.left_rear_closing_speed_mps is None
        assert context.right_lane_exists is True
        assert context.right_front_gap_m == pytest.approx(35.0)
        assert context.right_rear_gap_m == pytest.approx(10.0)
        assert context.right_rear_closing_speed_mps == pytest.approx(5.0)
    finally:
        env.close()


def test_side_gaps_use_vehicle_lengths_and_clamp_overlap_to_zero() -> None:
    scenario, env, _ = _make_scenario_env(seed=0)
    try:
        scenario.reset(env, seed=0)
        u = env.unwrapped
        ego = u.vehicle
        u.road.vehicles = [ego, scenario.initial_lead_vehicle]
        right_lane_index = (
            scenario.initial_ego_lane[0],
            scenario.initial_ego_lane[1],
            scenario.initial_ego_lane[2] + 1,
        )
        right_lane = u.road.network.get_lane(right_lane_index)
        ego_s = float(right_lane.local_coordinates(ego.position)[0])
        front = IDMVehicle.make_on_lane(
            u.road,
            right_lane_index,
            ego_s + 12.0,
            speed=10.0,
        )
        rear = IDMVehicle.make_on_lane(
            u.road,
            right_lane_index,
            ego_s - 8.0,
            speed=float(ego.speed) + 7.0,
        )
        front.enable_lane_change = rear.enable_lane_change = False
        u.road.vehicles.extend([front, rear])

        context = scenario.build_context(
            env,
            prev_primitive=None,
            available_primitives=["maintain-speed"],
        )

        assert context.right_front_gap_m == pytest.approx(
            12.0 - (ego.LENGTH + front.LENGTH) / 2
        )
        assert context.right_rear_gap_m == pytest.approx(
            8.0 - (ego.LENGTH + rear.LENGTH) / 2
        )
        assert context.right_rear_closing_speed_mps == pytest.approx(7.0)

        front.position = right_lane.position(ego_s + 4.0, 0)
        rear.position = right_lane.position(ego_s - 4.0, 0)
        front.on_state_update()
        rear.on_state_update()
        overlapping = scenario.build_context(
            env,
            prev_primitive=None,
            available_primitives=["maintain-speed"],
        )

        assert overlapping.right_front_gap_m == 0.0
        assert overlapping.right_rear_gap_m == 0.0
        assert overlapping.right_rear_closing_speed_mps == pytest.approx(7.0)
    finally:
        env.close()


def test_update_tracks_first_causal_steps_and_collision_is_sticky() -> None:
    scenario, env, _ = _make_scenario_env(seed=0)
    try:
        scenario.reset(env, seed=0)
        u = env.unwrapped
        reference_lane = u.road.network.get_lane(scenario.initial_ego_lane)
        lead_s = _longitudinal(
            reference_lane,
            scenario.initial_lead_vehicle.position,
        )

        scenario.update(env, t=0)
        assert scenario.lane_change_completed_step is None
        assert scenario.overtake_step is None

        target_lane = u.road.network.get_lane(scenario.target_lane)
        u.vehicle.position = target_lane.position(
            lead_s + scenario.success_margin + 1.0,
            0,
        )
        u.vehicle.on_state_update()
        scenario.update(env, t=4)
        scenario.update(env, t=7)

        assert scenario.lane_change_completed_step == 4
        assert scenario.overtake_step == 4
        assert scenario.evaluate(env) == (
            False,
            "overtake did not occur after lane change",
        )

        u.vehicle.crashed = True
        scenario.update(env, t=8)
        u.vehicle.crashed = False
        scenario.update(env, t=9)
        assert scenario.collision is True
        assert scenario.is_terminal(env) == (False, "")
        assert scenario.evaluate(env) == (False, "collision")
    finally:
        env.close()


@pytest.mark.parametrize("seed", range(5))
def test_scripted_oracle_completes_causal_overtake_without_collision(
    seed: int,
) -> None:
    minimum_spawn_gap = (
        load_skill("lane-change-overtake").parameters["min_lane_gap"]["default"]
        + Vehicle.LENGTH
    )
    scenario, env, name_to_index = _make_scenario_env(
        seed,
        spawn_gap=minimum_spawn_gap,
    )
    try:
        scenario.reset(env, seed)
        terminated = truncated = False

        for t in range(40):
            if t == 0:
                action_name = "LANE_LEFT"
            elif scenario.lane_change_completed_step is None:
                action_name = "IDLE"
            else:
                available_names = {
                    env.unwrapped.action_type.actions[index]
                    for index in env.unwrapped.get_available_actions()
                }
                action_name = "FASTER" if "FASTER" in available_names else "IDLE"

            _, _, terminated, truncated, _ = env.step(name_to_index[action_name])
            scenario.update(env, t)
            done, reason = scenario.is_terminal(env)
            if done:
                assert reason == "success"
                break
            assert terminated is False
            assert truncated is False
        else:
            pytest.fail("scripted oracle did not complete within 40 steps")

        success, reason = scenario.evaluate(env)
        assert success is True, reason
        assert scenario.lane_change_completed_step is not None
        assert scenario.overtake_step is not None
        assert scenario.lane_change_completed_step < scenario.overtake_step
        assert scenario.collision is False
        assert terminated is False
        assert truncated is False
    finally:
        env.close()
