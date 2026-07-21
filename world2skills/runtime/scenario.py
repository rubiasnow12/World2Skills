"""Deterministic lane-change-overtake scenario and causal evaluator."""

from __future__ import annotations

from copy import deepcopy
import math
from numbers import Real
from typing import Any

from .types import ObservationContext, SkillCard


class ScenarioSetupError(RuntimeError):
    """Raised when the overtake scenario preconditions cannot be established."""


def judge_success(
    target_initially_ahead: bool,
    lane_change_completed_step: int | None,
    overtake_step: int | None,
    collision: bool,
) -> tuple[bool, str]:
    """Judge causal overtake success without consulting environment state."""

    if collision:
        return False, "collision"
    if not target_initially_ahead:
        return False, "target was not initially ahead"
    if lane_change_completed_step is None:
        return False, "lane change not completed"
    if overtake_step is None:
        return False, "target not overtaken"
    if lane_change_completed_step >= overtake_step:
        return False, "overtake did not occur after lane change"
    return (
        True,
        f"lane change at step {lane_change_completed_step}, "
        f"then overtook at step {overtake_step}",
    )


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    )


class LaneChangeOvertakeScenario:
    """Build and evaluate a deterministic lane-change-overtake task."""

    def __init__(
        self,
        skill_card: SkillCard,
        *,
        success_margin: float = 5.0,
        lead_speed_ratio: float = 0.6,
        target_lane_front_clearance_m: float = 20.0,
        target_lane_rear_clearance_m: float = 20.0,
        spawn_gap: float = 30.0,
    ) -> None:
        self.skill_card = skill_card
        self.success_margin = self._positive(
            success_margin,
            "success_margin",
        )
        self.lead_speed_ratio = self._ratio(
            lead_speed_ratio,
            "lead_speed_ratio",
        )
        self.target_lane_front_clearance_m = self._nonnegative(
            target_lane_front_clearance_m,
            "target_lane_front_clearance_m",
        )
        self.target_lane_rear_clearance_m = self._nonnegative(
            target_lane_rear_clearance_m,
            "target_lane_rear_clearance_m",
        )
        self.spawn_gap = self._positive(spawn_gap, "spawn_gap")
        self.target_speed_mps = self._target_speed(skill_card)
        self.min_lane_gap_m = self._skill_positive_parameter(
            skill_card,
            "min_lane_gap",
        )
        from highway_env.vehicle.kinematics import Vehicle

        self.minimum_spawn_clearance_m = self.min_lane_gap_m + float(Vehicle.LENGTH)
        if self.spawn_gap < self.minimum_spawn_clearance_m:
            raise ValueError(
                "spawn_gap must be at least "
                f"{self.minimum_spawn_clearance_m:.1f} m to preserve "
                "min_lane_gap.default bumper clearance"
            )
        self._reset_state()

    @staticmethod
    def _positive(value: object, name: str) -> float:
        if not _finite_number(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")
        return float(value)

    @staticmethod
    def _nonnegative(value: object, name: str) -> float:
        if not _finite_number(value) or value < 0:
            raise ValueError(f"{name} must be a nonnegative finite number")
        return float(value)

    @staticmethod
    def _ratio(value: object, name: str) -> float:
        if not _finite_number(value) or not 0 < value < 1:
            raise ValueError(f"{name} must be strictly between 0 and 1")
        return float(value)

    @classmethod
    def _target_speed(cls, skill_card: SkillCard) -> float:
        return cls._skill_positive_parameter(skill_card, "target_speed")

    @classmethod
    def _skill_positive_parameter(
        cls,
        skill_card: SkillCard,
        parameter_name: str,
    ) -> float:
        try:
            value = skill_card.parameters[parameter_name]["default"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"skill {parameter_name}.default must be configured"
            ) from exc
        return cls._positive(value, f"{parameter_name}.default")

    def _reset_state(self) -> None:
        self.env_unwrapped: Any | None = None
        self.road: Any | None = None
        self.ego_vehicle: Any | None = None
        self.initial_ego_lane: tuple[str, str, int] | None = None
        self.reference_lane: Any | None = None
        self.target_lane: tuple[str, str, int] | None = None
        self.initial_lead_vehicle: Any | None = None
        self.target_initially_ahead = False
        self.lead_initial_gap_m: float | None = None
        self.lane_change_completed_step: int | None = None
        self.overtake_step: int | None = None
        self.collision = False

    @staticmethod
    def configure() -> dict[str, Any]:
        """Return the fixed M1 highway-v0 environment configuration."""

        config = {
            "lanes_count": 4,
            "vehicles_count": 30,
            "controlled_vehicles": 1,
            "initial_lane_id": 1,
            "duration": 40,
            "policy_frequency": 1,
            "simulation_frequency": 15,
            "obs_vehicles_count": 8,
        }
        if config["lanes_count"] < 2:
            raise ValueError("M1 requires at least 2 lanes")
        return config

    def config_snapshot(self) -> dict[str, Any]:
        """Return resolved scenario and environment parameters for run metadata."""

        return {
            "success_margin": self.success_margin,
            "lead_speed_ratio": self.lead_speed_ratio,
            "target_lane_front_clearance_m": (self.target_lane_front_clearance_m),
            "target_lane_rear_clearance_m": (self.target_lane_rear_clearance_m),
            "spawn_gap": self.spawn_gap,
            "min_lane_gap_m": self.min_lane_gap_m,
            "minimum_spawn_clearance_m": self.minimum_spawn_clearance_m,
            "target_speed_mps": self.target_speed_mps,
            "environment_config": deepcopy(self.configure()),
        }

    def _require_initialized(self) -> None:
        if (
            self.env_unwrapped is None
            or self.road is None
            or self.ego_vehicle is None
            or self.initial_ego_lane is None
            or self.target_lane is None
            or self.initial_lead_vehicle is None
            or self.reference_lane is None
        ):
            raise ScenarioSetupError("scenario has not been reset")

    def _require_owned_env(self, env: Any) -> Any:
        self._require_initialized()
        u = env.unwrapped
        if u is not self.env_unwrapped:
            raise ScenarioSetupError(
                "scenario received a different environment than the one reset"
            )
        if u.road is not self.road or u.vehicle is not self.ego_vehicle:
            raise ScenarioSetupError(
                "bound environment was externally reset after scenario setup"
            )
        if self.initial_lead_vehicle not in self.road.vehicles:
            raise ScenarioSetupError("bound lead vehicle is not on the road")
        return u

    def _longitudinal(self, position: Any) -> float:
        if self.reference_lane is None:
            raise ScenarioSetupError("initial reference lane is not configured")
        return float(self.reference_lane.local_coordinates(position)[0])

    @staticmethod
    def _lane_contains(lane: Any, vehicle: Any) -> bool:
        longitudinal, lateral = lane.local_coordinates(vehicle.position)
        return bool(
            lane.on_lane(
                vehicle.position,
                longitudinal=longitudinal,
                lateral=lateral,
            )
        )

    def _choose_target_lane(self, env: Any) -> tuple[str, str, int]:
        u = env.unwrapped
        ego = u.vehicle
        side_lanes = u.road.network.side_lanes(self.initial_ego_lane)
        reachable = [
            lane_index
            for lane_index in side_lanes
            if u.road.network.get_lane(lane_index).is_reachable_from(ego.position)
        ]
        if not reachable:
            raise ScenarioSetupError("no reachable adjacent target lane")
        return min(reachable, key=lambda lane_index: lane_index[2])

    def reset(self, env: Any, seed: int) -> tuple[Any, dict[str, Any]]:
        """Reset, replace random task traffic, and return a refreshed observation."""

        self._reset_state()
        _, info = env.reset(seed=seed)
        env.action_space.seed(seed)
        u = env.unwrapped
        ego = u.vehicle

        lane_count = len(u.road.network.all_side_lanes(ego.lane_index))
        if lane_count < 2:
            raise ScenarioSetupError("overtake scenario requires at least 2 lanes")

        self.env_unwrapped = u
        self.road = u.road
        self.ego_vehicle = ego
        self.initial_ego_lane = tuple(ego.lane_index)
        self.reference_lane = u.road.network.get_lane(self.initial_ego_lane)
        self.target_lane = self._choose_target_lane(env)
        target_lane_object = u.road.network.get_lane(self.target_lane)

        kept_vehicles = []
        for vehicle in u.road.vehicles:
            if vehicle is ego:
                kept_vehicles.append(vehicle)
                continue
            if self._lane_contains(
                self.reference_lane,
                vehicle,
            ) or self._lane_contains(target_lane_object, vehicle):
                continue
            if hasattr(vehicle, "enable_lane_change"):
                vehicle.enable_lane_change = False
            kept_vehicles.append(vehicle)
        u.road.vehicles = kept_vehicles

        from highway_env.vehicle.behavior import IDMVehicle

        ego_longitudinal = self._longitudinal(ego.position)
        lead_speed = self.lead_speed_ratio * self.target_speed_mps
        lead = IDMVehicle.make_on_lane(
            u.road,
            self.initial_ego_lane,
            ego_longitudinal + self.spawn_gap,
            speed=lead_speed,
        )
        lead.speed = lead_speed
        lead.target_speed = lead_speed
        lead.enable_lane_change = False
        u.road.vehicles.append(lead)
        self.initial_lead_vehicle = lead

        for vehicle in u.road.vehicles:
            if vehicle is not ego and hasattr(vehicle, "enable_lane_change"):
                vehicle.enable_lane_change = False

        lead_longitudinal = self._longitudinal(lead.position)
        self.lead_initial_gap_m = lead_longitudinal - ego_longitudinal
        self.target_initially_ahead = self.lead_initial_gap_m > 0
        self.validate_preconditions(env)

        observation = u.observation_type.observe()
        refreshed_info = dict(info)
        refreshed_info.pop("action", None)
        refreshed_info.update(
            {
                "scenario_seed": seed,
                "initial_ego_lane": self.initial_ego_lane,
                "target_lane": self.target_lane,
                "lead_initial_gap_m": self.lead_initial_gap_m,
            }
        )
        return observation, refreshed_info

    def validate_preconditions(self, env: Any) -> None:
        """Recompute and validate every scene invariant required by M1."""

        u = self._require_owned_env(env)
        ego = u.vehicle
        road = u.road
        lane_indexes = road.network.all_side_lanes(self.initial_ego_lane)
        if len(lane_indexes) < 2:
            raise ScenarioSetupError("overtake scenario requires at least 2 lanes")
        if self.target_lane not in road.network.side_lanes(self.initial_ego_lane):
            raise ScenarioSetupError("target lane is not adjacent to initial lane")

        lead = self.initial_lead_vehicle
        if not lead.on_road:
            raise ScenarioSetupError("bound lead vehicle is not on the road")
        if lead.lane_index != self.initial_ego_lane or not self._lane_contains(
            self.reference_lane,
            lead,
        ):
            raise ScenarioSetupError("bound lead vehicle left the initial lane")

        ego_longitudinal = self._longitudinal(ego.position)
        lead_longitudinal = self._longitudinal(lead.position)
        gap = lead_longitudinal - ego_longitudinal
        if gap <= 0:
            raise ScenarioSetupError("bound lead vehicle must be ahead of ego")
        actual_minimum_spawn_clearance = (
            self.min_lane_gap_m + float(ego.LENGTH) / 2 + float(lead.LENGTH) / 2
        )
        if gap < actual_minimum_spawn_clearance:
            raise ScenarioSetupError(
                "spawn_gap is unsafe for actual vehicle lengths: "
                f"requires at least {actual_minimum_spawn_clearance:.1f} m"
            )
        tolerance = max(1e-8, self.spawn_gap * 1e-8)
        if not math.isclose(gap, self.spawn_gap, abs_tol=tolerance):
            raise ScenarioSetupError(
                "bound lead gap does not match configured spawn_gap"
            )

        expected_lead_speed = self.lead_speed_ratio * self.target_speed_mps
        if not math.isclose(
            float(lead.speed),
            expected_lead_speed,
            abs_tol=1e-8,
        ) or not math.isclose(
            float(lead.target_speed),
            expected_lead_speed,
            abs_tol=1e-8,
        ):
            raise ScenarioSetupError("bound lead is not at the configured slower speed")
        if float(lead.speed) >= self.target_speed_mps:
            raise ScenarioSetupError("bound lead vehicle must be slower than target")

        target_lane = road.network.get_lane(self.target_lane)
        if not target_lane.is_reachable_from(ego.position):
            raise ScenarioSetupError("target lane is not reachable from ego")
        target_ego_longitudinal = float(target_lane.local_coordinates(ego.position)[0])
        corridor_start = target_ego_longitudinal - self.target_lane_rear_clearance_m
        corridor_end = (
            target_ego_longitudinal
            + self.spawn_gap
            + self.success_margin
            + self.target_lane_front_clearance_m
        )
        for vehicle in road.vehicles:
            if vehicle is ego or vehicle is lead:
                continue
            if self._lane_contains(target_lane, vehicle):
                vehicle_longitudinal = float(
                    target_lane.local_coordinates(vehicle.position)[0]
                )
                if corridor_start <= vehicle_longitudinal <= corridor_end:
                    raise ScenarioSetupError("target lane corridor is not clear")
            if self._lane_contains(self.reference_lane, vehicle) and vehicle is not ego:
                raise ScenarioSetupError(
                    "unexpected traffic remains in the initial lane"
                )
            if bool(getattr(vehicle, "enable_lane_change", False)):
                raise ScenarioSetupError(
                    "remaining traffic has autonomous lane changes enabled"
                )

    def update(self, env: Any, t: int) -> None:
        """Record first causal milestones using the initial lane reference axis."""

        if type(t) is not int or t < 0:
            raise ValueError("t must be a nonnegative integer")
        u = self._require_owned_env(env)
        ego = u.vehicle
        self.collision = self.collision or bool(ego.crashed)

        if (
            self.lane_change_completed_step is None
            and ego.lane_index != self.initial_ego_lane
        ):
            self.lane_change_completed_step = t

        if self.overtake_step is None:
            ego_longitudinal = self._longitudinal(ego.position)
            lead_longitudinal = self._longitudinal(self.initial_lead_vehicle.position)
            if ego_longitudinal - lead_longitudinal >= self.success_margin:
                self.overtake_step = t

    def is_terminal(self, env: Any | None = None) -> tuple[bool, str]:
        """Return terminal only for scenario success, never for Gym termination."""

        if env is not None:
            self._require_owned_env(env)
        success, _ = self.evaluate()
        return (True, "success") if success else (False, "")

    def evaluate(self, env: Any | None = None) -> tuple[bool, str]:
        """Evaluate the currently recorded causal state."""

        del env
        return judge_success(
            self.target_initially_ahead,
            self.lane_change_completed_step,
            self.overtake_step,
            self.collision,
        )

    def _side_gap(
        self,
        env: Any,
        lane_index: tuple[str, str, int] | None,
    ) -> tuple[float | None, float | None, float | None]:
        if lane_index is None:
            return None, None, None
        u = env.unwrapped
        ego = u.vehicle
        lane = u.road.network.get_lane(lane_index)
        ego_longitudinal = float(lane.local_coordinates(ego.position)[0])
        front, rear = u.road.neighbour_vehicles(ego, lane_index)
        front_gap = (
            max(
                0.0,
                float(lane.local_coordinates(front.position)[0])
                - ego_longitudinal
                - (float(ego.LENGTH) + float(front.LENGTH)) / 2,
            )
            if front is not None
            else None
        )
        rear_gap = (
            max(
                0.0,
                ego_longitudinal
                - float(lane.local_coordinates(rear.position)[0])
                - (float(ego.LENGTH) + float(rear.LENGTH)) / 2,
            )
            if rear is not None
            else None
        )
        rear_closing_speed = (
            float(rear.speed) - float(ego.speed) if rear is not None else None
        )
        return front_gap, rear_gap, rear_closing_speed

    @staticmethod
    def _context_primitives(
        available_primitives: list[str],
    ) -> list[str]:
        if not isinstance(available_primitives, list) or any(
            not isinstance(primitive, str) or not primitive.strip()
            for primitive in available_primitives
        ):
            raise ValueError("available_primitives must be a list of non-empty strings")
        if len(set(available_primitives)) != len(available_primitives):
            raise ValueError("available_primitives must be unique")
        return available_primitives.copy()

    def build_context(
        self,
        env: Any,
        prev_primitive: str | None,
        available_primitives: list[str],
    ) -> ObservationContext:
        """Build a validated target/side-lane context for observation rendering."""

        if prev_primitive is not None and (
            not isinstance(prev_primitive, str) or not prev_primitive.strip()
        ):
            raise ValueError("prev_primitive must be None or a non-empty string")
        u = self._require_owned_env(env)
        ego = u.vehicle

        lane_indexes = u.road.network.all_side_lanes(ego.lane_index)
        lane_ids = {lane_index[2] for lane_index in lane_indexes}
        start, end, lane_id = ego.lane_index
        left_lane = (start, end, lane_id - 1) if lane_id - 1 in lane_ids else None
        right_lane = (start, end, lane_id + 1) if lane_id + 1 in lane_ids else None

        ego_longitudinal = self._longitudinal(ego.position)
        lead_longitudinal = self._longitudinal(self.initial_lead_vehicle.position)
        target_delta = lead_longitudinal - ego_longitudinal
        left_front, left_rear, left_closing = self._side_gap(
            env,
            left_lane,
        )
        right_front, right_rear, right_closing = self._side_gap(
            env,
            right_lane,
        )
        return ObservationContext(
            available_primitives=self._context_primitives(available_primitives),
            ego_lane=int(lane_id),
            prev_primitive=prev_primitive,
            target_ahead=target_delta > 0,
            target_gap_m=abs(target_delta),
            target_rel_speed_mps=(
                float(self.initial_lead_vehicle.speed) - float(ego.speed)
            ),
            left_lane_exists=left_lane is not None,
            right_lane_exists=right_lane is not None,
            left_front_gap_m=left_front,
            left_rear_gap_m=left_rear,
            left_rear_closing_speed_mps=left_closing,
            right_front_gap_m=right_front,
            right_rear_gap_m=right_rear,
            right_rear_closing_speed_mps=right_closing,
        )
