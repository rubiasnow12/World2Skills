import numpy as np
import pytest

from world2skills.runtime.obs_render import render
from world2skills.runtime.types import ObservationContext


FEATURE_NAMES = ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"]


def _context(**overrides):
    values = {
        "available_primitives": [
            "maintain-speed",
            "accelerate",
            "change-lane-left",
        ],
        "ego_lane": 3,
        "prev_primitive": "accelerate",
        "target_ahead": True,
        "target_gap_m": 18.0,
        "target_rel_speed_mps": -10.0,
        "left_lane_exists": True,
        "right_lane_exists": False,
        "left_front_gap_m": 30.0,
        "left_rear_gap_m": 25.0,
        "left_rear_closing_speed_mps": -2.0,
        "right_front_gap_m": None,
        "right_rear_gap_m": None,
        "right_rear_closing_speed_mps": None,
    }
    values.update(overrides)
    return ObservationContext(**values)


def test_render_returns_deterministic_compact_text_with_full_context():
    observation = np.array(
        [
            [1, 200.0, 12.0, 3.0, 4.0, 1.0, 0.0],
            [1, 18.0, -4.0, -10.0, 1.5, 1.0, 0.0],
            [0, 999.0, 999.0, 999.0, 999.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    expected = "\n".join(
        [
            "Ego: position=(0.0, 0.0) lane=3 speed=5.0 m/s",
            "Neighbors: count=1 frame=ego-relative",
            "  #1 dx=18.0 m dy=-4.0 m dvx=-10.0 m/s dvy=1.5 m/s",
            "Target: relation=ahead gap=18.0 m rel_speed=-10.0 m/s",
            "Sign conventions: target rel_speed = target_speed - ego_speed; "
            "negative means target slower. rear_closing_speed = "
            "rear_speed - ego_speed; positive means rear closing.",
            "Left lane: exists=true front_gap=30.0 m rear_gap=25.0 m "
            "rear_closing_speed=-2.0 m/s",
            "Right lane: exists=false front_gap=none rear_gap=none "
            "rear_closing_speed=none",
            "Previous primitive: accelerate",
            "Available primitives: maintain-speed, accelerate, change-lane-left",
        ]
    )

    text = render(observation, _context(), FEATURE_NAMES)

    assert text == expected
    assert render(observation, _context(), FEATURE_NAMES) == expected
    assert "999.0" not in text


def test_render_zero_neighbors_and_none_values():
    observation = np.array(
        [
            [1, 50.0, 8.0, 0.0, 0.0, 1.0, 0.0],
            [0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ],
        dtype=float,
    )
    context = _context(
        available_primitives=[],
        prev_primitive=None,
        target_ahead=False,
        left_lane_exists=False,
        left_front_gap_m=None,
        left_rear_gap_m=None,
        left_rear_closing_speed_mps=None,
        right_lane_exists=True,
    )

    text = render(observation, context, FEATURE_NAMES)

    assert "Neighbors: count=0 frame=ego-relative" in text
    assert "\n  #1" not in text
    assert "Target: relation=behind" in text
    assert "Left lane: exists=false front_gap=none rear_gap=none" in text
    assert "Previous primitive: none" in text
    assert "Available primitives: none" in text


@pytest.mark.parametrize(
    ("observation", "message"),
    [
        ([1, 0, 0, 0, 0], "2-D numpy array"),
        (np.ones(5), "2-D"),
        (np.empty((0, 5)), "at least one row"),
        (np.ones((1, 4)), "column count"),
    ],
)
def test_render_rejects_malformed_observation_shape(observation, message):
    with pytest.raises(ValueError, match=message):
        render(
            observation,
            _context(),
            ["presence", "x", "y", "vx", "vy"],
        )


def test_render_rejects_duplicate_feature_names():
    observation = np.ones((1, 5))

    with pytest.raises(ValueError, match="feature_names must be unique"):
        render(
            observation,
            _context(),
            ["presence", "x", "y", "vx", "vx"],
        )


@pytest.mark.parametrize("missing", ["presence", "x", "y", "vx", "vy"])
def test_render_requires_kinematics_features(missing):
    feature_names = [name for name in FEATURE_NAMES if name != missing]
    observation = np.ones((1, len(feature_names)))

    with pytest.raises(ValueError, match=f"missing required feature.*{missing}"):
        render(observation, _context(), feature_names)


def test_render_requires_present_ego():
    observation = np.array([[0, 0, 0, 0, 0]], dtype=float)

    with pytest.raises(ValueError, match="ego presence must be positive"):
        render(
            observation,
            _context(),
            ["presence", "x", "y", "vx", "vy"],
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_render_rejects_nonfinite_values_in_present_rows(bad_value):
    observation = np.array(
        [
            [1, 0, 0, 25, 0],
            [1, 10, 0, bad_value, 0],
        ],
        dtype=float,
    )

    with pytest.raises(
        ValueError, match="present rows must contain only finite values"
    ):
        render(
            observation,
            _context(),
            ["presence", "x", "y", "vx", "vy"],
        )


def test_render_rejects_nonfinite_presence():
    observation = np.array([[1, 0, 0, 25, 0], [np.nan, 0, 0, 0, 0]])

    with pytest.raises(ValueError, match="presence values must be finite"):
        render(
            observation,
            _context(),
            ["presence", "x", "y", "vx", "vy"],
        )


@pytest.mark.parametrize(
    "presence",
    [-1.0, 0.5, 2.0, np.nextafter(1.0, 2.0)],
)
def test_render_rejects_nonbinary_presence_before_rows_can_hide_nan(presence):
    observation = np.array(
        [
            [1, 0, 0, 25, 0],
            [presence, np.nan, 0, 0, 0],
        ]
    )

    with pytest.raises(ValueError, match="presence values must be exactly 0 or 1"):
        render(
            observation,
            _context(),
            ["presence", "x", "y", "vx", "vy"],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("target_gap_m", -0.1, "target_gap_m must be finite and nonnegative"),
        ("target_gap_m", np.nan, "target_gap_m must be finite and nonnegative"),
        ("target_gap_m", np.inf, "target_gap_m must be finite and nonnegative"),
        ("target_rel_speed_mps", np.nan, "target_rel_speed_mps must be finite"),
        ("target_rel_speed_mps", np.inf, "target_rel_speed_mps must be finite"),
    ],
)
def test_render_rejects_invalid_target_context(field, value, message):
    context = _context(**{field: value})
    observation = np.array([[1, 0, 0, 25, 0]], dtype=float)

    with pytest.raises(ValueError, match=message):
        render(
            observation,
            context,
            ["presence", "x", "y", "vx", "vy"],
        )


@pytest.mark.parametrize(
    "field",
    [
        "left_front_gap_m",
        "left_rear_gap_m",
        "right_front_gap_m",
        "right_rear_gap_m",
    ],
)
@pytest.mark.parametrize("value", [-0.1, np.nan, np.inf])
def test_render_rejects_invalid_optional_lane_gaps(field, value):
    context = _context(
        right_lane_exists=True,
        **{field: value},
    )
    observation = np.array([[1, 0, 0, 25, 0]], dtype=float)

    with pytest.raises(ValueError, match=f"{field} must be finite and nonnegative"):
        render(
            observation,
            context,
            ["presence", "x", "y", "vx", "vy"],
        )


@pytest.mark.parametrize(
    "field",
    [
        "left_rear_closing_speed_mps",
        "right_rear_closing_speed_mps",
    ],
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_render_rejects_nonfinite_rear_closing_speeds(field, value):
    context = _context(
        right_lane_exists=True,
        **{field: value},
    )
    observation = np.array([[1, 0, 0, 25, 0]], dtype=float)

    with pytest.raises(ValueError, match=f"{field} must be finite"):
        render(
            observation,
            context,
            ["presence", "x", "y", "vx", "vy"],
        )


@pytest.mark.parametrize(
    ("lane", "field"),
    [
        ("left", "left_front_gap_m"),
        ("left", "left_rear_gap_m"),
        ("left", "left_rear_closing_speed_mps"),
        ("right", "right_front_gap_m"),
        ("right", "right_rear_gap_m"),
        ("right", "right_rear_closing_speed_mps"),
    ],
)
def test_render_requires_absent_lane_metrics_to_be_none(lane, field):
    overrides = {
        f"{lane}_lane_exists": False,
        f"{lane}_front_gap_m": None,
        f"{lane}_rear_gap_m": None,
        f"{lane}_rear_closing_speed_mps": None,
        field: 1.0,
    }
    context = _context(**overrides)
    observation = np.array([[1, 0, 0, 25, 0]], dtype=float)

    with pytest.raises(
        ValueError,
        match=f"{lane} lane metrics must be None when {lane}_lane_exists is false",
    ):
        render(
            observation,
            context,
            ["presence", "x", "y", "vx", "vy"],
        )


@pytest.mark.parametrize(
    "available_primitives",
    [[""], ["   "], ["accelerate", ""], ["accelerate", 1]],
)
def test_render_rejects_invalid_available_primitives(available_primitives):
    context = _context(available_primitives=available_primitives)
    observation = np.array([[1, 0, 0, 25, 0]], dtype=float)

    with pytest.raises(
        ValueError,
        match="available_primitives must contain only non-empty strings",
    ):
        render(
            observation,
            context,
            ["presence", "x", "y", "vx", "vy"],
        )


@pytest.mark.parametrize("prev_primitive", [1, [], {}])
def test_render_rejects_nonstring_previous_primitive(prev_primitive):
    context = _context(prev_primitive=prev_primitive)
    observation = np.array([[1, 0, 0, 25, 0]], dtype=float)

    with pytest.raises(ValueError, match="prev_primitive must be None or a string"):
        render(
            observation,
            context,
            ["presence", "x", "y", "vx", "vy"],
        )
