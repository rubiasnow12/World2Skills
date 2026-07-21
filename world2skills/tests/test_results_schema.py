from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from world2skills.evaluation.metrics import (
    aggregate,
    episode_to_result_dict,
    write_outputs,
)
from world2skills.runtime.types import (
    BatchResult,
    DecisionResult,
    EpisodeResult,
    StepRecord,
)


def _step(*, obs_summary: str = "full observation\nwith context") -> StepRecord:
    decision = DecisionResult(
        primitive="maintain-speed",
        backend_action="IDLE",
        action_index=1,
        request_hash="request-123",
        raw_response='{"primitive":"maintain-speed"}',
        cache_hit=False,
        latency_ms=12.5,
        decision_status="ok",
        fallback_reason=None,
        available_primitives=["maintain-speed", "accelerate"],
    )
    return StepRecord.from_decision(
        decision,
        t=0,
        obs_summary=obs_summary,
        reward=0.75,
        crashed=False,
    )


def _episode(
    seed: int,
    *,
    status: str = "ok",
    success: bool = False,
    crashed: bool = False,
    episode_return: float = 10.0,
    step_records: list[StepRecord] | None = None,
    exception_type: str | None = None,
    exception_message: str | None = None,
    cleanup_error: str | None = None,
) -> EpisodeResult:
    return EpisodeResult(
        seed=seed,
        status=status,
        success=success,
        success_reason="overtake complete" if success else "",
        episode_return=episode_return,
        crashed=crashed,
        steps=len(step_records or []),
        mean_speed=20.0,
        parse_failures=1,
        unavailable_action_attempts=2,
        llm_errors=3,
        terminated=False,
        truncated=False,
        max_steps_reached=False,
        scenario_completed=success,
        termination_reason="success" if success else "error",
        target_initially_ahead=True,
        lane_change_completed_step=1 if success else None,
        overtake_step=2 if success else None,
        lead_initial_gap_m=30.0,
        exception_type=exception_type,
        exception_message=exception_message,
        cleanup_error=cleanup_error,
        step_records=list(step_records or []),
    )


def _metadata(seeds: list[int]) -> dict[str, object]:
    return {
        "model": "mock-model",
        "prompt_version": "lco-v1",
        "skill": "lane-change-overtake",
        "environment": "highway-v0",
        "seeds": seeds,
    }


def _batch(
    results: list[EpisodeResult],
    seeds: list[int] | None = None,
) -> BatchResult:
    actual_seeds = seeds if seeds is not None else [r.seed for r in results]
    return aggregate(results, **_metadata(actual_seeds))


def test_episode_to_result_dict_excludes_only_step_records() -> None:
    episode = _episode(
        4,
        status="error",
        crashed=True,
        step_records=[_step()],
        exception_type="RuntimeError",
        exception_message="provider failed",
        cleanup_error="close failed",
    )

    result = episode_to_result_dict(episode)
    expected = asdict(episode)
    expected.pop("step_records")

    assert result == expected
    assert "step_records" not in result
    assert result["exception_type"] == "RuntimeError"
    assert result["exception_message"] == "provider failed"
    assert result["cleanup_error"] == "close failed"


def test_aggregate_counts_rates_and_collided_error_episode() -> None:
    results = [
        _episode(0, success=True, episode_return=20.0),
        _episode(1, crashed=True, episode_return=5.0),
        _episode(
            2,
            status="error",
            crashed=True,
            episode_return=1000.0,
            exception_type="RuntimeError",
        ),
    ]

    batch = _batch(results)

    assert batch.schema_version == "0.1"
    assert batch.requested_episodes == 3
    assert batch.completed_episodes == 2
    assert batch.error_episodes == 1
    assert batch.completed_episodes + batch.error_episodes == 3
    assert batch.successful_episodes == 1
    assert batch.success_rate == pytest.approx(1 / 3)
    assert batch.collision_rate == pytest.approx(2 / 3)
    assert batch.mean_return == pytest.approx(12.5)


def test_aggregate_orders_results_by_requested_seeds() -> None:
    results = [_episode(7), _episode(2), _episode(5)]

    batch = _batch(results, seeds=[5, 7, 2])

    assert batch.seeds == [5, 7, 2]
    assert [result["seed"] for result in batch.results] == [5, 7, 2]


@pytest.mark.parametrize(
    ("seeds", "error"),
    [
        ([], "non-empty"),
        ((0, 1), "list"),
        ([0, 0], "unique"),
        ([-1], "nonnegative"),
        ([True], "nonnegative integer"),
        ([1.5], "nonnegative integer"),
    ],
)
def test_aggregate_rejects_invalid_seed_contract(
    seeds: object,
    error: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        aggregate([], **_metadata(seeds))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("results", "seeds", "error"),
    [
        ([_episode(0)], [0, 1], "missing"),
        ([_episode(0), _episode(1)], [0], "extra"),
        ([_episode(0), _episode(0)], [0], "duplicate"),
    ],
)
def test_aggregate_rejects_result_seed_mismatch(
    results: list[EpisodeResult],
    seeds: list[int],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        _batch(results, seeds=seeds)


def test_aggregate_rejects_noninteger_episode_seed() -> None:
    episode = replace(_episode(1), seed=True)

    with pytest.raises(ValueError, match="result seeds.*nonnegative integers"):
        _batch([episode], seeds=[1])


@pytest.mark.parametrize("status", ["", "failed", "OK"])
def test_aggregate_rejects_invalid_episode_status(status: str) -> None:
    with pytest.raises(ValueError, match="status"):
        _batch([_episode(0, status=status)])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("success", 1),
        ("crashed", 2),
        ("terminated", 0),
        ("truncated", "false"),
        ("max_steps_reached", None),
        ("scenario_completed", 1),
    ],
)
def test_aggregate_rejects_nonboolean_summary_flags(
    field: str,
    value: object,
) -> None:
    episode = replace(_episode(0), **{field: value})

    with pytest.raises(ValueError, match=rf"{field}.*bool"):
        _batch([episode])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("steps", -1),
        ("steps", 1.5),
        ("parse_failures", True),
        ("unavailable_action_attempts", -1),
        ("llm_errors", 2.5),
        ("lane_change_completed_step", -1),
        ("overtake_step", True),
    ],
)
def test_aggregate_rejects_invalid_nonnegative_integer_fields(
    field: str,
    value: object,
) -> None:
    episode = replace(_episode(0), **{field: value})

    with pytest.raises(ValueError, match=rf"{field}.*nonnegative integer"):
        _batch([episode])


def test_aggregate_rejects_successful_error_episode() -> None:
    episode = _episode(0, status="error", success=True)

    with pytest.raises(ValueError, match="error.*success"):
        _batch([episode])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("episode_return", float("nan")),
        ("episode_return", float("inf")),
        ("episode_return", True),
        ("mean_speed", float("-inf")),
        ("mean_speed", "fast"),
        ("lead_initial_gap_m", float("nan")),
        ("lead_initial_gap_m", False),
    ],
)
def test_aggregate_rejects_invalid_summary_numbers(
    field: str,
    value: object,
) -> None:
    episode = replace(_episode(0), **{field: value})

    with pytest.raises(ValueError, match=rf"{field}.*finite"):
        _batch([episode])


def test_write_outputs_writes_strict_summaries_and_full_traces(
    tmp_path: Path,
) -> None:
    full_obs = (
        "ego: x=0.00 y=0.00 speed=20.00\n"
        "target: gap=30.00 rel_speed=-5.00\n"
        "left: front_gap=80.00 rear_gap=60.00\n"
        "allowed: maintain-speed, accelerate"
    )
    results = [
        _episode(2, status="error", exception_type="TimeoutError"),
        _episode(0, success=True, step_records=[_step(obs_summary=full_obs)]),
    ]
    batch = _batch(results, seeds=[0, 2])
    config = {
        "highway_env": "1.12.0",
        "nested": {"max_steps": 40},
    }

    write_outputs(tmp_path, batch, results, config)

    results_bytes = (tmp_path / "results.json").read_bytes()
    config_bytes = (tmp_path / "config.json").read_bytes()
    results_payload = json.loads(results_bytes)
    assert results_bytes.endswith(b"\n")
    assert config_bytes.endswith(b"\n")
    assert results_payload["seeds"] == [0, 2]
    assert [item["seed"] for item in results_payload["results"]] == [0, 2]
    assert all(
        "step_records" not in item for item in results_payload["results"]
    )
    assert json.loads(config_bytes) == config

    trace_lines = (tmp_path / "episode_0.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(trace_lines) == 1
    trace = json.loads(trace_lines[0])
    assert trace == asdict(results[1].step_records[0])
    assert trace["obs_summary"] == full_obs


def test_write_outputs_creates_empty_trace_file(
    tmp_path: Path,
) -> None:
    results = [_episode(0)]

    write_outputs(tmp_path, _batch(results), results, config={})

    assert (tmp_path / "episode_0.jsonl").read_bytes() == b""


def test_write_outputs_rejects_batch_result_seed_inconsistency(
    tmp_path: Path,
) -> None:
    results = [_episode(0), _episode(1)]
    batch = _batch(results)
    batch.results.reverse()

    with pytest.raises(ValueError, match="batch.*results|seed"):
        write_outputs(tmp_path, batch, results, config={})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("requested_episodes", 99),
        ("completed_episodes", 99),
        ("success_rate", 0.75),
        ("collision_rate", 0.75),
        ("mean_return", 999.0),
        ("schema_version", "mutated"),
    ],
)
def test_write_outputs_rejects_mutated_batch(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    results = [_episode(0, success=True)]
    batch = replace(_batch(results), **{field: value})

    with pytest.raises(ValueError, match="batch"):
        write_outputs(tmp_path, batch, results, config={})

    assert not (tmp_path / "results.json").exists()


def test_write_outputs_rejects_duplicate_episode_files(
    tmp_path: Path,
) -> None:
    original = [_episode(0)]
    batch = _batch(original)

    with pytest.raises(ValueError, match="duplicate"):
        write_outputs(
            tmp_path,
            batch,
            [_episode(0), _episode(0)],
            config={},
        )

    assert not (tmp_path / "episode_0.jsonl").exists()


@pytest.mark.parametrize(
    "config",
    [
        {"threshold": float("nan")},
        {"unsupported": object()},
    ],
)
def test_write_outputs_rejects_invalid_config(
    tmp_path: Path,
    config: dict[str, object],
) -> None:
    results = [_episode(0)]

    with pytest.raises((TypeError, ValueError)):
        write_outputs(tmp_path, _batch(results), results, config=config)

    assert not (tmp_path / "results.json").exists()
    assert not (tmp_path / "config.json").exists()


def test_write_outputs_rejects_nonfinite_trace_values(
    tmp_path: Path,
) -> None:
    step = replace(_step(), latency_ms=float("nan"))
    results = [_episode(0, step_records=[step])]

    with pytest.raises(ValueError, match="JSON"):
        write_outputs(tmp_path, _batch(results), results, config={})

    assert not (tmp_path / "episode_0.jsonl").exists()


def test_write_outputs_rejects_nonempty_run_dir_without_changes(
    tmp_path: Path,
) -> None:
    results = [_episode(0)]
    out = tmp_path / "run"
    out.mkdir()
    (out / "results.json").write_bytes(b"old-results\n")
    (out / "sentinel.bin").write_bytes(b"\x00\x01old")
    before = {
        path.name: path.read_bytes()
        for path in out.iterdir()
    }

    with pytest.raises(ValueError, match="non-empty"):
        write_outputs(out, _batch(results), results, config={})

    after = {
        path.name: path.read_bytes()
        for path in out.iterdir()
    }
    assert after == before
    assert list(tmp_path.glob(".run.staging-*")) == []


def test_stage_failure_does_not_publish_partial_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world2skills.evaluation import metrics

    results = [_episode(0, step_records=[_step()])]
    out = tmp_path / "run"
    real_write = getattr(metrics, "_write_staged_file", None)
    calls = 0

    def fail_second_write(path: Path, text: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("stage write failed")
        if real_write is not None:
            real_write(path, text)

    monkeypatch.setattr(
        metrics,
        "_write_staged_file",
        fail_second_write,
        raising=False,
    )

    with pytest.raises(OSError, match="stage write failed"):
        write_outputs(out, _batch(results), results, config={})

    assert not out.exists()
    assert list(tmp_path.glob(".run.staging-*")) == []


def test_write_outputs_publishes_over_existing_empty_directory(
    tmp_path: Path,
) -> None:
    results = [_episode(0, step_records=[_step()])]
    out = tmp_path / "run"
    out.mkdir()

    write_outputs(out, _batch(results), results, config={"seed": 0})

    assert sorted(path.name for path in out.iterdir()) == [
        "config.json",
        "episode_0.jsonl",
        "results.json",
    ]
    assert json.loads((out / "config.json").read_text()) == {"seed": 0}
