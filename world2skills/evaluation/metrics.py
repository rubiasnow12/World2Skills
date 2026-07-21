"""Aggregate episode metrics and write strict, auditable run artifacts."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from numbers import Real
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from ..runtime.types import BatchResult, EpisodeResult, StepRecord


class ArtifactPublicationError(RuntimeError):
    """Artifact publication failed after the output became visible."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        published: bool,
    ) -> None:
        self.path = Path(path)
        self.published = published
        state = "published" if published else "not published"
        super().__init__(
            f"artifact publication durability failed for "
            f"{self.path} ({state})"
        )


SCHEMA_VERSION = "0.1"
_VALID_STATUSES = frozenset({"ok", "error"})
_VALID_DECISION_STATUSES = frozenset(
    {
        "ok",
        "parse_fallback",
        "unavailable_fallback",
        "llm_error_fallback",
    }
)
_FALLBACK_COUNTERS = {
    "parse_fallback": "parse_failures",
    "unavailable_fallback": "unavailable_action_attempts",
    "llm_error_fallback": "llm_errors",
}
_BOOLEAN_FIELDS = (
    "success",
    "crashed",
    "terminated",
    "truncated",
    "max_steps_reached",
    "scenario_completed",
    "target_initially_ahead",
)
_NONNEGATIVE_INTEGER_FIELDS = (
    "steps",
    "parse_failures",
    "unavailable_action_attempts",
    "llm_errors",
)
_OPTIONAL_NONNEGATIVE_INTEGER_FIELDS = (
    "lane_change_completed_step",
    "overtake_step",
)
_FINITE_NUMBER_FIELDS = (
    "episode_return",
    "mean_speed",
)
_OPTIONAL_FINITE_NUMBER_FIELDS = ("lead_initial_gap_m",)
_RETURN_REL_TOLERANCE = 1e-12
_RETURN_ABS_TOLERANCE = 1e-12


def episode_to_result_dict(result: EpisodeResult) -> dict[str, Any]:
    """Return the episode summary, excluding only its per-step trace."""

    if not isinstance(result, EpisodeResult):
        raise TypeError("result must be an EpisodeResult")
    summary = asdict(result)
    del summary["step_records"]
    return summary


def _validate_seeds(seeds: object) -> list[int]:
    if type(seeds) is not list:
        raise TypeError("seeds must be a non-empty list")
    if not seeds:
        raise ValueError("seeds must be a non-empty list")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must contain nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    return list(seeds)


def _ordered_results(
    results: object,
    seeds: list[int],
) -> list[EpisodeResult]:
    if type(results) is not list:
        raise TypeError("results must be a list of EpisodeResult values")
    if any(not isinstance(result, EpisodeResult) for result in results):
        raise TypeError("results must contain only EpisodeResult values")

    result_seeds = [result.seed for result in results]
    if any(type(seed) is not int or seed < 0 for seed in result_seeds):
        raise ValueError(
            "result seeds must be nonnegative integers"
        )
    if len(set(result_seeds)) != len(result_seeds):
        raise ValueError("results contain duplicate seeds")

    requested = set(seeds)
    actual = set(result_seeds)
    missing = requested - actual
    extra = actual - requested
    if missing:
        raise ValueError(f"results are missing requested seeds: {sorted(missing)}")
    if extra:
        raise ValueError(f"results contain extra seeds: {sorted(extra)}")

    by_seed = {result.seed: result for result in results}
    ordered = [by_seed[seed] for seed in seeds]
    for result in ordered:
        if result.status not in _VALID_STATUSES:
            raise ValueError(
                "episode status must be exactly 'ok' or 'error'"
            )
        _validate_episode_summary(result)
    return ordered


def _validate_episode_summary(result: EpisodeResult) -> None:
    for field_name in _BOOLEAN_FIELDS:
        if type(getattr(result, field_name)) is not bool:
            raise ValueError(f"{field_name} must be bool")

    for field_name in _NONNEGATIVE_INTEGER_FIELDS:
        value = getattr(result, field_name)
        if type(value) is not int or value < 0:
            raise ValueError(
                f"{field_name} must be a nonnegative integer"
            )

    for field_name in _OPTIONAL_NONNEGATIVE_INTEGER_FIELDS:
        value = getattr(result, field_name)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(
                f"{field_name} must be a nonnegative integer or None"
            )
        if value is not None and value >= result.steps:
            raise ValueError(
                f"{field_name} must be less than steps"
            )

    if result.steps != len(result.step_records):
        raise ValueError("steps must equal len(step_records)")
    trace_counts, trace_return, trace_crashed = _validate_step_records(
        result.step_records,
        status=result.status,
    )

    for field_name in _FINITE_NUMBER_FIELDS:
        _validate_finite_number(getattr(result, field_name), field_name)

    for field_name in _OPTIONAL_FINITE_NUMBER_FIELDS:
        value = getattr(result, field_name)
        if value is not None:
            _validate_finite_number(value, field_name)

    if result.mean_speed < 0:
        raise ValueError("mean_speed must be finite and nonnegative")
    if (
        result.lead_initial_gap_m is not None
        and result.lead_initial_gap_m <= 0
    ):
        raise ValueError(
            "lead_initial_gap_m must be finite and positive"
        )

    if not math.isclose(
        result.episode_return,
        trace_return,
        rel_tol=_RETURN_REL_TOLERANCE,
        abs_tol=_RETURN_ABS_TOLERANCE,
    ):
        raise ValueError(
            "episode_return must equal the sum of recorded rewards"
        )
    if result.crashed != trace_crashed:
        raise ValueError(
            "crashed must equal any(record.crashed) in step_records"
        )

    fallback_count = (
        result.parse_failures
        + result.unavailable_action_attempts
        + result.llm_errors
    )
    fallback_limit = result.steps + (result.status == "error")
    if fallback_count > fallback_limit:
        limit_name = (
            "steps+1 for error episodes"
            if result.status == "error"
            else "steps"
        )
        raise ValueError(
            f"fallback counter sum must not exceed {limit_name}"
        )
    _validate_fallback_counters(result, trace_counts)

    if result.status == "error":
        if result.success:
            raise ValueError("error episode cannot report success")
        return

    if result.success:
        _validate_success_contract(result)
    elif result.scenario_completed:
        raise ValueError(
            "scenario_completed=True requires success=True"
        )


def _validate_success_contract(result: EpisodeResult) -> None:
    if result.status != "ok":
        raise ValueError("success=True requires status='ok'")
    if result.crashed:
        raise ValueError("success=True requires crashed=False")
    if not result.scenario_completed:
        raise ValueError(
            "success=True requires scenario_completed=True"
        )
    if not result.target_initially_ahead:
        raise ValueError(
            "success=True requires target_initially_ahead=True"
        )
    if (
        result.lane_change_completed_step is None
        or result.overtake_step is None
    ):
        raise ValueError(
            "success=True requires non-None causal steps"
        )
    if result.lane_change_completed_step >= result.overtake_step:
        raise ValueError(
            "lane_change_completed_step must precede overtake_step"
        )
    if (
        not isinstance(result.success_reason, str)
        or not result.success_reason.strip()
    ):
        raise ValueError(
            "success=True requires a non-empty success_reason"
        )


def _validate_step_records(
    records: list[StepRecord],
    *,
    status: str,
) -> tuple[dict[str, int], float, bool]:
    counts = {
        "parse_failures": 0,
        "unavailable_action_attempts": 0,
        "llm_errors": 0,
    }
    rewards: list[Real] = []
    crashed = False
    for index, record in enumerate(records):
        prefix = f"step_records[{index}]"
        if not isinstance(record, StepRecord):
            raise ValueError(f"{prefix} must be a StepRecord")
        if type(record.t) is not int or record.t != index:
            raise ValueError(
                f"{prefix}.t must equal its sequential list index"
            )
        _validate_nonempty_string(
            record.obs_summary,
            f"{prefix}.obs_summary",
        )
        _validate_nonempty_string(record.primitive, f"{prefix}.primitive")
        _validate_nonempty_string(
            record.backend_action,
            f"{prefix}.backend_action",
        )
        if type(record.action_index) is not int or record.action_index < 0:
            raise ValueError(
                f"{prefix}.action_index must be a nonnegative integer"
            )
        if record.reward is None:
            if status != "error" or index != len(records) - 1:
                raise ValueError(
                    f"{prefix}.reward may be None only for "
                    "status='error' on the final record"
                )
        else:
            _validate_step_number(
                record.reward,
                f"{prefix}.reward",
                nonnegative=False,
            )
            rewards.append(record.reward)
        for field_name in ("crashed", "cache_hit"):
            if type(getattr(record, field_name)) is not bool:
                raise ValueError(f"{prefix}.{field_name} must be bool")
        crashed = crashed or record.crashed
        _validate_nonempty_string(
            record.request_hash,
            f"{prefix}.request_hash",
        )
        if not isinstance(record.raw_response, str):
            raise ValueError(f"{prefix}.raw_response must be str")
        _validate_step_number(
            record.latency_ms,
            f"{prefix}.latency_ms",
            nonnegative=True,
        )
        if (
            not isinstance(record.decision_status, str)
            or record.decision_status not in _VALID_DECISION_STATUSES
        ):
            raise ValueError(
                f"{prefix}.decision_status is invalid"
            )
        if record.decision_status == "ok":
            if record.fallback_reason is not None:
                raise ValueError(
                    f"{prefix}.fallback_reason must be None for ok decisions"
                )
        else:
            _validate_nonempty_string(
                record.fallback_reason,
                f"{prefix}.fallback_reason",
            )
            counter_name = _FALLBACK_COUNTERS[record.decision_status]
            counts[counter_name] += 1

        primitives = record.available_primitives
        if (
            type(primitives) is not list
            or not primitives
            or any(
                not isinstance(primitive, str) or not primitive.strip()
                for primitive in primitives
            )
            or len(set(primitives)) != len(primitives)
        ):
            raise ValueError(
                f"{prefix}.available_primitives must be unique "
                "non-empty strings"
            )
        if record.primitive not in primitives:
            raise ValueError(
                f"{prefix}.available_primitives must contain "
                "the chosen primitive"
            )
    try:
        episode_return = math.fsum(rewards)
    except OverflowError as exc:
        raise ValueError(
            "recorded rewards must have a finite sum"
        ) from exc
    return counts, episode_return, crashed


def _validate_fallback_counters(
    result: EpisodeResult,
    trace_counts: dict[str, int],
) -> None:
    untraced_total = 0
    for field_name, trace_count in trace_counts.items():
        summary_count = getattr(result, field_name)
        if trace_count > summary_count:
            raise ValueError(
                f"{field_name} trace count exceeds episode summary"
            )
        if result.status == "ok" and trace_count != summary_count:
            raise ValueError(
                f"{field_name} must exactly match trace count "
                "for ok episodes"
            )
        untraced_total += summary_count - trace_count

    if result.status == "error" and untraced_total > 1:
        raise ValueError(
            "error episode fallback summary may exceed trace "
            "counts by at most one decision"
        )


def _validate_nonempty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_step_number(
    value: object,
    field_name: str,
    *,
    nonnegative: bool,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or (nonnegative and value < 0)
    ):
        qualifier = "finite nonnegative" if nonnegative else "finite"
        raise ValueError(
            f"{field_name} must be a {qualifier} Real for strict JSON"
        )


def _validate_finite_number(value: object, field_name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{field_name} must be a finite number")


def _strict_json(
    value: object,
    *,
    label: str,
    indent: int | None = None,
) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        )
    except ValueError as exc:
        raise ValueError(f"{label} must contain finite JSON numbers") from exc


def aggregate(
    results: list[EpisodeResult],
    *,
    model: str,
    prompt_version: str,
    skill: str,
    environment: str,
    seeds: list[int],
) -> BatchResult:
    """Validate and aggregate exactly one episode for every requested seed."""

    for field_name, value in (
        ("model", model),
        ("prompt_version", prompt_version),
        ("skill", skill),
        ("environment", environment),
    ):
        _validate_nonempty_string(value, field_name)
    validated_seeds = _validate_seeds(seeds)
    ordered = _ordered_results(results, validated_seeds)
    summaries = [episode_to_result_dict(result) for result in ordered]
    _strict_json(summaries, label="episode summaries")

    requested = len(validated_seeds)
    completed = sum(result.status == "ok" for result in ordered)
    errors = sum(result.status == "error" for result in ordered)
    if completed + errors != requested:
        raise ValueError(
            "completed and error episodes must equal requested episodes"
        )

    successful = sum(
        result.status == "ok" and result.success for result in ordered
    )
    collisions = sum(result.crashed for result in ordered)
    ok_returns = [
        result.episode_return
        for result in ordered
        if result.status == "ok"
    ]
    mean_return = (
        math.fsum(ok_returns) / len(ok_returns) if ok_returns else 0.0
    )
    if not math.isfinite(mean_return):
        raise ValueError("mean_return must be finite")

    batch = BatchResult(
        schema_version=SCHEMA_VERSION,
        model=model,
        prompt_version=prompt_version,
        skill=skill,
        environment=environment,
        seeds=validated_seeds,
        requested_episodes=requested,
        completed_episodes=completed,
        error_episodes=errors,
        successful_episodes=successful,
        success_rate=successful / requested,
        collision_rate=collisions / requested,
        mean_return=mean_return,
        results=summaries,
    )
    _strict_json(asdict(batch), label="aggregate result")
    return batch


def _validate_batch_results(
    batch: BatchResult,
    results: list[EpisodeResult],
) -> list[EpisodeResult]:
    if not isinstance(batch, BatchResult):
        raise TypeError("batch must be a BatchResult")
    expected = aggregate(
        results,
        model=batch.model,
        prompt_version=batch.prompt_version,
        skill=batch.skill,
        environment=batch.environment,
        seeds=batch.seeds,
    )
    if asdict(batch) != asdict(expected):
        raise ValueError(
            "batch results are inconsistent with recomputed episode aggregates"
        )
    by_seed = {result.seed: result for result in results}
    return [by_seed[seed] for seed in expected.seeds]


def _write_staged_file(path: Path, text: str) -> None:
    with path.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_absent_or_empty_directory(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_symlink() or not path.is_dir():
        raise ValueError("output path must be absent or an empty directory")
    if next(path.iterdir(), None) is not None:
        raise ValueError(
            "output run directory is non-empty; expected empty or absent"
        )
    return True


def _publish_staged_directory(
    staging: Path,
    out: Path,
) -> None:
    removed_empty_target = False
    try:
        if _require_absent_or_empty_directory(out):
            out.rmdir()
            removed_empty_target = True
        os.replace(staging, out)
    except Exception:
        if removed_empty_target and not out.exists():
            out.mkdir()
        raise
    try:
        _fsync_directory(out.parent)
    except Exception as exc:
        raise ArtifactPublicationError(
            out,
            published=True,
        ) from exc


def write_outputs(
    out_dir: str | os.PathLike[str],
    batch: BatchResult,
    results: list[EpisodeResult],
    config: dict[str, Any],
) -> None:
    """Write summaries, the supplied config, and one full trace per seed."""

    ordered = _validate_batch_results(batch, results)
    if type(config) is not dict:
        raise TypeError("config must be a dict")

    results_text = (
        _strict_json(asdict(batch), label="results.json", indent=2) + "\n"
    )
    config_text = (
        _strict_json(config, label="config.json", indent=2) + "\n"
    )
    trace_texts: dict[int, str] = {}
    for result in ordered:
        lines = [
            _strict_json(asdict(record), label=f"episode_{result.seed}.jsonl")
            for record in result.step_records
        ]
        trace_texts[result.seed] = (
            "".join(f"{line}\n" for line in lines) if lines else ""
        )

    out = Path(out_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    _require_absent_or_empty_directory(out)

    staging: Path | None = Path(
        tempfile.mkdtemp(
            dir=out.parent,
            prefix=f".{out.name}.staging-",
        )
    )
    try:
        assert staging is not None
        _write_staged_file(staging / "results.json", results_text)
        _write_staged_file(staging / "config.json", config_text)
        for seed in batch.seeds:
            _write_staged_file(
                staging / f"episode_{seed}.jsonl",
                trace_texts[seed],
            )
        _fsync_directory(staging)
        _publish_staged_directory(staging, out)
        staging = None
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
