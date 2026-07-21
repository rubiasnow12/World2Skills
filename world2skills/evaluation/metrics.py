"""Aggregate episode metrics and write strict, auditable run artifacts."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from ..runtime.types import BatchResult, EpisodeResult


SCHEMA_VERSION = "0.1"
_VALID_STATUSES = frozenset({"ok", "error"})


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
    return ordered


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
    validated_seeds = _validate_seeds(batch.seeds)
    ordered = _ordered_results(results, validated_seeds)
    expected_summaries = [
        episode_to_result_dict(result) for result in ordered
    ]
    if batch.results != expected_summaries:
        raise ValueError(
            "batch results are inconsistent with episode result seed order"
        )
    return ordered


def _atomic_write_text(path: Path, text: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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
    out.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(out / "results.json", results_text)
    _atomic_write_text(out / "config.json", config_text)
    for seed in batch.seeds:
        _atomic_write_text(
            out / f"episode_{seed}.jsonl",
            trace_texts[seed],
        )
