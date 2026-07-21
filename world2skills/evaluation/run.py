"""Command-line entry point for the M1 experiment loop."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import math
from numbers import Real
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml

from ..runtime.env_factory import (
    build_custom_env_config,
    make_env,
    resolve_env_config,
)
from ..runtime.executor import LLMSkillExecutor
from ..runtime.llm import (
    DEFAULT_AZURE_API_VERSION,
    DEFAULT_AZURE_PROXY_URL,
    DEFAULT_OPENAI_BASE_URL,
    AzureResponsesClient,
    Message,
    MockLLMClient,
    OpenAIClient,
)
from ..runtime.scenario import LaneChangeOvertakeScenario
from ..runtime.skill_loader import SKILLS_DIR, load_skill, select_grounding
from ..runtime.types import EpisodeResult, Grounding, SkillCard
from .episode import run_episode
from .metrics import aggregate, write_outputs


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "configs"
    / "lane-change-overtake.yaml"
)
_CONFIG_KEYS = frozenset(
    {"skill", "backend", "prompt_version", "max_steps", "scenario"}
)
_REQUIRED_CONFIG_KEYS = _CONFIG_KEYS
_SCENARIO_KEYS = frozenset(
    {
        "success_margin",
        "lead_speed_ratio",
        "target_lane_front_clearance_m",
        "target_lane_rear_clearance_m",
        "spawn_gap",
    }
)
_SCENARIO_DEFAULTS = {
    "success_margin": 5.0,
    "lead_speed_ratio": 0.6,
    "target_lane_front_clearance_m": 20.0,
    "target_lane_rear_clearance_m": 20.0,
    "spawn_gap": 30.0,
}
_SENSITIVE_ENV_MARKERS = (
    "API_KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "BASE_URL",
    "ENDPOINT",
)
_DISTRIBUTIONS = (
    "highway-env",
    "gymnasium",
    "openai",
    "diskcache",
    "numpy",
    "PyYAML",
    "jsonschema",
    "pytest",
)
_EGO_LANE = re.compile(r"^Ego: .* lane=(?P<lane>\d+)", re.MULTILINE)
_ENDPOINT_PATH_HASH_LENGTH = 16


def _nonnegative_seed(value: str) -> int:
    try:
        seed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be nonnegative integers") from exc
    if seed < 0:
        raise argparse.ArgumentTypeError("seeds must be nonnegative integers")
    return seed


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the World2Skills M1 highway-env experiment loop"
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="experiment YAML (default: checked-in M1 config)",
    )
    parser.add_argument("--skill", help="override config skill")
    parser.add_argument("--backend", help="override config backend")
    parser.add_argument(
        "--prompt-version",
        help="override config prompt_version",
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_int,
        help="override config max_steps",
    )
    parser.add_argument(
        "--success-margin",
        type=float,
        help="override scenario success_margin",
    )
    parser.add_argument(
        "--lead-speed-ratio",
        type=float,
        help="override scenario lead_speed_ratio",
    )
    parser.add_argument(
        "--target-lane-front-clearance-m",
        type=float,
        help="override scenario target_lane_front_clearance_m",
    )
    parser.add_argument(
        "--target-lane-rear-clearance-m",
        type=float,
        help="override scenario target_lane_rear_clearance_m",
    )
    parser.add_argument(
        "--spawn-gap",
        type=float,
        help="override scenario spawn_gap",
    )
    parser.add_argument(
        "--seeds",
        type=_nonnegative_seed,
        nargs="+",
        required=True,
    )
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="use a deterministic prompt-aware mock policy",
    )
    parser.add_argument("--out", required=True)
    return parser


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field} must be a non-empty string")
    return value


def _positive_config_int(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise TypeError(f"{field} must be a positive integer")
    return value


def _finite_scenario_number(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise TypeError(f"scenario.{field} must be a finite number")
    return float(value)


def _load_yaml_config(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read experiment config {path}: {exc}") from exc
    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"experiment config is invalid YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        raise TypeError("experiment config must be a mapping")

    unknown = sorted(set(loaded) - _CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    missing = sorted(_REQUIRED_CONFIG_KEYS - set(loaded))
    if missing:
        raise ValueError(f"missing required config keys: {missing}")

    skill = _nonempty_string(loaded["skill"], "skill")
    backend = _nonempty_string(loaded["backend"], "backend")
    prompt_version = _nonempty_string(
        loaded["prompt_version"],
        "prompt_version",
    )
    max_steps = _positive_config_int(loaded["max_steps"], "max_steps")
    raw_scenario = loaded["scenario"]
    if not isinstance(raw_scenario, dict):
        raise TypeError("scenario must be a mapping")
    unknown_scenario = sorted(set(raw_scenario) - _SCENARIO_KEYS)
    if unknown_scenario:
        raise ValueError(f"unknown scenario keys: {unknown_scenario}")
    scenario = _SCENARIO_DEFAULTS.copy()
    scenario.update(
        {
            key: _finite_scenario_number(value, key)
            for key, value in raw_scenario.items()
        }
    )
    return (
        {
            "skill": skill,
            "backend": backend,
            "prompt_version": prompt_version,
            "max_steps": max_steps,
            "scenario": scenario,
        },
        hashlib.sha256(raw).hexdigest(),
    )


def _resolved_config(
    parsed: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, str]]:
    source_path = Path(parsed.config).expanduser().resolve()
    config, source_hash = _load_yaml_config(source_path)
    for key in ("skill", "backend", "prompt_version", "max_steps"):
        value = getattr(parsed, key)
        if value is not None:
            config[key] = value

    cli_scenario = {
        "success_margin": parsed.success_margin,
        "lead_speed_ratio": parsed.lead_speed_ratio,
        "target_lane_front_clearance_m": (parsed.target_lane_front_clearance_m),
        "target_lane_rear_clearance_m": (parsed.target_lane_rear_clearance_m),
        "spawn_gap": parsed.spawn_gap,
    }
    for key, value in cli_scenario.items():
        if value is not None:
            config["scenario"][key] = _finite_scenario_number(value, key)

    config["skill"] = _nonempty_string(config["skill"], "skill")
    config["backend"] = _nonempty_string(config["backend"], "backend")
    config["prompt_version"] = _nonempty_string(
        config["prompt_version"],
        "prompt_version",
    )
    config["max_steps"] = _positive_config_int(
        config["max_steps"],
        "max_steps",
    )
    if len(set(parsed.seeds)) != len(parsed.seeds):
        raise ValueError("seeds must be unique")
    return config, {"path": str(source_path), "sha256": source_hash}


def _validate_output_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink() or not path.is_dir():
        raise ValueError("output path must be absent or an empty directory")
    if next(path.iterdir(), None) is not None:
        raise ValueError("output run directory is non-empty; expected empty or absent")


def _mock_overtake_policy(messages: list[Message]) -> str:
    prompt = messages[-1].content
    lane_match = _EGO_LANE.search(prompt)
    if lane_match is None:
        raise ValueError("mock policy prompt omitted ego lane")
    if "Previous primitive: none" in prompt:
        primitive = "change-lane-left"
    elif int(lane_match.group("lane")) == 1:
        primitive = "maintain-speed"
    else:
        primitive = "accelerate"
    return f'{{"primitive": "{primitive}"}}'


def _build_client(model: str, mock: bool) -> Any:
    if mock:
        client = MockLLMClient(_mock_overtake_policy)
        client.model = "mock"
        client.api_type = "mock"
        client.base_url = "mock://local/"
        client.api_version = ""
        return client
    if model.startswith("gpt-5"):
        actual_model = os.getenv("OPENAI_MODEL") or model
        return AzureResponsesClient(model=actual_model)
    return OpenAIClient(model=model)


def _requested_client(model: str, mock: bool) -> dict[str, Any]:
    if mock:
        return {
            "model": model,
            "client_type": "MockLLMClient",
            "base_url": "mock://local/",
            "api_version": "",
        }
    if model.startswith("gpt-5"):
        return {
            "model": model,
            "client_type": "AzureResponsesClient",
            "base_url": _sanitize_endpoint(
                os.getenv("OPENAI_BASE_URL") or DEFAULT_AZURE_PROXY_URL
            ),
            "api_version": (
                os.getenv("OPENAI_API_VERSION") or DEFAULT_AZURE_API_VERSION
            ),
        }
    return {
        "model": model,
        "client_type": "OpenAIClient",
        "base_url": _sanitize_endpoint(
            os.getenv("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL
        ),
        "api_version": "",
    }


def _string_attribute(client: Any, name: str) -> str | None:
    value = getattr(client, name, None)
    if value is None:
        return None
    return str(value)


def _sanitize_endpoint(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        if hostname is None:
            return None
        host = f"[{hostname}]" if ":" in hostname else hostname
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        raw_path = parsed.path
        raw_query = parsed.query
        if raw_path in {"", "/"} and not raw_query:
            path = "/"
        else:
            identity = raw_path.encode("utf-8") + b"\0" + raw_query.encode("utf-8")
            path_hash = hashlib.sha256(identity).hexdigest()[
                :_ENDPOINT_PATH_HASH_LENGTH
            ]
            path = f"/_path_sha256_{path_hash}/"
        return urlunsplit((parsed.scheme, host, path, "", ""))
    except ValueError:
        return None


def _client_snapshot(
    client: Any | None,
    construction_error: Exception | None,
) -> dict[str, Any]:
    if client is None:
        return {
            "constructed": False,
            "type": None,
            "model": None,
            "api_type": None,
            "base_url": None,
            "api_version": None,
            "construction_error": _exception_text(construction_error),
            "close_error": None,
        }
    return {
        "constructed": True,
        "type": type(client).__name__,
        "model": _string_attribute(client, "model"),
        "api_type": _string_attribute(client, "api_type"),
        "base_url": _sanitize_endpoint(_string_attribute(client, "base_url")),
        "api_version": _string_attribute(client, "api_version"),
        "construction_error": None,
        "close_error": None,
    }


def _exception_text(error: Exception | None) -> str | None:
    if error is None:
        return None
    return f"{type(error).__name__}: {_redact_secrets(str(error))}"


def _redact_secrets(text: str) -> str:
    redacted = text
    for name, value in os.environ.items():
        if value and any(marker in name.upper() for marker in _SENSITIVE_ENV_MARKERS):
            redacted = redacted.replace(value, "<redacted>")
    return redacted


def _close_resource(resource: Any) -> str | None:
    try:
        resource.close()
    except Exception as exc:  # noqa: BLE001 - cleanup must remain observable
        return _exception_text(exc)
    return None


def _scenario_value(scenario: Any | None, name: str, default: Any) -> Any:
    if scenario is None:
        return default
    try:
        return getattr(scenario, name, default)
    except Exception:
        return default


def _error_episode(
    seed: int,
    error: Exception,
    *,
    scenario: Any | None = None,
    cleanup_error: str | None = None,
) -> EpisodeResult:
    lead_gap = _scenario_value(scenario, "lead_initial_gap_m", None)
    if not isinstance(lead_gap, Real) or isinstance(lead_gap, bool):
        lead_gap = None
    return EpisodeResult(
        seed=seed,
        status="error",
        success=False,
        success_reason="",
        episode_return=0.0,
        crashed=False,
        steps=0,
        mean_speed=0.0,
        parse_failures=0,
        unavailable_action_attempts=0,
        llm_errors=0,
        terminated=False,
        truncated=False,
        max_steps_reached=False,
        scenario_completed=False,
        termination_reason="error",
        target_initially_ahead=bool(
            _scenario_value(scenario, "target_initially_ahead", False)
        ),
        lane_change_completed_step=None,
        overtake_step=None,
        lead_initial_gap_m=float(lead_gap) if lead_gap is not None else None,
        exception_type=type(error).__name__,
        exception_message=_redact_secrets(str(error)),
        cleanup_error=cleanup_error,
    )


def _skill_hash(card: SkillCard) -> str:
    skill_dir = SKILLS_DIR / card.name
    digest = hashlib.sha256()
    for filename in ("SKILL.md", "skill.yaml"):
        path = skill_dir / filename
        digest.update(filename.encode("ascii"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _grounding_snapshot(grounding: Grounding) -> dict[str, Any]:
    return {
        "backend": grounding.backend,
        "backend_version": grounding.backend_version,
        "environment": grounding.environment,
        "observation": deepcopy(grounding.observation),
        "action": deepcopy(grounding.action),
        "primitive_map": deepcopy(grounding.primitive_map),
    }


def _installed_versions() -> dict[str, str]:
    return {
        distribution: metadata.version(distribution) for distribution in _DISTRIBUTIONS
    }


def _git_commit() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    commit = completed.stdout.strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError("git rev-parse HEAD did not return a full commit hash")
    return commit.lower()


def _run_seed(
    *,
    seed: int,
    card: SkillCard,
    grounding: Grounding,
    client: Any,
    prompt_version: str,
    max_steps: int,
    scenario_parameters: dict[str, float],
) -> tuple[
    EpisodeResult,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    env = None
    handed_to_runner = False
    scenario = None
    resolved_env_config = None
    scenario_snapshot = None
    try:
        scenario = LaneChangeOvertakeScenario(
            card,
            **scenario_parameters,
        )
        scenario_snapshot = scenario.config_snapshot()
        env, name_to_index = make_env(
            grounding,
            scenario.configure(),
            seed=seed,
        )
        resolved_env_config = deepcopy(env.unwrapped.config)
        executor = LLMSkillExecutor(
            card,
            grounding,
            client,
            prompt_version,
            name_to_index,
        )
        handed_to_runner = True
        return (
            run_episode(
                env,
                executor,
                scenario,
                seed=seed,
                max_steps=max_steps,
            ),
            resolved_env_config,
            scenario_snapshot,
        )
    except Exception as exc:  # noqa: BLE001 - isolate one seed
        cleanup_error = None
        if env is not None and not handed_to_runner:
            cleanup_error = _close_resource(env)
        return (
            _error_episode(
                seed,
                exc,
                scenario=scenario,
                cleanup_error=cleanup_error,
            ),
            resolved_env_config,
            scenario_snapshot,
        )


def _fallback_scenario_snapshot(
    card: SkillCard,
    scenario_parameters: dict[str, float],
) -> dict[str, Any]:
    try:
        scenario = LaneChangeOvertakeScenario(
            card,
            **scenario_parameters,
        )
        return scenario.config_snapshot()
    except Exception as exc:  # noqa: BLE001 - preserve client-failure artifacts
        return {
            **scenario_parameters,
            "snapshot_error": _exception_text(exc),
        }


def _resolved_snapshot(
    *,
    parsed: argparse.Namespace,
    config: dict[str, Any],
    source_config: dict[str, str],
    card: SkillCard,
    grounding: Grounding,
    scenario_snapshot: dict[str, Any],
    client_snapshot: dict[str, Any],
    environment_config: dict[str, Any] | None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "timestamp": now.isoformat(),
        "date": now.date().isoformat(),
        "source_config": source_config,
        "requested": _requested_client(parsed.model, parsed.mock),
        "client": client_snapshot,
        "prompt_version": config["prompt_version"],
        "seeds": list(parsed.seeds),
        "max_steps": config["max_steps"],
        "skill": {
            "name": card.name,
            "schema_version": card.schema_version,
            "version": card.version,
            "sha256": _skill_hash(card),
        },
        "grounding": _grounding_snapshot(grounding),
        "scenario": scenario_snapshot,
        "environment_config": environment_config,
        "installed_versions": _installed_versions(),
        "git_commit": _git_commit(),
    }


def main(argv: list[str] | None = None) -> int:
    parsed = _parser().parse_args(argv)
    config, source_config = _resolved_config(parsed)
    out = Path(parsed.out).expanduser()
    _validate_output_path(out)

    card = load_skill(config["skill"])
    if card.name != "lane-change-overtake":
        raise ValueError("M1 CLI supports only the lane-change-overtake skill")
    grounding = select_grounding(card, config["backend"])

    client = None
    construction_error = None
    results: list[EpisodeResult] = []
    environment_config = None
    scenario_snapshot = None
    try:
        client = _build_client(parsed.model, parsed.mock)
    except Exception as exc:  # noqa: BLE001 - publish auditable error batch
        construction_error = exc
        results = [_error_episode(seed, construction_error) for seed in parsed.seeds]
    else:
        try:
            for seed in parsed.seeds:
                result, seed_env_config, seed_scenario_snapshot = _run_seed(
                    seed=seed,
                    card=card,
                    grounding=grounding,
                    client=client,
                    prompt_version=config["prompt_version"],
                    max_steps=config["max_steps"],
                    scenario_parameters=config["scenario"],
                )
                results.append(result)
                if environment_config is None and seed_env_config is not None:
                    environment_config = seed_env_config
                if scenario_snapshot is None and seed_scenario_snapshot is not None:
                    scenario_snapshot = seed_scenario_snapshot
        finally:
            close_error = _close_resource(client)

    client_snapshot = _client_snapshot(client, construction_error)
    if client is not None:
        client_snapshot["close_error"] = close_error
        if close_error is not None:
            print(
                f"[w2s] client close failed: {close_error}",
                file=sys.stderr,
            )

    actual_model = client_snapshot["model"] or parsed.model
    if scenario_snapshot is None:
        scenario_snapshot = _fallback_scenario_snapshot(
            card,
            config["scenario"],
        )
    if environment_config is None:
        scenario_environment_config = scenario_snapshot.get("environment_config")
        if not isinstance(scenario_environment_config, dict):
            scenario_environment_config = LaneChangeOvertakeScenario.configure()
        try:
            environment_config = resolve_env_config(
                grounding,
                scenario_environment_config,
            )
        except Exception as exc:  # noqa: BLE001 - preserve error-run publication
            try:
                custom_config: Any = build_custom_env_config(
                    grounding,
                    scenario_environment_config,
                )
            except Exception as custom_exc:  # noqa: BLE001
                custom_config = {
                    "build_status": "error",
                    "build_error": _exception_text(custom_exc),
                }
            environment_config = {
                "resolution_status": "error",
                "resolution_error": _exception_text(exc),
                "custom_config": custom_config,
            }
    batch = aggregate(
        results,
        model=actual_model,
        prompt_version=config["prompt_version"],
        skill=card.name,
        environment=grounding.environment,
        seeds=list(parsed.seeds),
    )
    snapshot = _resolved_snapshot(
        parsed=parsed,
        config=config,
        source_config=source_config,
        card=card,
        grounding=grounding,
        scenario_snapshot=scenario_snapshot,
        client_snapshot=client_snapshot,
        environment_config=environment_config,
    )
    write_outputs(out, batch, results, snapshot)
    print(
        f"[w2s] wrote {out / 'results.json'} "
        f"success_rate={batch.success_rate:.2f} "
        f"({batch.successful_episodes}/{batch.requested_episodes}), "
        f"errors={batch.error_episodes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
