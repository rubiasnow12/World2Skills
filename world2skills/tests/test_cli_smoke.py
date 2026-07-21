from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
from typing import Any

import pytest

pytest.importorskip("highway_env")

from world2skills.evaluation import run
from world2skills.runtime import env_factory as env_factory_module
from world2skills.runtime.types import EpisodeResult, StepRecord


_ACTION_MAP = {
    "LANE_LEFT": 0,
    "IDLE": 1,
    "LANE_RIGHT": 2,
    "FASTER": 3,
    "SLOWER": 4,
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ok_episode(seed: int) -> EpisodeResult:
    return EpisodeResult(
        seed=seed,
        status="ok",
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
        max_steps_reached=True,
        scenario_completed=False,
        termination_reason="max_steps",
        target_initially_ahead=False,
        lane_change_completed_step=None,
        overtake_step=None,
        lead_initial_gap_m=None,
    )


class _FakeEnv:
    def __init__(self, seed: int, config: dict[str, Any] | None = None) -> None:
        self.seed = seed
        self.close_calls = 0
        self.unwrapped = SimpleNamespace(
            config=config
            or {
                "duration": 77,
                "observation": {
                    "type": "Kinematics",
                    "normalize": False,
                },
                "action": {"type": "DiscreteMetaAction"},
            }
        )

    def close(self) -> None:
        self.close_calls += 1


class _FakeClient:
    def __init__(
        self,
        *,
        model: str = "actual-deployment",
        close_error: Exception | None = None,
    ) -> None:
        self.model = model
        self.api_type = "fake-responses"
        self.base_url = "https://user:password@example.test/openai/?api_key=hidden"
        self.api_version = "2099-01-01"
        self.close_error = close_error
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class _FakeExecutor:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


def _patch_fast_success(
    monkeypatch: pytest.MonkeyPatch,
    *,
    client: _FakeClient | None = None,
    env_config: dict[str, Any] | None = None,
) -> tuple[_FakeClient, list[_FakeEnv]]:
    actual_client = client or _FakeClient()
    envs: list[_FakeEnv] = []

    def fake_make_env(
        grounding: Any,
        scenario_config: dict[str, Any],
        seed: int,
    ) -> tuple[_FakeEnv, dict[str, int]]:
        del grounding, scenario_config
        env = _FakeEnv(seed, env_config)
        envs.append(env)
        return env, _ACTION_MAP.copy()

    def fake_run_episode(
        env: _FakeEnv,
        executor: Any,
        scenario: Any,
        seed: int,
        max_steps: int,
    ) -> EpisodeResult:
        del executor, scenario, max_steps
        env.close()
        return _ok_episode(seed)

    monkeypatch.setattr(run, "_build_client", lambda model, mock: actual_client)
    monkeypatch.setattr(run, "make_env", fake_make_env)
    monkeypatch.setattr(run, "LLMSkillExecutor", _FakeExecutor)
    monkeypatch.setattr(run, "run_episode", fake_run_episode)
    return actual_client, envs


def _valid_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "skill": "lane-change-overtake",
        "backend": "highway-env",
        "prompt_version": "lco-v1",
        "max_steps": 40,
        "scenario": {
            "success_margin": 5.0,
            "lead_speed_ratio": 0.6,
            "target_lane_front_clearance_m": 20.0,
            "target_lane_rear_clearance_m": 20.0,
            "spawn_gap": 30.0,
        },
    }
    config.update(overrides)
    return config


def _write_yaml(path: Path, data: object) -> None:
    import yaml

    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_cli_mock_writes_successful_artifacts_and_full_config(
    tmp_path: Path,
) -> None:
    out = tmp_path / "mock-run"

    rc = run.main(
        [
            "--seeds",
            "0",
            "--mock",
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    results = _read_json(out / "results.json")
    assert results["seeds"] == [0]
    assert results["requested_episodes"] == 1
    assert results["completed_episodes"] == 1
    assert results["successful_episodes"] == 1
    assert results["results"][0]["status"] == "ok"
    assert results["results"][0]["success"] is True
    assert (out / "episode_0.jsonl").stat().st_size > 0

    config = _read_json(out / "config.json")
    assert config["source_config"]["path"].endswith(
        "experiments/configs/lane-change-overtake.yaml"
    )
    assert len(config["source_config"]["sha256"]) == 64
    assert config["requested"]["model"] == "gpt-5.4"
    assert config["client"]["type"] == "MockLLMClient"
    assert config["client"]["model"] == "mock"
    assert config["prompt_version"] == "lco-v1"
    assert config["scenario"]["spawn_gap"] == 30.0
    assert config["environment_config"]["observation"]["normalize"] is False


def test_yaml_nondefaults_and_cli_overrides_reach_scenario(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    config = _valid_config(max_steps=45)
    config["scenario"] = {
        "success_margin": 6.0,
        "lead_speed_ratio": 0.5,
        "target_lane_front_clearance_m": 25.0,
        "target_lane_rear_clearance_m": 26.0,
        "spawn_gap": 35.0,
    }
    _write_yaml(config_path, config)
    out = tmp_path / "overrides"

    rc = run.main(
        [
            "--config",
            str(config_path),
            "--seeds",
            "0",
            "--mock",
            "--out",
            str(out),
            "--max-steps",
            "50",
            "--spawn-gap",
            "40",
        ]
    )

    assert rc == 0
    snapshot = _read_json(out / "config.json")
    assert snapshot["max_steps"] == 50
    assert snapshot["scenario"]["success_margin"] == 6.0
    assert snapshot["scenario"]["lead_speed_ratio"] == 0.5
    assert snapshot["scenario"]["target_lane_front_clearance_m"] == 25.0
    assert snapshot["scenario"]["target_lane_rear_clearance_m"] == 26.0
    assert snapshot["scenario"]["spawn_gap"] == 40.0
    result = _read_json(out / "results.json")["results"][0]
    assert result["lead_initial_gap_m"] == pytest.approx(40.0)


def test_middle_seed_setup_failure_is_isolated_and_envs_close_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _FakeClient()
    envs: dict[int, _FakeEnv] = {}

    def fake_make_env(
        grounding: Any,
        scenario_config: dict[str, Any],
        seed: int,
    ) -> tuple[_FakeEnv, dict[str, int]]:
        del grounding, scenario_config
        env = _FakeEnv(seed)
        envs[seed] = env
        mapping = _ACTION_MAP.copy()
        mapping["TEST_SEED"] = seed
        return env, mapping

    class SeedAwareExecutor:
        def __init__(
            self,
            card: Any,
            grounding: Any,
            llm_client: Any,
            prompt_version: str,
            name_to_index: dict[str, int],
        ) -> None:
            del card, grounding, llm_client, prompt_version
            if name_to_index["TEST_SEED"] == 1:
                raise RuntimeError("executor setup failed for seed 1")

    def fake_run_episode(
        env: _FakeEnv,
        executor: Any,
        scenario: Any,
        seed: int,
        max_steps: int,
    ) -> EpisodeResult:
        del executor, scenario, max_steps
        env.close()
        return _ok_episode(seed)

    monkeypatch.setattr(run, "_build_client", lambda model, mock: client)
    monkeypatch.setattr(run, "make_env", fake_make_env)
    monkeypatch.setattr(run, "LLMSkillExecutor", SeedAwareExecutor)
    monkeypatch.setattr(run, "run_episode", fake_run_episode)
    out = tmp_path / "isolated"

    rc = run.main(
        [
            "--seeds",
            "0",
            "1",
            "2",
            "--mock",
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    payload = _read_json(out / "results.json")
    assert [item["seed"] for item in payload["results"]] == [0, 1, 2]
    assert [item["status"] for item in payload["results"]] == [
        "ok",
        "error",
        "ok",
    ]
    assert payload["results"][1]["exception_type"] == "RuntimeError"
    assert payload["completed_episodes"] == 2
    assert payload["error_episodes"] == 1
    assert {seed: env.close_calls for seed, env in envs.items()} == {
        0: 1,
        1: 1,
        2: 1,
    }
    assert client.close_calls == 1


def test_middle_seed_scenario_constructor_failure_is_isolated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, envs = _patch_fast_success(monkeypatch)
    real_scenario = run.LaneChangeOvertakeScenario
    calls = 0

    def scenario_factory(card: Any, **parameters: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("scenario construction failed")
        return real_scenario(card, **parameters)

    monkeypatch.setattr(run, "LaneChangeOvertakeScenario", scenario_factory)
    out = tmp_path / "scenario-error"

    rc = run.main(
        [
            "--seeds",
            "0",
            "1",
            "2",
            "--mock",
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    payload = _read_json(out / "results.json")
    assert [item["status"] for item in payload["results"]] == [
        "ok",
        "error",
        "ok",
    ]
    assert payload["results"][1]["exception_message"] == (
        "scenario construction failed"
    )
    assert [env.seed for env in envs] == [0, 2]
    assert [env.close_calls for env in envs] == [1, 1]
    assert client.close_calls == 1


def test_client_constructor_failure_writes_one_error_per_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_client(model: str, mock: bool) -> Any:
        del model, mock
        raise RuntimeError("client construction failed")

    monkeypatch.setattr(run, "_build_client", fail_client)
    monkeypatch.setattr(
        run,
        "make_env",
        lambda *args, **kwargs: pytest.fail("make_env must not be called"),
    )
    out = tmp_path / "client-error"

    rc = run.main(
        [
            "--seeds",
            "3",
            "4",
            "--model",
            "gpt-5.4",
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    payload = _read_json(out / "results.json")
    assert payload["seeds"] == [3, 4]
    assert payload["completed_episodes"] == 0
    assert payload["error_episodes"] == 2
    assert all(item["status"] == "error" for item in payload["results"])
    assert all(
        item["exception_message"] == "client construction failed"
        for item in payload["results"]
    )
    config = _read_json(out / "config.json")
    assert config["client"]["constructed"] is False
    assert config["client"]["construction_error"].startswith("RuntimeError:")
    assert config["client"]["model"] is None
    environment = config["environment_config"]
    card = run.load_skill("lane-change-overtake")
    grounding = run.select_grounding(card, "highway-env")
    expected = run.resolve_env_config(
        grounding,
        run.LaneChangeOvertakeScenario.configure(),
    )
    assert environment == expected


def test_all_seed_env_failures_use_deterministic_environment_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _FakeClient()
    monkeypatch.setattr(run, "_build_client", lambda model, mock: client)

    def fail_make_env(
        grounding: Any,
        scenario_config: dict[str, Any],
        seed: int,
    ) -> Any:
        del grounding, scenario_config
        raise RuntimeError(f"env setup failed for seed {seed}")

    monkeypatch.setattr(run, "make_env", fail_make_env)
    out = tmp_path / "all-env-errors"

    assert (
        run.main(
            [
                "--seeds",
                "0",
                "1",
                "--mock",
                "--out",
                str(out),
            ]
        )
        == 0
    )

    results = _read_json(out / "results.json")
    assert results["error_episodes"] == 2
    environment = _read_json(out / "config.json")["environment_config"]
    card = run.load_skill("lane-change-overtake")
    grounding = run.select_grounding(card, "highway-env")
    assert environment == run.resolve_env_config(
        grounding,
        run.LaneChangeOvertakeScenario.configure(),
    )
    assert client.close_calls == 1


def test_low_level_env_creation_failure_publishes_honest_custom_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _FakeClient()
    secret = "probe-secret-that-must-not-be-published"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setattr(run, "_build_client", lambda model, mock: client)

    def fail_env_creation(grounding: Any) -> Any:
        del grounding
        raise RuntimeError(f"probe creation failed with {secret}")

    monkeypatch.setattr(
        env_factory_module,
        "_make_unconfigured_env",
        fail_env_creation,
    )
    out = tmp_path / "low-level-env-errors"

    assert (
        run.main(
            [
                "--seeds",
                "0",
                "1",
                "--mock",
                "--out",
                str(out),
            ]
        )
        == 0
    )

    results = _read_json(out / "results.json")
    assert results["error_episodes"] == 2
    assert [item["seed"] for item in results["results"]] == [0, 1]
    assert all(item["status"] == "error" for item in results["results"])

    config_text = (out / "config.json").read_text(encoding="utf-8")
    config = json.loads(config_text)
    environment = config["environment_config"]
    assert environment["resolution_status"] == "error"
    assert environment["resolution_error"] == (
        "RuntimeError: probe creation failed with <redacted>"
    )
    card = run.load_skill("lane-change-overtake")
    grounding = run.select_grounding(card, "highway-env")
    assert environment["custom_config"] == (
        env_factory_module.build_custom_env_config(
            grounding,
            run.LaneChangeOvertakeScenario.configure(),
        )
    )
    assert secret not in config_text
    assert client.close_calls == 1


def test_setup_cleanup_note_populates_separate_episode_cleanup_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    secret = "sk-cleanup-note-secret-123456"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    client, _ = _patch_fast_success(monkeypatch)

    def fail_make_env(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        error = RuntimeError("setup primary failed")
        error.add_note(f"cleanup_error: RuntimeError: close failed with {secret}")
        raise error

    monkeypatch.setattr(run, "make_env", fail_make_env)
    out = tmp_path / "setup-cleanup-note"

    assert run.main(["--seeds", "0", "--mock", "--out", str(out)]) == 0

    result = _read_json(out / "results.json")["results"][0]
    assert result["status"] == "error"
    assert result["exception_message"] == "setup primary failed"
    assert result["cleanup_error"] == "RuntimeError: close failed with <redacted>"
    assert secret not in (out / "results.json").read_text(encoding="utf-8")
    assert client.close_calls == 1


def test_artifacts_redact_provider_episode_trace_and_snapshot_strings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api_key = "sk-live-raw-secret-123456"
    bearer = "Bearer eyJhbGciOiJIUzI1NiJ9.secret.signature"
    base_url = (
        "https://user%40name:pass%2Fword@example.test/v1/"
        "?api_key=query%2Fsecret&safe=visible"
    )
    decoded_values = ("user@name", "pass/word", "query/secret")
    provider_url = (
        "https://alice:password-value@example.test/path"
        "?token=provider-token&safe=visible#fragment-secret"
    )
    model_secret = "sk-model-secret-123456"
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    client = _FakeClient(model=model_secret)
    client.base_url = base_url
    client, _ = _patch_fast_success(monkeypatch, client=client)

    def secret_run_episode(
        env: _FakeEnv,
        executor: Any,
        scenario: Any,
        seed: int,
        max_steps: int,
    ) -> EpisodeResult:
        del executor, scenario, max_steps
        env.close()
        step = StepRecord(
            t=0,
            obs_summary=f"observation {decoded_values[0]} {provider_url}",
            primitive="maintain-speed",
            backend_action="IDLE",
            action_index=1,
            reward=0.0,
            crashed=False,
            request_hash="request-hash",
            raw_response=f"reply {api_key} {bearer}",
            cache_hit=False,
            latency_ms=1.0,
            decision_status="parse_fallback",
            fallback_reason=f"fallback {decoded_values[1]} {provider_url}",
            available_primitives=["maintain-speed"],
        )
        return EpisodeResult(
            seed=seed,
            status="ok",
            success=False,
            success_reason=f"reason {decoded_values[2]}",
            episode_return=0.0,
            crashed=False,
            steps=1,
            mean_speed=0.0,
            parse_failures=1,
            unavailable_action_attempts=0,
            llm_errors=0,
            terminated=False,
            truncated=False,
            max_steps_reached=True,
            scenario_completed=False,
            termination_reason="max_steps",
            target_initially_ahead=False,
            lane_change_completed_step=None,
            overtake_step=None,
            exception_message=f"setup {base_url}",
            cleanup_error=f"cleanup {bearer}",
            step_records=[step],
        )

    monkeypatch.setattr(run, "run_episode", secret_run_episode)
    out = tmp_path / "redacted-artifacts"

    assert (
        run.main(
            [
                "--seeds",
                "0",
                "--model",
                model_secret,
                "--out",
                str(out),
            ]
        )
        == 0
    )

    artifact_text = "".join(
        (out / filename).read_text(encoding="utf-8")
        for filename in ("results.json", "config.json", "episode_0.jsonl")
    )
    for secret in (
        api_key,
        bearer,
        base_url,
        *decoded_values,
        "alice",
        "password-value",
        "provider-token",
        "fragment-secret",
        model_secret,
    ):
        assert secret not in artifact_text
    assert "<redacted>" in artifact_text

    stored = _read_json(out / "results.json")["results"][0]
    assert stored["status"] == "ok"
    trace = json.loads((out / "episode_0.jsonl").read_text(encoding="utf-8"))
    assert trace["primitive"] == "maintain-speed"
    assert trace["decision_status"] == "parse_fallback"


def test_exception_text_includes_redacted_notes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-note-secret-123456"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    error = RuntimeError(f"primary {secret}")
    error.add_note(
        "cleanup_error: RuntimeError: "
        "https://user:password@example.test/?token=note-token "
        "Bearer bearer-note-token"
    )

    rendered = run._exception_text(error)

    assert rendered is not None
    assert "notes:" in rendered
    for value in (
        secret,
        "user",
        "password",
        "note-token",
        "bearer-note-token",
    ):
        assert value not in rendered
    assert "<redacted>" in rendered


def test_root_endpoint_identity_survives_recursive_snapshot_sanitizing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root_endpoint = "https://example.test/"
    monkeypatch.setenv("OPENAI_BASE_URL", root_endpoint)
    client = _FakeClient()
    client.base_url = root_endpoint
    _patch_fast_success(monkeypatch, client=client)
    out = tmp_path / "root-endpoint"

    assert (
        run.main(
            [
                "--seeds",
                "0",
                "--model",
                "gpt-5.4",
                "--out",
                str(out),
            ]
        )
        == 0
    )

    config = _read_json(out / "config.json")
    assert config["requested"]["base_url"] == root_endpoint
    assert config["client"]["base_url"] == root_endpoint


def test_short_sensitive_env_values_do_not_corrupt_snapshot_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TOKEN", "/")
    monkeypatch.setenv("API_KEY", "short")
    _patch_fast_success(monkeypatch)
    out = tmp_path / "short-secrets"

    assert run.main(["--seeds", "0", "--mock", "--out", str(out)]) == 0

    config = _read_json(out / "config.json")
    assert config["skill"]["name"] == "lane-change-overtake"
    assert config["client"]["type"] == "_FakeClient"
    assert config["client"]["model"] == "actual-deployment"
    assert config["source_config"]["path"].startswith("/")


@pytest.mark.parametrize("token", ["5", "0"])
def test_single_character_token_does_not_corrupt_model_or_versions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    token: str,
) -> None:
    monkeypatch.setenv("TOKEN", token)
    _patch_fast_success(monkeypatch)
    out = tmp_path / f"single-token-{token}"

    assert (
        run.main(
            [
                "--seeds",
                "0",
                "--model",
                "gpt-5.4",
                "--out",
                str(out),
            ]
        )
        == 0
    )

    config = _read_json(out / "config.json")
    assert config["requested"]["model"] == "gpt-5.4"
    assert config["installed_versions"]["gymnasium"] == "1.3.0"


def test_one_and_two_character_secrets_use_strict_identifier_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOKEN", "5")
    monkeypatch.setenv("PASSWORD", "xy")
    monkeypatch.setenv("PUNCT_SECRET", "/")

    redacted = run._redact_secrets(
        "standalone 5 xy; model gpt-5.4 version 1.3.0 identifier foo_xy_bar slash /"
    )

    assert redacted == (
        "standalone <redacted> <redacted>; model gpt-5.4 "
        "version 1.3.0 identifier foo_xy_bar slash /"
    )


def test_short_sensitive_values_use_token_boundaries_in_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("API_KEY", "short")
    monkeypatch.setenv("TOKEN", "abc1234")
    monkeypatch.setenv("PATH_SECRET", "/")
    _patch_fast_success(monkeypatch)

    def short_secret_episode(
        env: _FakeEnv,
        executor: Any,
        scenario: Any,
        seed: int,
        max_steps: int,
    ) -> EpisodeResult:
        del executor, scenario, max_steps
        env.close()
        step = StepRecord(
            t=0,
            obs_summary="step short shortfall /tmp/abc1234",
            primitive="maintain-speed",
            backend_action="IDLE",
            action_index=1,
            reward=0.0,
            crashed=False,
            request_hash="request-hash",
            raw_response="provider short shortfall",
            cache_hit=False,
            latency_ms=1.0,
            decision_status="parse_fallback",
            fallback_reason="error abc1234 xabc1234y",
            available_primitives=["maintain-speed"],
        )
        return EpisodeResult(
            seed=seed,
            status="ok",
            success=False,
            success_reason="",
            episode_return=0.0,
            crashed=False,
            steps=1,
            mean_speed=0.0,
            parse_failures=1,
            unavailable_action_attempts=0,
            llm_errors=0,
            terminated=False,
            truncated=False,
            max_steps_reached=True,
            scenario_completed=False,
            termination_reason="max_steps",
            target_initially_ahead=False,
            lane_change_completed_step=None,
            overtake_step=None,
            exception_message="setup short shortfall",
            cleanup_error="provider abc1234 xabc1234y",
            step_records=[step],
        )

    monkeypatch.setattr(run, "run_episode", short_secret_episode)
    out = tmp_path / "short-token-boundaries"

    assert run.main(["--seeds", "0", "--mock", "--out", str(out)]) == 0

    result = _read_json(out / "results.json")["results"][0]
    trace = json.loads((out / "episode_0.jsonl").read_text(encoding="utf-8"))
    config = _read_json(out / "config.json")
    assert result["exception_message"] == "setup <redacted> shortfall"
    assert result["cleanup_error"] == "provider <redacted> xabc1234y"
    assert trace["raw_response"] == "provider <redacted> shortfall"
    assert trace["fallback_reason"] == "error <redacted> xabc1234y"
    assert trace["obs_summary"] == "step <redacted> shortfall /tmp/<redacted>"
    assert config["source_config"]["path"].startswith("/")


def test_endpoint_path_variants_redact_standalone_but_preserve_larger_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_path = "/api/path-secret-%58%59%5A"
    raw_segment = "path-secret-%58%59%5A"
    decoded_path = "/api/path-secret-XYZ"
    segment = "path-secret-XYZ"
    raw_query_value = "query%2Fsecret"
    decoded_query_value = "query/secret"
    endpoint = f"https://example.test{raw_path}?token={raw_query_value}"
    monkeypatch.setenv("OPENAI_BASE_URL", endpoint)
    expected_hash = hashlib.sha256(
        raw_path.encode() + b"\0" + f"token={raw_query_value}".encode()
    ).hexdigest()[:16]

    redacted = run._redact_secrets(
        (
            f"error {raw_path} {raw_segment} {decoded_path} {segment} "
            f"{raw_query_value} {decoded_query_value} "
            f"{segment}suffix unrelated {endpoint}"
        ),
        include_endpoint_path_variants=True,
    )

    assert redacted == (
        "error <redacted> <redacted> <redacted> <redacted> "
        "<redacted> <redacted> "
        f"{segment}suffix unrelated "
        f"https://example.test/_path_sha256_{expected_hash}/"
    )


def test_endpoint_path_variants_only_apply_to_error_and_trace_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    endpoint = "https://example.test/openai/v1"
    monkeypatch.setenv("OPENAI_BASE_URL", endpoint)
    client = _FakeClient(model="openai")
    client.api_type = "openai"
    client.api_version = "v1"
    client.base_url = endpoint
    _patch_fast_success(monkeypatch, client=client)

    def path_error_episode(
        env: _FakeEnv,
        executor: Any,
        scenario: Any,
        seed: int,
        max_steps: int,
    ) -> EpisodeResult:
        del executor, scenario, max_steps
        env.close()
        step = StepRecord(
            t=0,
            obs_summary="obs /openai/v1 openai v1",
            primitive="maintain-speed",
            backend_action="IDLE",
            action_index=1,
            reward=0.0,
            crashed=False,
            request_hash="request-hash",
            raw_response="provider /openai/v1 openai v1",
            cache_hit=False,
            latency_ms=1.0,
            decision_status="parse_fallback",
            fallback_reason="failure openai v1",
            available_primitives=["maintain-speed"],
        )
        return EpisodeResult(
            seed=seed,
            status="ok",
            success=False,
            success_reason="",
            episode_return=0.0,
            crashed=False,
            steps=1,
            mean_speed=0.0,
            parse_failures=1,
            unavailable_action_attempts=0,
            llm_errors=0,
            terminated=False,
            truncated=False,
            max_steps_reached=True,
            scenario_completed=False,
            termination_reason="max_steps",
            target_initially_ahead=False,
            lane_change_completed_step=None,
            overtake_step=None,
            exception_message="setup /openai/v1 openai v1",
            step_records=[step],
        )

    monkeypatch.setattr(run, "run_episode", path_error_episode)
    out = tmp_path / "path-mode-split"

    assert (
        run.main(
            [
                "--seeds",
                "0",
                "--model",
                "gpt-5.4",
                "--out",
                str(out),
            ]
        )
        == 0
    )

    config = _read_json(out / "config.json")
    results = _read_json(out / "results.json")
    trace = json.loads((out / "episode_0.jsonl").read_text(encoding="utf-8"))
    assert config["client"]["model"] == "openai"
    assert config["client"]["api_type"] == "openai"
    assert config["client"]["api_version"] == "v1"
    assert config["prompt_version"] == "lco-v1"
    assert results["model"] == "openai"
    assert results["results"][0]["exception_message"] == (
        "setup <redacted> <redacted> <redacted>"
    )
    assert trace["raw_response"] == ("provider <redacted> <redacted> <redacted>")
    assert trace["fallback_reason"] == "failure <redacted> <redacted>"
    assert trace["obs_summary"] == "obs <redacted> <redacted> <redacted>"


def test_identity_endpoint_path_does_not_alter_controlled_snapshot_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = "https://identity.test/common/path"
    monkeypatch.setenv("IDENTITY_ENDPOINT", endpoint)
    safe_endpoint = run._sanitize_endpoint(endpoint)
    assert safe_endpoint is not None

    sanitized = run._sanitize_snapshot(
        {
            "client": {
                "base_url": safe_endpoint,
                "model": "common",
                "api_type": "path",
                "api_version": "common",
            },
            "prompt_version": "common",
            "skill": {"name": "path"},
            "environment": "common",
            "installed_versions": {"common": "1.2.3"},
            "construction_error": "failed /common/path common path",
        }
    )

    assert sanitized["client"] == {
        "base_url": safe_endpoint,
        "model": "common",
        "api_type": "path",
        "api_version": "common",
    }
    assert sanitized["prompt_version"] == "common"
    assert sanitized["skill"]["name"] == "path"
    assert sanitized["environment"] == "common"
    assert sanitized["installed_versions"] == {"common": "1.2.3"}
    assert sanitized["construction_error"] == (
        "failed <redacted> <redacted> <redacted>"
    )


def test_exception_url_keeps_origin_and_hash_without_plaintext_components() -> None:
    raw_url = (
        "https://user:password@example.test/private/path"
        "?token=query-secret&safe=visible#fragment-secret"
    )
    expected_hash = hashlib.sha256(
        b"/private/path\0token=query-secret&safe=visible"
    ).hexdigest()[:16]

    redacted = run._redact_secrets(f"provider failed at {raw_url}")

    assert f"https://example.test/_path_sha256_{expected_hash}/" in redacted
    for secret in (
        "user",
        "password",
        "private",
        "query-secret",
        "visible",
        "fragment-secret",
    ):
        assert secret not in redacted


def test_snapshot_preserves_already_sanitized_base_url_identity() -> None:
    endpoint = run._sanitize_endpoint(
        "https://user:password@example.test/v1?token=secret"
    )
    assert endpoint is not None

    sanitized = run._sanitize_snapshot(
        {
            "client": {"base_url": endpoint},
            "message": ("https://user:password@example.test/v1?token=secret"),
        }
    )

    assert sanitized["client"]["base_url"] == endpoint
    assert sanitized["message"] == endpoint
    assert "password" not in sanitized["message"]
    assert "secret" not in sanitized["message"]


def test_all_scenario_constructor_failures_still_publish_environment_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _FakeClient()
    monkeypatch.setattr(run, "_build_client", lambda model, mock: client)

    def fail_scenario_init(self: Any, card: Any, **parameters: Any) -> None:
        del self, card, parameters
        raise RuntimeError("scenario construction failed")

    monkeypatch.setattr(
        run.LaneChangeOvertakeScenario,
        "__init__",
        fail_scenario_init,
    )
    monkeypatch.setattr(
        run,
        "make_env",
        lambda *args, **kwargs: pytest.fail("make_env must not be called"),
    )
    out = tmp_path / "all-scenario-errors"

    assert (
        run.main(
            [
                "--seeds",
                "0",
                "1",
                "--mock",
                "--out",
                str(out),
            ]
        )
        == 0
    )

    results = _read_json(out / "results.json")
    assert results["error_episodes"] == 2
    environment = _read_json(out / "config.json")["environment_config"]
    assert environment["lanes_count"] == 4
    assert environment["observation"]["vehicles_count"] == 8
    assert environment["action"] == {"type": "DiscreteMetaAction"}
    assert client.close_calls == 1


def test_client_constructor_error_redacts_secret_environment_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    secret = "sk-secret-value-that-must-not-be-published"
    secret_endpoint = "https://user:password@example.test/v1?token=endpoint-secret"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setenv("OPENAI_BASE_URL", secret_endpoint)

    def fail_client(model: str, mock: bool) -> Any:
        del model, mock
        raise RuntimeError(f"provider rejected {secret} at {secret_endpoint}")

    monkeypatch.setattr(run, "_build_client", fail_client)
    out = tmp_path / "redacted-client-error"

    assert run.main(["--seeds", "0", "--out", str(out)]) == 0

    artifacts = (out / "results.json").read_text(encoding="utf-8") + (
        out / "config.json"
    ).read_text(encoding="utf-8")
    assert secret not in artifacts
    assert secret_endpoint not in artifacts
    assert "<redacted>" in artifacts


def test_actual_client_identity_and_resolved_env_config_are_recorded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_config = {
        "duration": 123,
        "policy_frequency": 2,
        "observation": {
            "type": "Kinematics",
            "normalize": False,
            "absolute": False,
        },
        "action": {"type": "DiscreteMetaAction"},
    }
    client, envs = _patch_fast_success(
        monkeypatch,
        client=_FakeClient(model="azure-deployment-actual"),
        env_config=env_config,
    )
    out = tmp_path / "identity"

    rc = run.main(
        [
            "--seeds",
            "9",
            "--model",
            "gpt-5.4-requested",
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    config = _read_json(out / "config.json")
    assert config["requested"]["model"] == "gpt-5.4-requested"
    path_hash = hashlib.sha256(b"/openai/\0api_key=hidden").hexdigest()[:16]
    assert config["client"] == {
        "api_type": "fake-responses",
        "api_version": "2099-01-01",
        "base_url": f"https://example.test/_path_sha256_{path_hash}/",
        "close_error": None,
        "constructed": True,
        "construction_error": None,
        "model": "azure-deployment-actual",
        "type": "_FakeClient",
    }
    assert config["environment_config"] == env_config
    assert _read_json(out / "results.json")["model"] == ("azure-deployment-actual")
    assert client.close_calls == 1
    assert [env.close_calls for env in envs] == [1]


def test_sanitize_endpoint_hashes_exact_raw_path_and_query_without_leaking() -> None:
    endpoint = (
        "https://user-secret:password-secret@example.test:8443/"
        "path-secret/nested?token=query-secret#fragment-secret"
    )

    sanitized = run._sanitize_endpoint(endpoint)
    same_identity = run._sanitize_endpoint(
        "https://other-user:other-password@example.test:8443/"
        "path-secret/nested?token=query-secret#other-fragment"
    )
    encoded_path = run._sanitize_endpoint("https://example.test:8443/a%2Fb")
    decoded_path = run._sanitize_endpoint("https://example.test:8443/a/b")
    different_query = run._sanitize_endpoint(
        "https://example.test:8443/path-secret/nested?token=other-secret"
    )

    expected_hash = hashlib.sha256(
        b"/path-secret/nested\0token=query-secret"
    ).hexdigest()[:16]
    assert sanitized == (f"https://example.test:8443/_path_sha256_{expected_hash}/")
    assert same_identity == sanitized
    assert encoded_path != decoded_path
    assert different_query != sanitized
    for secret in (
        "user-secret",
        "password-secret",
        "path-secret",
        "nested",
        "query-secret",
        "fragment-secret",
    ):
        assert secret not in sanitized
    assert "a%2Fb" not in encoded_path
    assert "a/b" not in decoded_path
    assert "other-secret" not in different_query

    assert (
        run._sanitize_endpoint(
            "https://user-secret:password-secret@example.test:8443/#fragment-secret"
        )
        == "https://example.test:8443/"
    )
    root_query = run._sanitize_endpoint("https://example.test:8443/?token=query-secret")
    assert root_query != "https://example.test:8443/"
    assert "query-secret" not in root_query


def test_client_close_failure_is_recorded_without_losing_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client, _ = _patch_fast_success(
        monkeypatch,
        client=_FakeClient(close_error=RuntimeError("close failed")),
    )
    out = tmp_path / "close-error"

    rc = run.main(
        [
            "--seeds",
            "0",
            "--model",
            "gpt-5.4",
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    assert client.close_calls == 1
    assert _read_json(out / "results.json")["completed_episodes"] == 1
    config = _read_json(out / "config.json")
    assert config["client"]["close_error"] == "RuntimeError: close failed"
    assert "client close failed: RuntimeError: close failed" in capsys.readouterr().err


def test_config_records_versions_git_hashes_and_excludes_secret_env_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_fast_success(monkeypatch)
    secrets = {
        "OPENAI_API_KEY": "top-secret-openai-key",
        "AZURE_TOKEN": "top-secret-azure-token",
        "OPENAI_BASE_URL": (
            "https://person:password@example.test/v1?token=top-secret-query"
        ),
    }
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)
    out = tmp_path / "metadata"

    assert run.main(["--seeds", "0", "--mock", "--out", str(out)]) == 0

    config_text = (out / "config.json").read_text(encoding="utf-8")
    config = json.loads(config_text)
    assert set(config["installed_versions"]) == {
        "highway-env",
        "gymnasium",
        "openai",
        "diskcache",
        "numpy",
        "PyYAML",
        "jsonschema",
        "pytest",
    }
    assert all(config["installed_versions"].values())
    assert len(config["git_commit"]) == 40
    int(config["git_commit"], 16)
    assert len(config["source_config"]["sha256"]) == 64
    assert len(config["skill"]["sha256"]) == 64
    assert "T" in config["timestamp"]
    assert len(config["date"]) == 10
    for secret in secrets.values():
        assert secret not in config_text


@pytest.mark.parametrize(
    ("config_data", "error"),
    [
        (["not", "a", "mapping"], "mapping"),
        (_valid_config(unexpected=True), "unknown config keys"),
        (_valid_config(max_steps="40"), "max_steps"),
        (
            _valid_config(
                scenario={
                    "success_margin": 5.0,
                    "lead_speed_ratio": 0.6,
                    "target_lane_front_clearance_m": 20.0,
                    "target_lane_rear_clearance_m": 20.0,
                    "spawn_gap": 30.0,
                    "front_clearance": 99.0,
                }
            ),
            "unknown scenario keys",
        ),
        (
            _valid_config(
                scenario={
                    "success_margin": True,
                    "lead_speed_ratio": 0.6,
                    "target_lane_front_clearance_m": 20.0,
                    "target_lane_rear_clearance_m": 20.0,
                    "spawn_gap": 30.0,
                }
            ),
            "success_margin",
        ),
    ],
)
def test_invalid_yaml_config_is_rejected(
    tmp_path: Path,
    config_data: object,
    error: str,
) -> None:
    config_path = tmp_path / "invalid.yaml"
    _write_yaml(config_path, config_data)

    with pytest.raises((TypeError, ValueError), match=error):
        run.main(
            [
                "--config",
                str(config_path),
                "--seeds",
                "0",
                "--mock",
                "--out",
                str(tmp_path / "out"),
            ]
        )


def test_malformed_yaml_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "malformed.yaml"
    config_path.write_text("skill: [unterminated\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid YAML"):
        run.main(
            [
                "--config",
                str(config_path),
                "--seeds",
                "0",
                "--mock",
                "--out",
                str(tmp_path / "out"),
            ]
        )


def test_nonempty_output_directory_is_rejected_before_client_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    out = tmp_path / "existing"
    out.mkdir()
    (out / "keep.txt").write_text("do not overwrite", encoding="utf-8")
    monkeypatch.setattr(
        run,
        "_build_client",
        lambda *args: pytest.fail("client must not be built"),
    )

    with pytest.raises(ValueError, match="non-empty"):
        run.main(["--seeds", "0", "--mock", "--out", str(out)])

    assert (out / "keep.txt").read_text(encoding="utf-8") == "do not overwrite"


def test_module_help_smoke() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "world2skills.evaluation.run",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
        env=os.environ.copy(),
    )

    assert completed.returncode == 0, completed.stderr
    assert "--config" in completed.stdout
    assert "--seeds" in completed.stdout
    assert "--spawn-gap" in completed.stdout
