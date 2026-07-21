from __future__ import annotations

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
from world2skills.runtime.types import EpisodeResult


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
    assert config["client"] == {
        "api_type": "fake-responses",
        "api_version": "2099-01-01",
        "base_url": "https://example.test/openai/",
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
