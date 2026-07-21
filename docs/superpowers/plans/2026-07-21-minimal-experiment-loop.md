# Minimal Experiment Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the M1 minimal loop `skill.yaml → LLM executor → highway-env → metrics/eval` for `lane-change-overtake` @ `highway-v0`, driven by an LLM picking abstract primitives, on a deterministic overtake scenario with an executable, causal success criterion.

**Architecture:** A `world2skills.runtime` package (types, LLM clients, skill loader, deterministic scenario, observation renderer, executor, env factory) plus a `world2skills.evaluation` package (episode runner, metrics, CLI). The LLM reads the skill card + a rendered observation and returns one abstract primitive per step; Python maps it to a `DiscreteMetaAction` index (via the env's inverted action map) and steps the env. A `LaneChangeOvertakeScenario` guarantees a slow lead vehicle + a cleared adjacent lane, binds the lead object, and judges success by causal order (lane change *before* overtaking the bound target, no collision).

**Tech Stack:** Python 3.11 (conda `llama_factory`), `highway-env==1.12.0`, `gymnasium==1.3.0`, `numpy`, `pyyaml`, `openai`, `diskcache`, `pytest`. LLM default GPT-5.4 via the Azure Responses proxy (`scripts/connect_gpt54.sh`); `MockLLMClient` for tests.

**Spec:** `docs/superpowers/specs/2026-07-20-minimal-experiment-loop-design.md`

**Verified API facts (highway-env 1.12.0, confirmed by introspection during planning):**
- `gymnasium.make("highway-v0", render_mode=None)` then `env.unwrapped.configure(cfg)`; `action_type`/`observation_type` are materialized on the first `reset()`.
- `env.unwrapped.action_type.actions == {0:'LANE_LEFT',1:'IDLE',2:'LANE_RIGHT',3:'FASTER',4:'SLOWER'}`; invert for name→index. `get_available_actions()` returns a list of indices.
- Ego is `env.unwrapped.vehicle` (`MDPVehicle`); `.lane_index` is a tuple `('0','1',i)`; `.position` (np array), `.speed`, `.target_speed`, `.crashed`.
- Lane via `env.unwrapped.road.network.get_lane(lane_index)`; `lane.local_coordinates(pos) -> (longitudinal, lateral)`.
- `env.unwrapped.road.network.all_side_lanes(lane_index)` / `side_lanes(lane_index)`; `env.unwrapped.road.neighbour_vehicles(vehicle, lane_index) -> (front, rear)`.
- Spawn: `highway_env.vehicle.behavior.IDMVehicle.make_on_lane(road, lane_index, longitudinal, speed)`; append/remove via `env.unwrapped.road.vehicles`.
- Re-observe after scene edits: `env.unwrapped.observation_type.observe()`.
- `info` keys after `step`: `action, crashed, rewards, speed` — **no `arrived`/`success`** (success must come from the scenario evaluator).
- Kinematics with `features=["presence","x","y","vx","vy","cos_h","sin_h"]`, `vehicles_count=8`, `normalize=False`, `absolute=False`, `see_behind=True` → obs ndarray `(8,7)`; row 0 = ego (absolute), other rows ego-relative.

**Note:** `highway-env==1.12.0` and `gymnasium==1.3.0` were already installed into the `llama_factory` env during planning. Task 0 still records them in `requirements-runtime.txt` and verifies.

---

## File Structure

```
world2skills/
  __init__.py                     # package marker (empty)
  runtime/
    __init__.py
    types.py            # dataclasses: Grounding, SkillCard, ObservationContext,
                        #   DecisionResult, StepRecord, EpisodeResult, BatchResult
    llm.py              # Message, ModelSettings, ChatResult, LLMClient,
                        #   make_request_hash, MockLLMClient, OpenAIClient, AzureResponsesClient
    skill_loader.py     # load_skill, select_grounding
    obs_render.py       # render(observation, context, feature_names) -> str
    scenario.py         # judge_success, ScenarioSetupError, LaneChangeOvertakeScenario
    executor.py         # LLMSkillExecutor
    env_factory.py      # make_env
  evaluation/
    __init__.py
    episode.py          # run_episode
    metrics.py          # aggregate, episode_to_result_dict, write_outputs
    run.py              # CLI main()
  experiments/
    configs/lane-change-overtake.yaml
  outputs/              # run artifacts (gitignored)
  tests/
    conftest.py
    test_types.py
    test_skill_loader.py
    test_llm.py
    test_obs_render.py
    test_executor.py
    test_scenario_evaluator.py
    test_env_factory.py
    test_episode_smoke.py
    test_results_schema.py
    test_cli_smoke.py
  requirements-runtime.txt
```

Run all tests from the repo root: `python -m pytest world2skills/tests -q`.

---

## Task 0: Package scaffold, dependencies, gitignore, conftest

**Files:**
- Create: `world2skills/__init__.py`, `world2skills/runtime/__init__.py`, `world2skills/evaluation/__init__.py`, `world2skills/tests/__init__.py`
- Create: `world2skills/requirements-runtime.txt`
- Create: `world2skills/tests/conftest.py`
- Modify: `.gitignore` (append one line)

- [ ] **Step 1: Create package markers**

```bash
cd /data/yifan/zrx/World2Skills
touch world2skills/__init__.py
mkdir -p world2skills/runtime world2skills/evaluation world2skills/tests world2skills/experiments/configs world2skills/outputs
touch world2skills/runtime/__init__.py world2skills/evaluation/__init__.py world2skills/tests/__init__.py
```

- [ ] **Step 2: Write `world2skills/requirements-runtime.txt`**

```text
# Runtime deps for the World2Skills phase-2 experiment loop.
# Pins verified 2026-07-21 (PyPI highway-env 1.12.0 requires python>=3.10, gymnasium>=1.0).
highway-env==1.12.0
gymnasium==1.3.0
# Already present in the llama_factory env (listed for completeness):
#   openai>=2, diskcache>=5, numpy>=1.26, pyyaml>=6, jsonschema>=4, pytest>=8
```

- [ ] **Step 3: Write `world2skills/tests/conftest.py`**

Puts the repo root on `sys.path` so `import world2skills...` works when running pytest from anywhere.

```python
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
```

- [ ] **Step 4: Append the outputs dir to `.gitignore` (do NOT overwrite existing content)**

```bash
cd /data/yifan/zrx/World2Skills
grep -qxF 'world2skills/outputs/' .gitignore || printf '\n# World2Skills experiment run artifacts\nworld2skills/outputs/\n' >> .gitignore
```

- [ ] **Step 5: Verify deps installed and recorded**

Run: `python -c "import highway_env, gymnasium; print(highway_env.__version__, gymnasium.__version__)"`
Expected: `1.12.0 1.3.0`

(If missing on a fresh machine: `python -m pip install -r world2skills/requirements-runtime.txt`.)

- [ ] **Step 6: Verify pytest collects an empty suite**

Run: `python -m pytest world2skills/tests -q`
Expected: `no tests ran` (exit code 5) — confirms conftest imports cleanly.

- [ ] **Step 7: Commit**

```bash
git add world2skills/__init__.py world2skills/runtime/__init__.py world2skills/evaluation/__init__.py world2skills/tests/__init__.py world2skills/tests/conftest.py world2skills/requirements-runtime.txt .gitignore
git commit -m "chore(w2s): scaffold phase-2 runtime/evaluation packages and deps"
```

---

## Task 1: `runtime/types.py` — shared dataclasses

**Files:**
- Create: `world2skills/runtime/types.py`
- Test: `world2skills/tests/test_types.py`

- [ ] **Step 1: Write the failing test**

```python
# world2skills/tests/test_types.py
import json
from dataclasses import asdict

from world2skills.runtime.types import DecisionResult, StepRecord, EpisodeResult


def test_steprecord_from_decision_and_json():
    dr = DecisionResult(
        primitive="accelerate", backend_action="FASTER", action_index=3,
        request_hash="abc", raw_response='{"primitive": "accelerate"}',
        cache_hit=False, latency_ms=12.0, decision_status="ok",
        fallback_reason=None, available_primitives=["accelerate", "maintain-speed"],
    )
    rec = StepRecord.from_decision(dr, t=0, obs_summary="ego...", reward=1.0, crashed=False)
    assert rec.t == 0 and rec.primitive == "accelerate" and rec.action_index == 3
    assert rec.reward == 1.0 and rec.crashed is False
    # every field must be JSON-serializable
    json.dumps(asdict(rec))


def test_episode_result_defaults_json():
    er = EpisodeResult(
        seed=0, status="ok", success=True, success_reason="ok",
        episode_return=1.0, crashed=False, steps=3, mean_speed=20.0,
        parse_failures=0, unavailable_action_attempts=0, llm_errors=0,
        terminated=False, truncated=False, max_steps_reached=False,
        scenario_completed=True, termination_reason="success",
        target_initially_ahead=True, lane_change_completed_step=1, overtake_step=2,
    )
    assert er.exception_type is None and er.step_records == []
    json.dumps(asdict(er))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest world2skills/tests/test_types.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'world2skills.runtime.types'`

- [ ] **Step 3: Write `world2skills/runtime/types.py`**

```python
"""Shared dataclasses for the World2Skills experiment loop.

Pure data holders (no behavior beyond one convenience constructor). All are
`dataclasses.asdict`-serializable so results/traces can be written as JSON.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Grounding:
    backend: str
    environment: str
    observation: dict[str, Any]
    action: dict[str, Any]
    primitive_map: dict[str, str]


@dataclass
class SkillCard:
    name: str
    description: str
    skill_md_body: str
    parameters: dict[str, Any]
    interface: dict[str, Any]
    execution: dict[str, Any]
    preconditions: list[str]
    success_criteria: list[str]
    safety_constraints: list[str]
    groundings: list[Grounding]
    primitives: list[str]  # union of interface.actions[].primitives


@dataclass
class ObservationContext:
    available_primitives: list[str]
    ego_lane: int
    prev_primitive: str | None
    target_ahead: bool
    target_gap_m: float
    target_rel_speed_mps: float
    left_lane_exists: bool
    right_lane_exists: bool
    left_front_gap_m: float | None
    left_rear_gap_m: float | None
    left_rear_closing_speed_mps: float | None
    right_front_gap_m: float | None
    right_rear_gap_m: float | None
    right_rear_closing_speed_mps: float | None


@dataclass
class DecisionResult:
    """Executor output, available BEFORE env.step()."""
    primitive: str
    backend_action: str
    action_index: int
    request_hash: str
    raw_response: str
    cache_hit: bool
    latency_ms: float
    decision_status: str  # ok | parse_fallback | unavailable_fallback | llm_error_fallback
    fallback_reason: str | None
    available_primitives: list[str]


@dataclass
class StepRecord:
    """DecisionResult + step outcome, assembled by episode AFTER env.step()."""
    t: int
    obs_summary: str
    primitive: str
    backend_action: str
    action_index: int
    reward: float
    crashed: bool
    request_hash: str
    raw_response: str
    cache_hit: bool
    latency_ms: float
    decision_status: str
    fallback_reason: str | None
    available_primitives: list[str]

    @classmethod
    def from_decision(
        cls, dr: DecisionResult, *, t: int, obs_summary: str, reward: float, crashed: bool
    ) -> "StepRecord":
        return cls(
            t=t, obs_summary=obs_summary, primitive=dr.primitive,
            backend_action=dr.backend_action, action_index=dr.action_index,
            reward=reward, crashed=crashed, request_hash=dr.request_hash,
            raw_response=dr.raw_response, cache_hit=dr.cache_hit, latency_ms=dr.latency_ms,
            decision_status=dr.decision_status, fallback_reason=dr.fallback_reason,
            available_primitives=dr.available_primitives,
        )


@dataclass
class EpisodeResult:
    seed: int
    status: str  # ok | error
    success: bool
    success_reason: str
    episode_return: float
    crashed: bool
    steps: int
    mean_speed: float
    parse_failures: int
    unavailable_action_attempts: int
    llm_errors: int
    terminated: bool
    truncated: bool
    max_steps_reached: bool
    scenario_completed: bool
    termination_reason: str
    target_initially_ahead: bool
    lane_change_completed_step: int | None
    overtake_step: int | None
    exception_type: str | None = None
    step_records: list[StepRecord] = field(default_factory=list)


@dataclass
class BatchResult:
    schema_version: str
    model: str
    prompt_version: str
    skill: str
    environment: str
    seeds: list[int]
    requested_episodes: int
    completed_episodes: int
    error_episodes: int
    successful_episodes: int
    success_rate: float
    collision_rate: float
    mean_return: float
    results: list[dict[str, Any]]  # per-episode dicts WITHOUT step_records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest world2skills/tests/test_types.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add world2skills/runtime/types.py world2skills/tests/test_types.py
git commit -m "feat(w2s): add runtime dataclasses (types.py)"
```

---

## Task 2: `runtime/skill_loader.py` — load SKILL.md + skill.yaml

**Files:**
- Create: `world2skills/runtime/skill_loader.py`
- Test: `world2skills/tests/test_skill_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# world2skills/tests/test_skill_loader.py
from world2skills.runtime.skill_loader import load_skill, select_grounding


def test_load_lane_change_overtake():
    card = load_skill("lane-change-overtake")
    assert card.name == "lane-change-overtake"
    # primitives = union of interface.actions[].primitives
    assert set(card.primitives) == {
        "change-lane-left", "change-lane-right",
        "maintain-speed", "accelerate", "decelerate",
    }
    # SKILL.md body must contain the four mandatory sections
    for section in ("## When to use", "## Procedure", "## Reasoning cues", "## Failure modes"):
        assert section in card.skill_md_body
    assert card.parameters["target_speed"]["default"] == 25


def test_select_highway_grounding():
    card = load_skill("lane-change-overtake")
    g = select_grounding(card, backend="highway-env")
    assert g.environment == "highway-v0"
    assert g.primitive_map["accelerate"] == "FASTER"
    assert g.primitive_map["change-lane-left"] == "LANE_LEFT"
    assert "presence" in g.observation["features"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest world2skills/tests/test_skill_loader.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'world2skills.runtime.skill_loader'`

- [ ] **Step 3: Write `world2skills/runtime/skill_loader.py`**

```python
"""Load a skill directory (SKILL.md discovery layer + skill.yaml representation)."""
from __future__ import annotations

from pathlib import Path

import yaml

from .types import Grounding, SkillCard

SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Return (frontmatter_dict, body). Frontmatter is the first '---' fenced block."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            front = yaml.safe_load(parts[1]) or {}
            return front, parts[2].lstrip("\n")
    return {}, text


def load_skill(skill_id: str, skills_dir: Path = SKILLS_DIR) -> SkillCard:
    skill_dir = skills_dir / skill_id
    yaml_path = skill_dir / "skill.yaml"
    md_path = skill_dir / "SKILL.md"
    if not yaml_path.exists():
        raise FileNotFoundError(f"skill.yaml not found for '{skill_id}' at {yaml_path}")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    _, body = _split_frontmatter(md_path.read_text(encoding="utf-8")) if md_path.exists() else ({}, "")

    interface = data.get("interface", {})
    primitives: list[str] = []
    for action in interface.get("actions", []):
        for prim in action.get("primitives", []):
            if prim not in primitives:
                primitives.append(prim)

    groundings = [
        Grounding(
            backend=g["backend"],
            environment=g["environment"],
            observation=g["observation"],
            action=g["action"],
            primitive_map=g["primitive_map"],
        )
        for g in data.get("groundings", [])
    ]

    return SkillCard(
        name=data["name"],
        description=data.get("description", ""),
        skill_md_body=body,
        parameters=data.get("parameters", {}),
        interface=interface,
        execution=data.get("execution", {}),
        preconditions=data.get("preconditions", []),
        success_criteria=data.get("success_criteria", []),
        safety_constraints=data.get("safety_constraints", []),
        groundings=groundings,
        primitives=primitives,
    )


def select_grounding(card: SkillCard, backend: str = "highway-env") -> Grounding:
    for g in card.groundings:
        if g.backend == backend:
            return g
    raise ValueError(f"skill '{card.name}' has no grounding for backend '{backend}'")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest world2skills/tests/test_skill_loader.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add world2skills/runtime/skill_loader.py world2skills/tests/test_skill_loader.py
git commit -m "feat(w2s): add skill_loader (SKILL.md + skill.yaml -> SkillCard)"
```

---

## Task 3: `runtime/llm.py` — LLM clients with corrected cache key

**Files:**
- Create: `world2skills/runtime/llm.py`
- Test: `world2skills/tests/test_llm.py`

- [ ] **Step 1: Write the failing test**

```python
# world2skills/tests/test_llm.py
from world2skills.runtime.llm import Message, ModelSettings, MockLLMClient, make_request_hash


def _msgs():
    return [Message("system", "s"), Message("user", "u")]


def test_mock_client_returns_scripted_replies():
    client = MockLLMClient(replies=['{"primitive": "accelerate"}', "second"])
    r1 = client.chat(_msgs(), ModelSettings(), prompt_version="lco-v1")
    r2 = client.chat(_msgs(), ModelSettings(), prompt_version="lco-v1")
    assert r1.reply == '{"primitive": "accelerate"}'
    assert r2.reply == "second"
    assert r1.cache_hit is False and r1.request_hash


def test_request_hash_includes_config_endpoint_and_prompt_version():
    base = dict(
        client_type="OpenAIClient", model="m", endpoint="http://x/",
        messages=[{"role": "user", "content": "u"}],
        settings=ModelSettings(temperature=0.0), prompt_version="v1",
    )
    h = make_request_hash(**base)
    assert h == make_request_hash(**base)  # deterministic
    # any of these changes the hash
    assert h != make_request_hash(**{**base, "model": "m2"})
    assert h != make_request_hash(**{**base, "endpoint": "http://y/"})
    assert h != make_request_hash(**{**base, "prompt_version": "v2"})
    assert h != make_request_hash(**{**base, "settings": ModelSettings(temperature=0.7)})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest world2skills/tests/test_llm.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'world2skills.runtime.llm'`

- [ ] **Step 3: Write `world2skills/runtime/llm.py`**

```python
"""Minimal LLM clients for the experiment loop (reimplemented in-project).

Corrects the Trace2Skill cache key (which used only model+messages): our
request hash also includes client_type, endpoint identity, generation config,
and prompt_version, so distinct requests never collide. The cache guarantees
*replay* consistency, NOT model determinism (Azure Responses drops `seed`).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ModelSettings:
    temperature: float = 0.0
    max_tokens: int | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)

    def key(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "max_tokens": self.max_tokens,
                "extra_body": self.extra_body}


@dataclass
class ChatResult:
    reply: str
    request_hash: str
    cache_hit: bool
    latency_ms: float


def make_request_hash(
    *, client_type: str, model: str, endpoint: str,
    messages: list[dict[str, str]], settings: ModelSettings, prompt_version: str,
) -> str:
    payload = {
        "client_type": client_type, "model": model, "endpoint": endpoint,
        "messages": messages, "settings": settings.key(), "prompt_version": prompt_version,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class LLMClient(ABC):
    @abstractmethod
    def chat(self, messages: list[Message], settings: ModelSettings | None,
             prompt_version: str) -> ChatResult: ...


class MockLLMClient(LLMClient):
    """Returns scripted replies (list cycled, or a callable(messages)->str)."""

    def __init__(self, replies: list[str] | Callable[[list[Message]], str]):
        self._replies = replies
        self._i = 0

    def chat(self, messages, settings=None, prompt_version="") -> ChatResult:
        settings = settings or ModelSettings()
        if callable(self._replies):
            reply = self._replies(messages)
        else:
            reply = self._replies[self._i % len(self._replies)]
            self._i += 1
        h = make_request_hash(
            client_type="MockLLMClient", model="mock", endpoint="mock",
            messages=[{"role": m.role, "content": m.content} for m in messages],
            settings=settings, prompt_version=prompt_version,
        )
        return ChatResult(reply=reply, request_hash=h, cache_hit=False, latency_ms=0.0)


class _CachedOpenAILike(LLMClient):
    """Shared cache + timing wrapper for the two OpenAI-compatible clients."""

    client_type = "OpenAILike"

    def __init__(self, model: str, endpoint: str, use_cache: bool, cache_path: str | None):
        self.model = model
        self.endpoint = endpoint
        self._cache = None
        if use_cache:
            try:
                import diskcache as dc
                path = cache_path or os.path.join(
                    os.path.expanduser("~"), ".cache", "w2s_llm.diskcache")
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                self._cache = dc.Cache(path)
            except ImportError:
                self._cache = None

    def _raw_call(self, messages: list[Message], settings: ModelSettings) -> str:
        raise NotImplementedError

    def chat(self, messages, settings=None, prompt_version="") -> ChatResult:
        settings = settings or ModelSettings()
        h = make_request_hash(
            client_type=self.client_type, model=self.model, endpoint=self.endpoint,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            settings=settings, prompt_version=prompt_version,
        )
        if self._cache is not None and h in self._cache:
            return ChatResult(reply=self._cache[h], request_hash=h, cache_hit=True, latency_ms=0.0)
        t0 = time.perf_counter()
        reply = self._raw_call(messages, settings)
        latency = (time.perf_counter() - t0) * 1000.0
        if self._cache is not None and reply:
            self._cache[h] = reply
        return ChatResult(reply=reply, request_hash=h, cache_hit=False, latency_ms=latency)


class OpenAIClient(_CachedOpenAILike):
    """chat.completions client (OpenAI / vLLM / LiteLLM)."""

    client_type = "OpenAIClient"

    def __init__(self, model="gpt-4o-mini", api_key=None, base_url=None,
                 use_cache=True, cache_path=None, timeout=600.0, retry_times=(5, 10, 30)):
        base_url = base_url or os.getenv("OPENAI_BASE_URL") or ""
        super().__init__(model=model, endpoint=base_url, use_cache=use_cache, cache_path=cache_path)
        from openai import OpenAI
        self.retry_times = retry_times
        kwargs = {"api_key": api_key or os.getenv("OPENAI_API_KEY") or "EMPTY", "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def _raw_call(self, messages, settings):
        cfg: dict[str, Any] = {"temperature": settings.temperature}
        if settings.max_tokens:
            cfg["max_tokens"] = settings.max_tokens
        if settings.extra_body:
            cfg["extra_body"] = settings.extra_body
        last = None
        for wait in self.retry_times:
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": m.role, "content": m.content} for m in messages],
                    **cfg,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - retry transient provider errors
                last = exc
                time.sleep(wait)
        raise last


class AzureResponsesClient(_CachedOpenAILike):
    """OpenAI Responses API client for the local Azure AAD proxy (GPT-5.4)."""

    client_type = "AzureResponsesClient"

    def __init__(self, model, api_key=None, base_url=None,
                 use_cache=True, cache_path=None, timeout=600.0, retry_times=(5, 10, 30)):
        base_url = base_url or os.getenv("OPENAI_BASE_URL") or "http://127.0.0.1:8765/openai/"
        super().__init__(model=model, endpoint=base_url, use_cache=use_cache, cache_path=cache_path)
        from openai import OpenAI
        self.retry_times = retry_times
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "EMPTY",
            base_url=base_url,
            default_query={"api-version": os.getenv("OPENAI_API_VERSION", "2025-04-01-preview")},
            timeout=timeout,
        )

    def _raw_call(self, messages, settings):
        cfg: dict[str, Any] = {}
        if settings.max_tokens:
            cfg["max_output_tokens"] = settings.max_tokens
        last = None
        for wait in self.retry_times:
            try:
                resp = self._client.responses.create(
                    model=self.model,
                    input=[{"role": m.role, "content": m.content} for m in messages],
                    **cfg,
                )
                return resp.output_text or ""
            except Exception as exc:  # noqa: BLE001 - retry transient provider errors
                last = exc
                time.sleep(wait)
        raise last
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest world2skills/tests/test_llm.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add world2skills/runtime/llm.py world2skills/tests/test_llm.py
git commit -m "feat(w2s): add in-project LLM clients with corrected cache key"
```

---

## Task 4: `runtime/obs_render.py` — render observation for the LLM

**Files:**
- Create: `world2skills/runtime/obs_render.py`
- Test: `world2skills/tests/test_obs_render.py`

- [ ] **Step 1: Write the failing test**

```python
# world2skills/tests/test_obs_render.py
import numpy as np

from world2skills.runtime.obs_render import render
from world2skills.runtime.types import ObservationContext


def _ctx():
    return ObservationContext(
        available_primitives=["maintain-speed", "accelerate", "change-lane-left"],
        ego_lane=3, prev_primitive="accelerate",
        target_ahead=True, target_gap_m=18.0, target_rel_speed_mps=-10.0,
        left_lane_exists=True, right_lane_exists=False,
        left_front_gap_m=30.0, left_rear_gap_m=25.0, left_rear_closing_speed_mps=-2.0,
        right_front_gap_m=None, right_rear_gap_m=None, right_rear_closing_speed_mps=None,
    )


def test_render_contains_ego_neighbors_and_context():
    feats = ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"]
    obs = np.array([
        [1, 200.0, 12.0, 25.0, 0.0, 1.0, 0.0],   # ego (absolute)
        [1, 18.0, 0.0, -10.0, 0.0, 1.0, 0.0],     # neighbour (ego-relative)
        [0, 0, 0, 0, 0, 0, 0],                      # empty slot -> skipped
    ], dtype=float)
    text = render(obs, _ctx(), feats)
    assert "Ego" in text and "(0.0, 0.0)" in text          # ego pinned to origin
    assert "dx=18.0" in text                                 # neighbour relative x
    assert "change-lane-left" in text                        # available primitives
    assert "target" in text.lower() and "18.0" in text       # target gap
    assert "left" in text.lower() and "30.0" in text         # left-lane front gap
    assert "right" in text.lower() and "none" in text.lower()  # right lane absent
    assert "presence" not in text  # empty (presence=0) row not rendered as a vehicle
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest world2skills/tests/test_obs_render.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'world2skills.runtime.obs_render'`

- [ ] **Step 3: Write `world2skills/runtime/obs_render.py`**

```python
"""Render a Kinematics observation + scenario context into LLM-readable text.

Coordinate semantics are made explicit and uniform: the ego is presented at the
origin (0,0) and every neighbour is expressed relative to the ego (meters). The
raw array is NOT echoed row-by-row (row 0 is absolute ego, other rows already
ego-relative when absolute=False) — we normalise the presentation here.
"""
from __future__ import annotations

import numpy as np

from .types import ObservationContext


def _fmt(v: float | None) -> str:
    return "none" if v is None else f"{v:.1f}"


def render(observation: np.ndarray, context: ObservationContext, feature_names: list[str]) -> str:
    idx = {name: i for i, name in enumerate(feature_names)}
    ix, iy = idx.get("x"), idx.get("y")
    ivx, ivy = idx.get("vx"), idx.get("vy")
    ip = idx.get("presence")

    ego = observation[0]
    ego_speed = float(ego[ivx]) if ivx is not None else 0.0

    lines = [
        f"Ego (reference frame origin): position=(0.0, 0.0), lane={context.ego_lane}, "
        f"speed={ego_speed:.1f} m/s",
        "Nearby vehicles (position & velocity RELATIVE to ego, meters):",
    ]
    n = 0
    for row in observation[1:]:
        if ip is not None and row[ip] <= 0:
            continue  # empty slot
        n += 1
        dx = float(row[ix]) if ix is not None else 0.0
        dy = float(row[iy]) if iy is not None else 0.0
        dvx = float(row[ivx]) if ivx is not None else 0.0
        dvy = float(row[ivy]) if ivy is not None else 0.0
        lines.append(f"  #{n} dx={dx:.1f} dy={dy:.1f} dvx={dvx:.1f} dvy={dvy:.1f}")
    if n == 0:
        lines.append("  (none)")

    rel = "ahead" if context.target_ahead else "behind/alongside"
    lines += [
        "",
        f"Target (the slower lead to overtake): {rel}, "
        f"gap={context.target_gap_m:.1f} m, rel_speed={context.target_rel_speed_mps:.1f} m/s "
        f"(negative = target slower than ego)",
        f"Left lane: exists={context.left_lane_exists}, "
        f"front_gap={_fmt(context.left_front_gap_m)} m, rear_gap={_fmt(context.left_rear_gap_m)} m, "
        f"rear_closing_speed={_fmt(context.left_rear_closing_speed_mps)} m/s",
        f"Right lane: exists={context.right_lane_exists}, "
        f"front_gap={_fmt(context.right_front_gap_m)} m, rear_gap={_fmt(context.right_rear_gap_m)} m, "
        f"rear_closing_speed={_fmt(context.right_rear_closing_speed_mps)} m/s",
        f"Previous primitive: {context.prev_primitive}",
        f"Allowed primitives THIS step: {', '.join(context.available_primitives)}",
    ]
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest world2skills/tests/test_obs_render.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add world2skills/runtime/obs_render.py world2skills/tests/test_obs_render.py
git commit -m "feat(w2s): add observation renderer (unified ego-relative frame)"
```

---

## Task 5: `runtime/executor.py` — LLM executor (structured parse, injected map)

**Files:**
- Create: `world2skills/runtime/executor.py`
- Test: `world2skills/tests/test_executor.py`

- [ ] **Step 1: Write the failing test**

```python
# world2skills/tests/test_executor.py
from world2skills.runtime.executor import LLMSkillExecutor
from world2skills.runtime.llm import MockLLMClient
from world2skills.runtime.skill_loader import load_skill, select_grounding

NAME_TO_INDEX = {"LANE_LEFT": 0, "IDLE": 1, "LANE_RIGHT": 2, "FASTER": 3, "SLOWER": 4}
ALL = ["maintain-speed", "accelerate", "decelerate", "change-lane-left", "change-lane-right"]


def _executor(reply):
    card = load_skill("lane-change-overtake")
    g = select_grounding(card, "highway-env")
    return LLMSkillExecutor(card, g, MockLLMClient([reply]), "lco-v1", NAME_TO_INDEX)


def test_valid_json_primitive_maps_to_index():
    dr = _executor('{"primitive": "accelerate"}').decide("obs", None, ALL)
    assert dr.primitive == "accelerate" and dr.backend_action == "FASTER"
    assert dr.action_index == 3 and dr.decision_status == "ok"


def test_fenced_json_is_parsed():
    dr = _executor('```json\n{"primitive": "decelerate"}\n```').decide("obs", None, ALL)
    assert dr.action_index == 4 and dr.decision_status == "ok"


def test_invalid_json_falls_back_and_flags_parse():
    dr = _executor("do not accelerate, maintain-speed please").decide("obs", None, ALL)
    assert dr.decision_status == "parse_fallback"
    assert dr.primitive == "maintain-speed" and dr.action_index == 1


def test_unknown_primitive_falls_back_and_flags_parse():
    dr = _executor('{"primitive": "teleport"}').decide("obs", None, ALL)
    assert dr.decision_status == "parse_fallback" and dr.action_index == 1


def test_unavailable_primitive_falls_back_and_flags_unavailable():
    # LLM picks a lane change but only longitudinal primitives are available this step
    dr = _executor('{"primitive": "change-lane-left"}').decide(
        "obs", None, ["maintain-speed", "accelerate", "decelerate"])
    assert dr.decision_status == "unavailable_fallback"
    assert dr.primitive == "maintain-speed" and dr.action_index == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest world2skills/tests/test_executor.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'world2skills.runtime.executor'`

- [ ] **Step 3: Write `world2skills/runtime/executor.py`**

```python
"""LLM-as-executor: read the skill card + rendered observation, return one
abstract primitive per step, mapped to a backend DiscreteMetaAction index."""
from __future__ import annotations

import json
import re

from .llm import LLMClient, Message, ModelSettings
from .types import DecisionResult, Grounding, SkillCard

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class LLMSkillExecutor:
    def __init__(self, skill_card: SkillCard, grounding: Grounding,
                 llm_client: LLMClient, prompt_version: str, name_to_index: dict[str, int]):
        self.skill_card = skill_card
        self.grounding = grounding
        self.llm = llm_client
        self.prompt_version = prompt_version
        self.name_to_index = name_to_index
        self._index_to_name = {i: n for n, i in name_to_index.items()}
        self._name_to_primitive = {v: k for k, v in grounding.primitive_map.items()}

    # --- prompt ---------------------------------------------------------
    def build_messages(self, obs_text: str, allowed: list[str]) -> list[Message]:
        card = self.skill_card
        system = (
            "You are an autonomous-driving policy. Each step you pick ONE abstract "
            "driving primitive to execute. Primitive meanings: "
            "maintain-speed=hold, accelerate=speed up, decelerate=slow down, "
            "change-lane-left / change-lane-right=move one lane over.\n"
            "Safety constraints (must always hold): "
            + "; ".join(card.safety_constraints) + "\n"
            'Respond with ONLY a JSON object: {"primitive": "<one allowed primitive>"} '
            "and nothing else."
        )
        user = (
            f"# Skill: {card.name}\n{card.description}\n\n"
            f"{card.skill_md_body}\n\n"
            f"# Parameters\n{json.dumps(card.parameters, ensure_ascii=False)}\n\n"
            f"# Current situation\n{obs_text}\n\n"
            f"# Allowed primitives this step\n{', '.join(allowed)}\n\n"
            'Reply with JSON only, e.g. {"primitive": "accelerate"}.'
        )
        return [Message("system", system), Message("user", user)]

    # --- decision -------------------------------------------------------
    def _safe_fallback(self, allowed: list[str]) -> str:
        if "maintain-speed" in allowed:
            return "maintain-speed"
        if allowed:
            return allowed[0]
        return "maintain-speed"

    def _parse_primitive(self, reply: str) -> str | None:
        text = reply.strip()
        m = _FENCE.search(text)
        if m:
            text = m.group(1)
        try:
            obj = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(obj, dict):
            return None
        prim = obj.get("primitive")
        if prim in self.skill_card.primitives:
            return prim
        return None  # unknown / not in this skill's primitive set

    def decide(self, obs_text: str, context, available_primitives: list[str]) -> DecisionResult:
        allowed = available_primitives or [self._safe_fallback([])]
        messages = self.build_messages(obs_text, allowed)
        status, fallback_reason = "ok", None

        try:
            result = self.llm.chat(messages, ModelSettings(), self.prompt_version)
            reply, request_hash = result.reply, result.request_hash
            cache_hit, latency = result.cache_hit, result.latency_ms
        except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash the episode
            reply, request_hash, cache_hit, latency = "", "", False, 0.0
            status, fallback_reason = "llm_error_fallback", f"{type(exc).__name__}: {exc}"

        if status == "ok":
            prim = self._parse_primitive(reply)
            if prim is None:
                status, fallback_reason = "parse_fallback", "unparseable or unknown primitive"
                prim = self._safe_fallback(allowed)
            elif prim not in allowed:
                status, fallback_reason = "unavailable_fallback", f"{prim} not available this step"
                prim = self._safe_fallback(allowed)
        else:
            prim = self._safe_fallback(allowed)

        backend = self.grounding.primitive_map[prim]
        action_index = self.name_to_index[backend]
        return DecisionResult(
            primitive=prim, backend_action=backend, action_index=action_index,
            request_hash=request_hash, raw_response=reply, cache_hit=cache_hit,
            latency_ms=latency, decision_status=status, fallback_reason=fallback_reason,
            available_primitives=list(allowed),
        )

    # --- availability ---------------------------------------------------
    def available_primitives(self, env) -> list[str]:
        """Abstract primitives currently available = env's available action indices
        mapped back through name_to_index and the inverse primitive_map, kept to
        this skill's primitive set."""
        prims: list[str] = []
        for i in env.unwrapped.action_type.get_available_actions():
            name = self._index_to_name.get(i)
            prim = self._name_to_primitive.get(name)
            if prim in self.skill_card.primitives and prim not in prims:
                prims.append(prim)
        return prims
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest world2skills/tests/test_executor.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add world2skills/runtime/executor.py world2skills/tests/test_executor.py
git commit -m "feat(w2s): add LLM executor with structured parse and fallbacks"
```

---

## Task 6: `runtime/env_factory.py` — build & configure the env

**Files:**
- Create: `world2skills/runtime/env_factory.py`
- Test: `world2skills/tests/test_env_factory.py`

- [ ] **Step 1: Write the failing test**

```python
# world2skills/tests/test_env_factory.py
import numpy as np
import pytest

pytest.importorskip("highway_env")

from world2skills.runtime.env_factory import make_env
from world2skills.runtime.skill_loader import load_skill, select_grounding


def test_make_env_returns_kinematics_and_action_map():
    g = select_grounding(load_skill("lane-change-overtake"), "highway-env")
    cfg = {"lanes_count": 4, "vehicles_count": 20, "duration": 40,
           "policy_frequency": 1, "simulation_frequency": 15, "obs_vehicles_count": 8}
    env, name_to_index = make_env(g, cfg, seed=0)
    try:
        obs = env.unwrapped.observation_type.observe()
        assert isinstance(obs, np.ndarray) and obs.shape == (8, len(g.observation["features"]))
        assert name_to_index["FASTER"] == 3 and name_to_index["LANE_LEFT"] == 0
        assert env.action_space.n == 5
    finally:
        env.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest world2skills/tests/test_env_factory.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'world2skills.runtime.env_factory'`

- [ ] **Step 3: Write `world2skills/runtime/env_factory.py`**

```python
"""Create and configure a highway-env environment from a skill grounding."""
from __future__ import annotations

from .types import Grounding


def make_env(grounding: Grounding, scenario_config: dict, seed: int):
    """Return (env, name_to_index). The env is configured and reset once (to
    materialize the action/observation types); the scenario will reset it again
    with the real seed. name_to_index is derived by inverting the env's
    action_type.actions (index->label), NOT from an undocumented attribute."""
    import gymnasium
    import highway_env  # noqa: F401 - registers the envs

    base = dict(scenario_config)
    obs_vehicles = base.pop("obs_vehicles_count", 8)
    features = grounding.observation.get("features", ["presence", "x", "y", "vx", "vy"])

    cfg = dict(base)
    cfg["observation"] = {
        "type": "Kinematics", "features": features, "vehicles_count": obs_vehicles,
        "normalize": False, "absolute": False, "see_behind": True, "order": "sorted",
    }
    cfg["action"] = {"type": "DiscreteMetaAction"}

    env = gymnasium.make(grounding.environment, render_mode=None)
    env.unwrapped.configure(cfg)
    env.reset(seed=seed)  # materialize spaces / action_type
    name_to_index = {name: idx for idx, name in env.unwrapped.action_type.actions.items()}
    return env, name_to_index
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest world2skills/tests/test_env_factory.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add world2skills/runtime/env_factory.py world2skills/tests/test_env_factory.py
git commit -m "feat(w2s): add env_factory (Kinematics + DiscreteMetaAction, inverted action map)"
```

---

## Task 7: `runtime/scenario.py` — deterministic overtake scenario + evaluator

**Files:**
- Create: `world2skills/runtime/scenario.py`
- Test: `world2skills/tests/test_scenario_evaluator.py`

The causal success logic lives in a pure function `judge_success` (unit-tested here, no deps). The env-coupled `reset/update/build_context` are exercised by the highway-env smoke test (Task 8).

- [ ] **Step 1: Write the failing test**

```python
# world2skills/tests/test_scenario_evaluator.py
from world2skills.runtime.scenario import judge_success


def test_success_requires_change_then_overtake_no_collision():
    ok, reason = judge_success(target_initially_ahead=True,
                               lane_change_completed_step=5, overtake_step=9, collision=False)
    assert ok is True and "step5" in reason and "step9" in reason


def test_overtake_before_lane_change_is_not_success():
    ok, reason = judge_success(True, lane_change_completed_step=9, overtake_step=5, collision=False)
    assert ok is False and "before" in reason


def test_lane_change_only_is_not_success():
    ok, _ = judge_success(True, lane_change_completed_step=5, overtake_step=None, collision=False)
    assert ok is False


def test_no_lane_change_is_not_success():
    ok, _ = judge_success(True, lane_change_completed_step=None, overtake_step=7, collision=False)
    assert ok is False


def test_collision_is_not_success():
    ok, reason = judge_success(True, lane_change_completed_step=5, overtake_step=9, collision=True)
    assert ok is False and reason == "collision"


def test_target_not_initially_ahead_is_not_success():
    ok, _ = judge_success(False, lane_change_completed_step=5, overtake_step=9, collision=False)
    assert ok is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest world2skills/tests/test_scenario_evaluator.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'world2skills.runtime.scenario'`

- [ ] **Step 3: Write `world2skills/runtime/scenario.py`**

```python
"""Deterministic lane-change-overtake scenario: enforce preconditions, bind the
target lead vehicle, track causal progress, and judge success.

Longitudinal positions are measured with `lane.local_coordinates(pos)[0]` on the
INITIAL ego lane (a stable axis), never world `position[0]`. The bound target is
an actual env Vehicle object, never an observation row.
"""
from __future__ import annotations

from .types import ObservationContext, SkillCard


class ScenarioSetupError(RuntimeError):
    """Raised when preconditions cannot be established for a seed."""


def judge_success(target_initially_ahead: bool, lane_change_completed_step: int | None,
                  overtake_step: int | None, collision: bool) -> tuple[bool, str]:
    if collision:
        return False, "collision"
    if not target_initially_ahead:
        return False, "target not initially ahead"
    if lane_change_completed_step is None:
        return False, "no lane change"
    if overtake_step is None:
        return False, "did not overtake target"
    if lane_change_completed_step < overtake_step:
        return True, f"lane change @step{lane_change_completed_step} then overtook @step{overtake_step}"
    return False, "overtook before lane change (causal order violated)"


class LaneChangeOvertakeScenario:
    def __init__(self, skill_card: SkillCard, *, success_margin: float = 5.0,
                 lead_speed_ratio: float = 0.6, front_clearance: float = 20.0,
                 rear_clearance: float = 20.0, spawn_gap: float = 30.0):
        self.skill_card = skill_card
        self.success_margin = success_margin
        self.lead_speed_ratio = lead_speed_ratio
        self.front_clearance = front_clearance
        self.rear_clearance = rear_clearance
        self.spawn_gap = spawn_gap
        self._reset_state()

    def _reset_state(self):
        self.initial_ego_lane = None
        self.initial_lead = None
        self.target_lane = None
        self.target_initially_ahead = False
        self.lane_change_completed_step = None
        self.overtake_step = None
        self.collision = False

    # --- config ---------------------------------------------------------
    def configure(self) -> dict:
        return {"lanes_count": 4, "vehicles_count": 30, "duration": 40,
                "policy_frequency": 1, "simulation_frequency": 15, "obs_vehicles_count": 8}

    # --- helpers --------------------------------------------------------
    def _long(self, env, position) -> float:
        lane = env.unwrapped.road.network.get_lane(self.initial_ego_lane)
        return float(lane.local_coordinates(position)[0])

    # --- reset (ORDER MATTERS) -----------------------------------------
    def reset(self, env, seed: int):
        self._reset_state()
        env.reset(seed=seed)
        u = env.unwrapped
        ego = u.vehicle
        self.initial_ego_lane = ego.lane_index
        s_ego = self._long(env, ego.position)
        ego_target = float(self.skill_card.parameters["target_speed"]["default"])
        slow = self.lead_speed_ratio * ego_target

        # 1) ensure a slower lead in the ego lane (reuse nearest front, else spawn)
        front, _ = u.road.neighbour_vehicles(ego, ego.lane_index)
        if front is None:
            from highway_env.vehicle.behavior import IDMVehicle
            front = IDMVehicle.make_on_lane(u.road, ego.lane_index, s_ego + self.spawn_gap, speed=slow)
            u.road.vehicles.append(front)
        front.target_speed = slow
        front.speed = min(front.speed, slow)
        if hasattr(front, "enable_lane_change"):
            front.enable_lane_change = False  # stop IDMVehicle MOBIL lane changes
        self.initial_lead = front
        self.target_initially_ahead = self._long(env, front.position) > s_ego

        # 2) choose an adjacent target lane and clear its safety band
        sides = u.road.network.side_lanes(ego.lane_index)
        self.target_lane = sides[0] if sides else None
        if self.target_lane is not None:
            tlane = u.road.network.get_lane(self.target_lane)
            for v in list(u.road.vehicles):
                if v is ego or v is front:
                    continue
                if v.lane_index == self.target_lane:
                    s_v = float(tlane.local_coordinates(v.position)[0])
                    if s_ego - self.rear_clearance <= s_v <= s_ego + self.front_clearance:
                        u.road.vehicles.remove(v)

        # 3) scene changed -> regenerate the observation
        obs = u.observation_type.observe()

        # 4) verify preconditions actually hold
        self.validate_preconditions(env)
        return obs, {}

    def validate_preconditions(self, env):
        u = env.unwrapped
        if self.initial_lead is None:
            raise ScenarioSetupError("no lead vehicle bound")
        ego_target = float(self.skill_card.parameters["target_speed"]["default"])
        if self.initial_lead.speed > self.lead_speed_ratio * ego_target + 1e-6:
            raise ScenarioSetupError("lead vehicle is not slower than ego target")
        if not self.target_initially_ahead:
            raise ScenarioSetupError("target not initially ahead of ego")
        if self.target_lane is None:
            raise ScenarioSetupError("no adjacent target lane")
        tlane = u.road.network.get_lane(self.target_lane)
        ego = u.vehicle
        s_ego = self._long(env, ego.position)
        for v in u.road.vehicles:
            if v is ego or v is self.initial_lead:
                continue
            if v.lane_index == self.target_lane:
                s_v = float(tlane.local_coordinates(v.position)[0])
                if s_ego - self.rear_clearance <= s_v <= s_ego + self.front_clearance:
                    raise ScenarioSetupError("target-lane safety band not clear")

    # --- per-step tracking ---------------------------------------------
    def update(self, env, t: int):
        u = env.unwrapped
        ego = u.vehicle
        if ego.crashed:
            self.collision = True
        if self.lane_change_completed_step is None and ego.lane_index != self.initial_ego_lane:
            self.lane_change_completed_step = t
        if self.overtake_step is None:
            s_ego = self._long(env, ego.position)
            s_lead = self._long(env, self.initial_lead.position)
            if s_ego - s_lead >= self.success_margin:
                self.overtake_step = t

    def is_terminal(self, env) -> tuple[bool, str]:
        ok, _ = judge_success(self.target_initially_ahead, self.lane_change_completed_step,
                              self.overtake_step, self.collision)
        return (True, "success") if ok else (False, "")

    def evaluate(self, env) -> tuple[bool, str]:
        return judge_success(self.target_initially_ahead, self.lane_change_completed_step,
                             self.overtake_step, self.collision)

    # --- context for the renderer --------------------------------------
    def _side_gap(self, env, lane_index):
        """Return (front_gap, rear_gap, rear_closing_speed) for a side lane, or Nones."""
        if lane_index is None:
            return None, None, None
        u = env.unwrapped
        ego = u.vehicle
        lane = u.road.network.get_lane(lane_index)
        s_ego = float(lane.local_coordinates(ego.position)[0])
        front, rear = u.road.neighbour_vehicles(ego, lane_index)
        fg = (float(lane.local_coordinates(front.position)[0]) - s_ego) if front else None
        rg = (s_ego - float(lane.local_coordinates(rear.position)[0])) if rear else None
        rc = (float(rear.speed) - float(ego.speed)) if rear else None
        return fg, rg, rc

    def build_context(self, env, prev_primitive, available_primitives) -> ObservationContext:
        u = env.unwrapped
        ego = u.vehicle
        ego_i = ego.lane_index[2]
        lane_indices = {l[2] for l in u.road.network.all_side_lanes(ego.lane_index)}
        base = ego.lane_index
        left_idx = (base[0], base[1], ego_i - 1) if (ego_i - 1) in lane_indices else None
        right_idx = (base[0], base[1], ego_i + 1) if (ego_i + 1) in lane_indices else None

        s_ego = self._long(env, ego.position)
        s_lead = self._long(env, self.initial_lead.position)
        lfg, lrg, lrc = self._side_gap(env, left_idx)
        rfg, rrg, rrc = self._side_gap(env, right_idx)

        return ObservationContext(
            available_primitives=list(available_primitives),
            ego_lane=ego_i, prev_primitive=prev_primitive,
            target_ahead=(s_lead - s_ego) > 0,
            target_gap_m=abs(s_lead - s_ego),
            target_rel_speed_mps=float(self.initial_lead.speed) - float(ego.speed),
            left_lane_exists=left_idx is not None, right_lane_exists=right_idx is not None,
            left_front_gap_m=lfg, left_rear_gap_m=lrg, left_rear_closing_speed_mps=lrc,
            right_front_gap_m=rfg, right_rear_gap_m=rrg, right_rear_closing_speed_mps=rrc,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest world2skills/tests/test_scenario_evaluator.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add world2skills/runtime/scenario.py world2skills/tests/test_scenario_evaluator.py
git commit -m "feat(w2s): add deterministic overtake scenario with causal evaluator"
```

---

## Task 8: `evaluation/episode.py` — run one episode + smoke test

**Files:**
- Create: `world2skills/evaluation/episode.py`
- Test: `world2skills/tests/test_episode_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
# world2skills/tests/test_episode_smoke.py
import pytest

pytest.importorskip("highway_env")

from world2skills.evaluation.episode import run_episode
from world2skills.runtime.env_factory import make_env
from world2skills.runtime.executor import LLMSkillExecutor
from world2skills.runtime.llm import MockLLMClient
from world2skills.runtime.scenario import LaneChangeOvertakeScenario
from world2skills.runtime.skill_loader import load_skill, select_grounding


def test_scripted_episode_runs_and_records_metrics():
    card = load_skill("lane-change-overtake")
    g = select_grounding(card, "highway-env")
    scenario = LaneChangeOvertakeScenario(card)
    env, name_to_index = make_env(g, scenario.configure(), seed=0)
    # scripted "policy": always accelerate (valid JSON every step)
    executor = LLMSkillExecutor(card, g, MockLLMClient(lambda m: '{"primitive": "accelerate"}'),
                                "lco-v1", name_to_index)

    result = run_episode(env, executor, scenario, seed=0, max_steps=40)

    assert result.status == "ok"
    assert result.target_initially_ahead is True          # scenario enforced the precondition
    assert result.steps >= 1 and len(result.step_records) == result.steps
    for rec in result.step_records:
        assert rec.reward is not None and isinstance(rec.crashed, bool)
    # scenario_completed implies gym terminated/truncated stayed False at the early stop
    if result.scenario_completed:
        assert result.terminated is False and result.truncated is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest world2skills/tests/test_episode_smoke.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'world2skills.evaluation.episode'`

- [ ] **Step 3: Write `world2skills/evaluation/episode.py`**

```python
"""Run a single episode: obs -> render -> executor.decide -> env.step ->
scenario.update, assembling a StepRecord after each step."""
from __future__ import annotations

from ..runtime.obs_render import render
from ..runtime.types import EpisodeResult, StepRecord


def run_episode(env, executor, scenario, seed: int, max_steps: int) -> EpisodeResult:
    feature_names = executor.grounding.observation.get(
        "features", ["presence", "x", "y", "vx", "vy"])
    records: list[StepRecord] = []
    episode_return = 0.0
    speeds: list[float] = []
    prev_primitive = None
    parse_failures = unavailable = llm_errors = 0
    terminated = truncated = False
    scenario_completed = False
    max_steps_reached = False
    reason = ""

    try:
        obs, _ = scenario.reset(env, seed)
        u = env.unwrapped
        for t in range(max_steps):
            avail = executor.available_primitives(env)
            ctx = scenario.build_context(env, prev_primitive, avail)
            obs_text = render(obs, ctx, feature_names)
            dr = executor.decide(obs_text, ctx, avail)

            obs, reward, terminated, truncated, info = env.step(dr.action_index)
            scenario.update(env, t)
            crashed = bool(info.get("crashed", u.vehicle.crashed))

            records.append(StepRecord.from_decision(
                dr, t=t, obs_summary=obs_text[:200], reward=float(reward), crashed=crashed))
            episode_return += float(reward)
            speeds.append(float(info.get("speed", u.vehicle.speed)))
            prev_primitive = dr.primitive
            if dr.decision_status == "parse_fallback":
                parse_failures += 1
            elif dr.decision_status == "unavailable_fallback":
                unavailable += 1
            elif dr.decision_status == "llm_error_fallback":
                llm_errors += 1

            done, _ = scenario.is_terminal(env)
            if done:
                scenario_completed = True
                reason = "success"
                break
            if terminated or truncated:
                reason = "crashed" if crashed else "time"
                break
        else:
            max_steps_reached = True
            reason = "max_steps"

        success, success_reason = scenario.evaluate(env)
        crashed_any = any(r.crashed for r in records)
        return EpisodeResult(
            seed=seed, status="ok", success=success, success_reason=success_reason,
            episode_return=episode_return, crashed=crashed_any, steps=len(records),
            mean_speed=(sum(speeds) / len(speeds) if speeds else 0.0),
            parse_failures=parse_failures, unavailable_action_attempts=unavailable,
            llm_errors=llm_errors, terminated=bool(terminated), truncated=bool(truncated),
            max_steps_reached=max_steps_reached, scenario_completed=scenario_completed,
            termination_reason=reason, target_initially_ahead=scenario.target_initially_ahead,
            lane_change_completed_step=scenario.lane_change_completed_step,
            overtake_step=scenario.overtake_step, step_records=records,
        )
    except Exception as exc:  # noqa: BLE001 - one bad episode must not stop the batch
        return EpisodeResult(
            seed=seed, status="error", success=False, success_reason="",
            episode_return=episode_return, crashed=False, steps=len(records),
            mean_speed=0.0, parse_failures=parse_failures,
            unavailable_action_attempts=unavailable, llm_errors=llm_errors,
            terminated=False, truncated=False, max_steps_reached=False,
            scenario_completed=False, termination_reason="error",
            target_initially_ahead=getattr(scenario, "target_initially_ahead", False),
            lane_change_completed_step=getattr(scenario, "lane_change_completed_step", None),
            overtake_step=getattr(scenario, "overtake_step", None),
            exception_type=type(exc).__name__, step_records=records,
        )
    finally:
        env.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest world2skills/tests/test_episode_smoke.py -q`
Expected: PASS (1 passed). This exercises scenario `reset/update/build_context` end-to-end on real highway-env.

- [ ] **Step 5: Commit**

```bash
git add world2skills/evaluation/episode.py world2skills/tests/test_episode_smoke.py
git commit -m "feat(w2s): add episode runner + highway-env smoke test"
```

---

## Task 9: `evaluation/metrics.py` — aggregation + output writing

**Files:**
- Create: `world2skills/evaluation/metrics.py`
- Test: `world2skills/tests/test_results_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# world2skills/tests/test_results_schema.py
import json
from pathlib import Path

from world2skills.evaluation.metrics import aggregate, write_outputs
from world2skills.runtime.types import EpisodeResult


def _ep(seed, status="ok", success=False, crashed=False, ret=10.0):
    return EpisodeResult(
        seed=seed, status=status, success=success, success_reason="",
        episode_return=ret, crashed=crashed, steps=5, mean_speed=20.0,
        parse_failures=0, unavailable_action_attempts=0, llm_errors=0,
        terminated=False, truncated=False, max_steps_reached=False,
        scenario_completed=success, termination_reason="", target_initially_ahead=True,
        lane_change_completed_step=None, overtake_step=None,
    )


def _meta():
    return dict(model="mock", prompt_version="lco-v1", skill="lane-change-overtake",
                environment="highway-v0", seeds=[0, 1, 2])


def test_aggregate_counts_and_rates():
    results = [_ep(0, success=True, ret=20.0),
               _ep(1, success=False, crashed=True, ret=5.0),
               _ep(2, status="error", ret=0.0)]
    batch = aggregate(results, **_meta())
    assert batch.schema_version == "0.1"
    assert batch.requested_episodes == 3
    assert batch.completed_episodes == 2 and batch.error_episodes == 1
    assert batch.successful_episodes == 1
    assert batch.success_rate == 1 / 3            # denominator = requested
    assert batch.collision_rate == 1 / 3
    assert batch.mean_return == 12.5              # only status=ok episodes (20, 5)


def test_write_outputs_creates_files(tmp_path):
    results = [_ep(0, success=True), _ep(1, status="error")]
    batch = aggregate(results, **_meta())
    write_outputs(tmp_path, batch, results, config={"highway_env": "1.12.0"})
    rj = json.loads((tmp_path / "results.json").read_text())
    assert rj["schema_version"] == "0.1" and rj["seeds"] == [0, 1, 2]
    assert "step_records" not in rj["results"][0]         # traces excluded from results.json
    assert rj["results"][1]["status"] == "error"
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "episode_0.jsonl").exists()        # per-episode trace file
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest world2skills/tests/test_results_schema.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'world2skills.evaluation.metrics'`

- [ ] **Step 3: Write `world2skills/evaluation/metrics.py`**

```python
"""Aggregate EpisodeResults into a BatchResult and write run artifacts."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..runtime.types import BatchResult, EpisodeResult

SCHEMA_VERSION = "0.1"


def episode_to_result_dict(r: EpisodeResult) -> dict:
    """Per-episode summary for results.json (step_records are written separately)."""
    d = asdict(r)
    d.pop("step_records", None)
    return d


def aggregate(results: list[EpisodeResult], *, model: str, prompt_version: str,
              skill: str, environment: str, seeds: list[int]) -> BatchResult:
    requested = len(seeds)
    completed = sum(1 for r in results if r.status == "ok")
    errors = sum(1 for r in results if r.status == "error")
    successful = sum(1 for r in results if r.status == "ok" and r.success)
    collisions = sum(1 for r in results if r.status == "ok" and r.crashed)
    ok_returns = [r.episode_return for r in results if r.status == "ok"]
    return BatchResult(
        schema_version=SCHEMA_VERSION, model=model, prompt_version=prompt_version,
        skill=skill, environment=environment, seeds=list(seeds),
        requested_episodes=requested, completed_episodes=completed,
        error_episodes=errors, successful_episodes=successful,
        success_rate=(successful / requested if requested else 0.0),
        collision_rate=(collisions / requested if requested else 0.0),
        mean_return=(sum(ok_returns) / len(ok_returns) if ok_returns else 0.0),
        results=[episode_to_result_dict(r) for r in results],
    )


def write_outputs(out_dir, batch: BatchResult, results: list[EpisodeResult], config: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(asdict(batch), ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in results:
        with (out / f"episode_{r.seed}.jsonl").open("w", encoding="utf-8") as fh:
            for rec in r.step_records:
                fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest world2skills/tests/test_results_schema.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add world2skills/evaluation/metrics.py world2skills/tests/test_results_schema.py
git commit -m "feat(w2s): add metrics aggregation + output writing"
```

---

## Task 10: `evaluation/run.py` — CLI + experiment config + end-to-end smoke

**Files:**
- Create: `world2skills/evaluation/run.py`
- Create: `world2skills/experiments/configs/lane-change-overtake.yaml`
- Test: `world2skills/tests/test_cli_smoke.py`

- [ ] **Step 1: Write the experiment config `world2skills/experiments/configs/lane-change-overtake.yaml`**

```yaml
# M1 experiment config. Scenario params override LaneChangeOvertakeScenario defaults.
skill: lane-change-overtake
backend: highway-env
prompt_version: lco-v1
max_steps: 40
scenario:
  success_margin: 5.0
  lead_speed_ratio: 0.6
  front_clearance: 20.0
  rear_clearance: 20.0
```

- [ ] **Step 2: Write the failing test**

```python
# world2skills/tests/test_cli_smoke.py
import json

import pytest

pytest.importorskip("highway_env")

from world2skills.evaluation.run import main


def test_cli_mock_writes_results(tmp_path):
    out = tmp_path / "run1"
    rc = main([
        "--skill", "lane-change-overtake", "--seeds", "0",
        "--mock", "--out", str(out), "--max-steps", "10",
    ])
    assert rc == 0
    rj = json.loads((out / "results.json").read_text())
    assert rj["schema_version"] == "0.1"
    assert rj["requested_episodes"] == 1 and rj["seeds"] == [0]
    cfg = json.loads((out / "config.json").read_text())
    assert "highway_env" in cfg and "gymnasium" in cfg
    assert (out / "episode_0.jsonl").exists()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest world2skills/tests/test_cli_smoke.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'world2skills.evaluation.run'`

- [ ] **Step 4: Write `world2skills/evaluation/run.py`**

```python
"""CLI entry point for the M1 experiment loop.

Example:
  source scripts/connect_gpt54.sh
  python -m world2skills.evaluation.run --skill lane-change-overtake \
      --seeds 0 1 2 3 4 --model gpt-5.4 --out world2skills/outputs/lco_gpt54
"""
from __future__ import annotations

import argparse
import os
import sys

from ..runtime.env_factory import make_env
from ..runtime.executor import LLMSkillExecutor
from ..runtime.llm import AzureResponsesClient, MockLLMClient, OpenAIClient
from ..runtime.scenario import LaneChangeOvertakeScenario
from ..runtime.skill_loader import load_skill, select_grounding
from .episode import run_episode
from .metrics import aggregate, write_outputs


def _installed_versions() -> dict:
    import gymnasium
    import highway_env
    return {"highway_env": highway_env.__version__, "gymnasium": gymnasium.__version__}


def _build_client(model: str, mock: bool):
    if mock:
        return MockLLMClient(lambda messages: '{"primitive": "accelerate"}')
    # GPT-5.4 goes through the Azure Responses proxy; everything else via chat.completions.
    if model.startswith("gpt-5"):
        return AzureResponsesClient(model=os.getenv("OPENAI_MODEL", model))
    return OpenAIClient(model=model)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="World2Skills M1 experiment loop")
    p.add_argument("--skill", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--model", default="gpt-5.4")
    p.add_argument("--mock", action="store_true", help="use MockLLMClient (no model calls)")
    p.add_argument("--prompt-version", default="lco-v1")
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    card = load_skill(args.skill)
    grounding = select_grounding(card, "highway-env")
    model_name = "mock" if args.mock else args.model
    client = _build_client(args.model, args.mock)

    results = []
    for seed in args.seeds:
        scenario = LaneChangeOvertakeScenario(card)
        env, name_to_index = make_env(grounding, scenario.configure(), seed=seed)
        executor = LLMSkillExecutor(card, grounding, client, args.prompt_version, name_to_index)
        results.append(run_episode(env, executor, scenario, seed=seed, max_steps=args.max_steps))

    batch = aggregate(results, model=model_name, prompt_version=args.prompt_version,
                      skill=args.skill, environment=grounding.environment, seeds=args.seeds)
    config = {
        "model": model_name, "prompt_version": args.prompt_version, "skill": args.skill,
        "environment": grounding.environment, "seeds": args.seeds, "max_steps": args.max_steps,
        "scenario": scenario.configure(), **_installed_versions(),
    }
    write_outputs(args.out, batch, results, config)
    print(f"[w2s] wrote {args.out}/results.json  "
          f"success_rate={batch.success_rate:.2f} "
          f"({batch.successful_episodes}/{batch.requested_episodes}), "
          f"errors={batch.error_episodes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest world2skills/tests/test_cli_smoke.py -q`
Expected: PASS (1 passed)

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest world2skills/tests -q`
Expected: PASS (all tests green)

- [ ] **Step 7: Commit**

```bash
git add world2skills/evaluation/run.py world2skills/experiments/configs/lane-change-overtake.yaml world2skills/tests/test_cli_smoke.py
git commit -m "feat(w2s): add CLI runner + experiment config + end-to-end smoke"
```

---

## Task 11: M1 acceptance run (manual — real GPT-5.4, not pytest)

**Files:** none (produces `world2skills/outputs/lco_gpt54/`, gitignored)

This is the acceptance gate from spec §12. It needs the Azure proxy running and `az` login, so it is NOT part of the pytest suite.

- [ ] **Step 1: Configure GPT-5.4**

```bash
cd /data/yifan/zrx/World2Skills
source scripts/connect_gpt54.sh
```
Expected: `[INFO] GPT-5.4 configured ...` and `OPENAI_BASE_URL` / `OPENAI_MODEL` exported.

- [ ] **Step 2: Run 5 fixed seeds**

```bash
python -m world2skills.evaluation.run \
  --skill lane-change-overtake --seeds 0 1 2 3 4 \
  --model gpt-5.4 --out world2skills/outputs/lco_gpt54
```
Expected: a final line `[w2s] wrote world2skills/outputs/lco_gpt54/results.json success_rate=... (N/5), errors=0`.

- [ ] **Step 3: Verify the acceptance criteria (spec §12)**

```bash
python - <<'PY'
import json
d = json.load(open("world2skills/outputs/lco_gpt54/results.json"))
assert d["schema_version"] == "0.1"
assert d["completed_episodes"] == d["requested_episodes"] == 5, "must be 5/5 status=ok"
assert d["successful_episodes"] >= 1, "at least one success required"
for r in d["results"]:
    assert r["target_initially_ahead"] is True  # scenario precondition held every seed
    for k in ("parse_failures", "unavailable_action_attempts", "llm_errors"):
        assert k in r
print("M1 ACCEPTANCE PASS:",
      f'{d["successful_episodes"]}/{d["requested_episodes"]} success,',
      f'collision_rate={d["collision_rate"]:.2f}, mean_return={d["mean_return"]:.1f}')
PY
```
Expected: `M1 ACCEPTANCE PASS: >=1/5 success, ...`. If `completed_episodes != 5` or `successful_episodes == 0`, M1 is not met — investigate traces in `episode_<seed>.jsonl` (check `decision_status`/`fallback_reason`) before declaring done.

- [ ] **Step 4: (No commit — outputs are gitignored.)** Record the acceptance numbers in the PR/handoff notes.

---

## Self-Review (completed during planning)

**1. Spec coverage:** §1.3 causal success → `judge_success` + Task 7 + Task 8; §2 layout → Task 0 + File Structure; §3.1 types → Task 1; §3.2 llm+cache → Task 3; §3.3 loader → Task 2; §3.4 scenario (reset re-observe, enable_lane_change=False, target-lane clearance, validate_preconditions, is_terminal) → Task 7 code; §3.5 renderer (unified frame, front/rear gaps) → Task 4; §3.6 executor (injected name_to_index, DecisionResult, structured parse, availability) → Task 5; §3.7 env_factory (inverted action map, normalize=False) → Task 6; §3.8 episode (StepRecord assembly, stop conditions) → Task 8; §3.9/§7 metrics + results.json schema (counts, aggregation) → Task 9; §3.10 CLI + config.json versions → Task 10; §10 deps/pins → Task 0; §11 tests → Tasks 1-10; §12 acceptance → Task 11; §13 assumptions encoded as defaults in scenario/env_factory.

**2. Placeholder scan:** No TBD/TODO; every code step contains full file content; every test step has real assertions.

**3. Type consistency:** `DecisionResult`/`StepRecord` fields match across types.py, executor.py, episode.py; `make_request_hash` signature matches test and clients; `aggregate(...)` keyword args match run.py and the schema test; `run_episode(env, executor, scenario, seed, max_steps)` matches episode.py, the smoke test, and run.py; `make_env(grounding, scenario_config, seed) -> (env, name_to_index)` matches factory, test, run.py.

**Note on minor spec deltas (intentional, consistent with design intent):** `scenario.reset` returns `(obs, info)` and the episode builds the first `ObservationContext` in-loop (context needs `available_primitives`, which only the executor+env can compute) — matches §3.8's per-step `build_context`. `run_episode` takes `seed` (the spec signature omitted it; the seed is required to reset the scenario).
