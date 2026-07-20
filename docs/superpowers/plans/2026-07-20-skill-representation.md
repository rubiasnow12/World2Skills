# World2Skills 第一阶段实施计划：自动驾驶 Skill 构建与表示

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立自动驾驶领域统一的 Skill 表示形式，交付元模型规范、机读 JSON Schema、以及 5 个通过校验的手写种子技能。

**Architecture:** 每个技能是一个文件夹，含标准 `SKILL.md`（发现层，只放 `name`/`description`/`metadata`）与 `skill.yaml`（完整驾驶表示，JSON Schema 校验对象）。抽象接口（`interface`）与后端绑定（`groundings`）分离，以 highway-env 为参照后端。本阶段不写自研运行时；校验用现成 `jsonschema` 库 + skill-creator 的 `quick_validate.py`。

**Tech Stack:** Markdown、YAML、JSON Schema (draft 2020-12)、Python 3.11、`jsonschema` 库（已装）、`pyyaml`（已装）。

**Source spec:** `docs/superpowers/specs/2026-07-20-skill-representation-design.md`

---

## File Structure

本阶段全部产物在新建的自包含目录 `world2skills/` 下，不改动现有代码：

- `world2skills/schema/skill.schema.json` — 机读 JSON Schema，校验 `skill.yaml` 结构。
- `world2skills/docs/skill-representation-spec.md` — 元模型规范（人读），字段语义 + 粒度规则 + 条件约定。
- `world2skills/tools/validate_skills.py` — 校验驱动脚本（驱动现成 `jsonschema` 库 + 调 `quick_validate.py`；非自研校验逻辑）。
- `world2skills/skills/<skill-id>/SKILL.md` — 5 个，发现层。
- `world2skills/skills/<skill-id>/skill.yaml` — 5 个，表示层。
- `world2skills/README.md` — 阶段说明 + 新增技能指南 + 校验命令。

**构建顺序（每个技能都要过校验，所以先有 schema 和校验脚本）：**
1. Schema（Task 1–2）
2. 校验脚本（Task 3）
3. 首个种子技能 `unprotected-left-turn`，走通「写→校验→修 schema」闭环（Task 4）
4. 其余 4 个种子技能（Task 5–8）
5. 规范文档 + README（Task 9–10）
6. 全量校验 + 人工核对清单（Task 11）

> **给执行者的领域备注：** highway-env 的元动作是 `{LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER}`。抽象原语与后端动作的映射约定：`maintain-speed→IDLE`、`accelerate→FASTER`、`decelerate→SLOWER`、`change-lane-left→LANE_LEFT`、`change-lane-right→LANE_RIGHT`。`interface`/`execution` 只用抽象原语；后端名字只出现在 `groundings[].primitive_map`。

---

## Task 1: 建立目录骨架与 JSON Schema 顶层结构

**Files:**
- Create: `world2skills/schema/skill.schema.json`

- [ ] **Step 1: 创建目录骨架**

Run:
```bash
mkdir -p world2skills/schema world2skills/docs world2skills/tools world2skills/skills
```

- [ ] **Step 2: 写 schema 顶层（元数据 + 简单标量字段）**

写入 `world2skills/schema/skill.schema.json`。本步只放顶层骨架与标量字段，复杂对象（interface/execution/groundings）在 Task 2 补全。

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://world2skills/schema/skill.schema.json",
  "title": "World2Skills autonomous-driving skill representation",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "schema_version", "name", "version", "domain", "category",
    "description", "entities", "interface", "preconditions",
    "execution", "effects", "success_criteria", "failure_criteria",
    "safety_constraints", "failure_modes", "termination", "groundings"
  ],
  "properties": {
    "schema_version": { "type": "string" },
    "name": { "type": "string", "pattern": "^[a-z0-9]+(-[a-z0-9]+)*$" },
    "version": { "type": "string" },
    "domain": { "type": "string", "const": "autonomous-driving" },
    "category": { "type": "string", "enum": ["longitudinal", "lateral", "interaction"] },
    "tags": { "type": "array", "items": { "type": "string" } },
    "description": { "type": "string", "minLength": 1 },
    "entities": {
      "type": "array", "minItems": 1,
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["name", "role"],
        "properties": {
          "name": { "type": "string" },
          "role": { "type": "string", "enum": ["controlled-vehicle", "traffic", "static-topology"] }
        }
      }
    },
    "parameters": {
      "type": "object",
      "additionalProperties": {
        "type": "object", "additionalProperties": false,
        "required": ["type", "description"],
        "properties": {
          "type": { "type": "string", "enum": ["float", "int", "bool", "string"] },
          "unit": { "type": "string" },
          "range": { "type": "array", "items": { "type": "number" }, "minItems": 2, "maxItems": 2 },
          "default": {},
          "description": { "type": "string" }
        }
      }
    },
    "preconditions": { "type": "array", "items": { "type": "string" }, "minItems": 1 },
    "effects": { "type": "array", "items": { "type": "string" }, "minItems": 1 },
    "success_criteria": { "type": "array", "items": { "type": "string" }, "minItems": 1 },
    "failure_criteria": { "type": "array", "items": { "type": "string" }, "minItems": 1 },
    "safety_constraints": { "type": "array", "items": { "type": "string" }, "minItems": 1 },
    "failure_modes": { "type": "array", "items": { "type": "string" }, "minItems": 1 },
    "termination": { "type": "string", "minLength": 1 },
    "related_skills": { "type": "array", "items": { "type": "string" } },
    "interface": { "type": "object" },
    "execution": { "type": "object" },
    "groundings": { "type": "array" }
  }
}
```

- [ ] **Step 3: 确认是合法 JSON 且是合法 draft 2020-12 schema**

Run:
```bash
python3 -c "import json; from jsonschema import Draft202012Validator; s=json.load(open('world2skills/schema/skill.schema.json')); Draft202012Validator.check_schema(s); print('schema OK')"
```
Expected: 输出 `schema OK`，无异常。

- [ ] **Step 4: Commit**

```bash
git add world2skills/schema/skill.schema.json
git commit -m "feat(world2skills): add skill.yaml JSON Schema top-level structure"
```

---

## Task 2: 补全 Schema 的 interface / execution / groundings 子结构

**Files:**
- Modify: `world2skills/schema/skill.schema.json`（替换 `interface`/`execution`/`groundings` 三个占位定义）

- [ ] **Step 1: 用完整定义替换 `interface` 占位**

把 Task 1 中 `"interface": { "type": "object" }` 替换为：

```json
    "interface": {
      "type": "object", "additionalProperties": false,
      "required": ["observations", "actions"],
      "properties": {
        "observations": {
          "type": "array", "minItems": 1,
          "items": {
            "type": "object", "additionalProperties": false,
            "required": ["name", "type"],
            "properties": {
              "name": { "type": "string" },
              "type": { "type": "string", "enum": ["kinematic", "image", "grid", "track"] },
              "features": { "type": "array", "items": { "type": "string" } }
            }
          }
        },
        "actions": {
          "type": "array", "minItems": 1,
          "items": {
            "type": "object", "additionalProperties": false,
            "required": ["name", "type", "primitives"],
            "properties": {
              "name": { "type": "string" },
              "type": { "type": "string", "enum": ["discrete-meta", "continuous", "trajectory"] },
              "primitives": {
                "type": "array", "minItems": 1,
                "items": {
                  "type": "string",
                  "enum": ["maintain-speed", "accelerate", "decelerate", "change-lane-left", "change-lane-right"]
                }
              }
            }
          }
        }
      }
    },
```

- [ ] **Step 2: 用完整定义替换 `execution` 占位**

把 `"execution": { "type": "object" }` 替换为闭合状态图定义：

```json
    "execution": {
      "type": "object", "additionalProperties": false,
      "required": ["entry", "steps"],
      "properties": {
        "entry": { "type": "string" },
        "steps": {
          "type": "array", "minItems": 1,
          "items": {
            "type": "object", "additionalProperties": false,
            "required": ["id", "kind", "transitions"],
            "properties": {
              "id": { "type": "string" },
              "kind": { "type": "string", "enum": ["observe", "act", "branch"] },
              "action": {
                "type": "string",
                "enum": ["maintain-speed", "accelerate", "decelerate", "change-lane-left", "change-lane-right"]
              },
              "transitions": {
                "type": "array", "minItems": 1,
                "items": {
                  "type": "object", "additionalProperties": false,
                  "required": ["condition", "next"],
                  "properties": {
                    "condition": { "type": "string" },
                    "next": { "type": "string" }
                  }
                }
              }
            }
          }
        }
      }
    },
```

- [ ] **Step 3: 用完整定义替换 `groundings` 占位**

把 `"groundings": { "type": "array" }` 替换为：

```json
    "groundings": {
      "type": "array", "minItems": 1,
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["backend", "backend_version", "environment", "observation", "action", "primitive_map"],
        "properties": {
          "backend": { "type": "string" },
          "backend_version": { "type": "string" },
          "environment": { "type": "string" },
          "observation": {
            "type": "object", "additionalProperties": false,
            "required": ["type"],
            "properties": {
              "type": { "type": "string" },
              "features": { "type": "array", "items": { "type": "string" } }
            }
          },
          "action": {
            "type": "object", "additionalProperties": false,
            "required": ["type"],
            "properties": { "type": { "type": "string" } }
          },
          "primitive_map": {
            "type": "object",
            "additionalProperties": { "type": "string" }
          }
        }
      }
    }
```

- [ ] **Step 4: 确认仍是合法 draft 2020-12 schema**

Run:
```bash
python3 -c "import json; from jsonschema import Draft202012Validator; s=json.load(open('world2skills/schema/skill.schema.json')); Draft202012Validator.check_schema(s); print('schema OK')"
```
Expected: 输出 `schema OK`。

- [ ] **Step 5: Commit**

```bash
git add world2skills/schema/skill.schema.json
git commit -m "feat(world2skills): complete interface/execution/groundings schema"
```

---

## Task 3: 校验驱动脚本

写一个驱动脚本：对每个技能目录，(a) 调外部 `quick_validate.py` 校验 `SKILL.md`；(b) 用现成 `jsonschema` 库对照 schema 校验 `skill.yaml`。**这是校验驱动器（driver），不含自研校验规则**——结构规则全在 schema 与 `quick_validate.py` 里。跨文件/引用完整性（图闭合、primitive_map 覆盖、name 三处一致）留作人工核对（见 Task 11），本脚本不做。

**Files:**
- Create: `world2skills/tools/validate_skills.py`

- [ ] **Step 1: 写脚本**

写入 `world2skills/tools/validate_skills.py`：

```python
#!/usr/bin/env python3
"""Structural validator driver for World2Skills skills.

Drives two off-the-shelf checks per skill directory:
  1. SKILL.md  -> skill-creator's quick_validate.py (standard Agent Skills)
  2. skill.yaml -> jsonschema library against skill.schema.json

This is a driver only. It intentionally does NOT check cross-file or
reference integrity (graph closure, primitive_map coverage, name agreement
across files) -- those are manual checks per the spec.
"""
import json
import subprocess
import sys
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schema" / "skill.schema.json"
SKILLS_DIR = ROOT / "skills"

# External skill-creator validator (authoritative Agent Skills checker).
QUICK_VALIDATE = Path(
    "/home/yifanyang/.claude/plugins/marketplaces/claude-plugins-official"
    "/plugins/skill-creator/skills/skill-creator/scripts/quick_validate.py"
)


def validate_skill_md(skill_dir: Path) -> list[str]:
    if not QUICK_VALIDATE.exists():
        return [f"quick_validate.py not found at {QUICK_VALIDATE}"]
    proc = subprocess.run(
        [sys.executable, str(QUICK_VALIDATE), str(skill_dir)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return [f"SKILL.md: {proc.stdout.strip() or proc.stderr.strip()}"]
    return []


def validate_skill_yaml(skill_dir: Path, validator: Draft202012Validator) -> list[str]:
    yaml_path = skill_dir / "skill.yaml"
    if not yaml_path.exists():
        return ["skill.yaml not found"]
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return [f"skill.yaml: invalid YAML: {exc}"]
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
    return [f"skill.yaml: {'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors]


def main() -> int:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    skill_dirs = sorted(p for p in SKILLS_DIR.iterdir() if p.is_dir())
    if not skill_dirs:
        print("no skills found")
        return 1

    total_errors = 0
    for skill_dir in skill_dirs:
        errors = validate_skill_md(skill_dir) + validate_skill_yaml(skill_dir, validator)
        if errors:
            total_errors += len(errors)
            print(f"[FAIL] {skill_dir.name}")
            for err in errors:
                print(f"    - {err}")
        else:
            print(f"[OK]   {skill_dir.name}")

    print(f"\n{len(skill_dirs)} skills checked, {total_errors} error(s).")
    return 1 if total_errors else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: 对空 skills 目录运行，确认脚本本身能跑（无技能时报 "no skills found"）**

Run:
```bash
python3 world2skills/tools/validate_skills.py; echo "exit=$?"
```
Expected: 输出 `no skills found`，`exit=1`。（此时还没有任何技能，属正常。）

- [ ] **Step 3: Commit**

```bash
git add world2skills/tools/validate_skills.py
git commit -m "feat(world2skills): add structural validation driver"
```

---

## Task 4: 首个种子技能 `unprotected-left-turn`（走通校验闭环）

这是最完整的示例（spec 第 6 节已给全量）。先把它做到过校验，验证 schema 与脚本正确。

**Files:**
- Create: `world2skills/skills/unprotected-left-turn/SKILL.md`
- Create: `world2skills/skills/unprotected-left-turn/skill.yaml`

- [ ] **Step 1: 写 `SKILL.md`（发现层，description 单行）**

写入 `world2skills/skills/unprotected-left-turn/SKILL.md`：

```markdown
---
name: unprotected-left-turn
description: Use when the ego must turn left across oncoming traffic at an unsignalized intersection, yielding to cross and oncoming vehicles.
metadata:
  schema_version: "0.1"
  representation: skill.yaml
---

# Unprotected Left Turn

## When to use
The ego approaches an unsignalized intersection intending to turn left, and
must cross the path of oncoming and cross traffic that has right of way.

## Procedure
1. Stop or slow at the intersection entry and observe oncoming traffic.
2. Estimate the time gap to the nearest oncoming vehicle.
3. If the gap is below the yield threshold, keep waiting (decelerate/hold).
4. When the gap is sufficient, accelerate through the conflict zone.
5. Once past the conflict zone, resume normal speed on the target lane.

## Reasoning cues
- A larger relative speed of oncoming traffic requires a larger accepted gap.
- Never commit on a marginal gap; waiting one more cycle is cheap, a collision
  is not.

## Failure modes & recovery
- Committing on an insufficient gap risks a side collision — abort by holding
  if the ego has not yet entered the conflict zone.
- Over-cautious freezing blocks the intersection — if no vehicle is within the
  yield gap, proceed rather than waiting indefinitely.
```

- [ ] **Step 2: 写 `skill.yaml`（完整表示，闭合图 + primitive_map）**

写入 `world2skills/skills/unprotected-left-turn/skill.yaml`：

```yaml
schema_version: "0.1"
name: unprotected-left-turn
version: "1.0"
domain: autonomous-driving
category: interaction
tags: [yield, unsignalized, left-turn]

description: >
  Turn left across oncoming traffic at an unsignalized intersection,
  yielding to cross and oncoming vehicles.

entities:
  - { name: ego, role: controlled-vehicle }
  - { name: oncoming_vehicles, role: traffic }
  - { name: intersection, role: static-topology }

parameters:
  target_speed:
    type: float
    unit: m/s
    range: [0, 10]
    default: 4
    description: cruising speed when the path is clear
  yield_gap:
    type: float
    unit: s
    range: [2, 6]
    default: 3
    description: minimum time gap to oncoming traffic before committing

interface:
  observations:
    - { name: surrounding_vehicles, type: kinematic, features: [presence, x, y, vx, vy, heading] }
    - { name: ego_state, type: kinematic }
  actions:
    - { name: longitudinal, type: discrete-meta, primitives: [maintain-speed, accelerate, decelerate] }

preconditions:
  - "intersection.signal == none"
  - "ego.intent == turn_left"

execution:
  entry: assess-gap
  steps:
    - id: assess-gap
      kind: observe
      transitions:
        - { condition: "always", next: decide }
    - id: decide
      kind: branch
      transitions:
        - { condition: "gap < yield_gap", next: wait }
        - { condition: "gap >= yield_gap", next: commit }
    - id: wait
      kind: act
      action: decelerate
      transitions:
        - { condition: "always", next: assess-gap }
    - id: commit
      kind: act
      action: accelerate
      transitions:
        - { condition: "ego past conflict zone", next: clear }
        - { condition: "ego not yet clear", next: commit }
    - id: clear
      kind: act
      action: maintain-speed
      transitions:
        - { condition: "always", next: done }

effects:
  - "ego.position beyond intersection on the west approach"
  - "ego.heading aligned with target lane"

success_criteria:
  - "crossed_intersection == true"
  - "collision == false"

failure_criteria:
  - "collision == true"
  - "ego stalled in conflict zone for > 10 s"

safety_constraints:
  - "no overlap between ego bounding box and any other vehicle"
  - "ego stays within drivable area"

failure_modes:
  - "enters intersection without sufficient gap causing a collision"
  - "over-cautious freeze blocking the intersection"

termination: "ego past intersection OR aborted on imminent collision"
related_skills: [follow-keep-distance]

groundings:
  - backend: highway-env
    backend_version: ">=1.8"
    environment: intersection-v0
    observation: { type: Kinematics, features: [presence, x, y, vx, vy, cos_h, sin_h] }
    action: { type: DiscreteMetaAction }
    primitive_map:
      maintain-speed: IDLE
      accelerate: FASTER
      decelerate: SLOWER
```

- [ ] **Step 3: 运行校验驱动，确认该技能通过**

Run:
```bash
python3 world2skills/tools/validate_skills.py; echo "exit=$?"
```
Expected: 输出含 `[OK]   unprotected-left-turn`，`exit=0`。

> 若报 `skill.yaml: ...`，按错误路径修 `skill.yaml`；若报的是 schema 表达不了的约束，回到 Task 2 修 schema 并重跑本步。

- [ ] **Step 4: 人工核对该技能的跨文件/引用完整性**

逐项确认（这些是 schema 查不了的）：
- `name` 三处一致：文件夹 `unprotected-left-turn` == `SKILL.md` 的 `name` == `skill.yaml` 的 `name`。
- 图闭合：`entry: assess-gap` 存在；每个 `next`（decide/wait/commit/assess-gap/clear/done）指向存在的 step id 或字面量 `done`。
- `primitive_map` 覆盖 `interface.actions[].primitives`（maintain-speed/accelerate/decelerate 全在 map 里）；`execution` 中 `kind: act` 的 action 都在该原语集合内。
- `SKILL.md` 含四个必含小节。

- [ ] **Step 5: Commit**

```bash
git add world2skills/skills/unprotected-left-turn/
git commit -m "feat(world2skills): add unprotected-left-turn seed skill"
```

---

## Task 5: 种子技能 `lane-change-overtake`（横向）

**Files:**
- Create: `world2skills/skills/lane-change-overtake/SKILL.md`
- Create: `world2skills/skills/lane-change-overtake/skill.yaml`

- [ ] **Step 1: 写 `SKILL.md`**

写入 `world2skills/skills/lane-change-overtake/SKILL.md`：

```markdown
---
name: lane-change-overtake
description: Use when a slower vehicle ahead blocks progress on a multi-lane highway and the ego should change lanes to overtake when an adjacent lane gap is safe.
metadata:
  schema_version: "0.1"
  representation: skill.yaml
---

# Lane Change Overtake

## When to use
The ego is on a multi-lane highway behind a slower lead vehicle, an adjacent
lane is available, and overtaking would restore desired speed.

## Procedure
1. Detect that the lead vehicle is slower than the ego's target speed.
2. Check the adjacent lane for a safe gap (front and rear).
3. If a gap exists, change lanes toward it; otherwise keep following.
4. After the lane change, accelerate to pass the overtaken vehicle.
5. Resume cruising once clear.

## Reasoning cues
- Prefer the left lane for overtaking; a rear vehicle closing fast shrinks the
  usable gap.
- Abort the lane change if the target gap closes before commitment.

## Failure modes & recovery
- Changing into an occupied gap causes a side collision — re-check the rear gap
  immediately before committing.
- Oscillating between lanes wastes time — only change when the speed gain is
  worth it.
```

- [ ] **Step 2: 写 `skill.yaml`**

写入 `world2skills/skills/lane-change-overtake/skill.yaml`：

```yaml
schema_version: "0.1"
name: lane-change-overtake
version: "1.0"
domain: autonomous-driving
category: lateral
tags: [overtake, lane-change, highway]

description: >
  Change lanes on a multi-lane highway to overtake a slower lead vehicle
  when an adjacent-lane gap is safe.

entities:
  - { name: ego, role: controlled-vehicle }
  - { name: lead_vehicle, role: traffic }
  - { name: adjacent_traffic, role: traffic }

parameters:
  target_speed:
    type: float
    unit: m/s
    range: [0, 30]
    default: 25
    description: desired cruising speed
  min_lane_gap:
    type: float
    unit: m
    range: [10, 40]
    default: 20
    description: minimum clear distance in the target lane (front and rear)

interface:
  observations:
    - { name: surrounding_vehicles, type: kinematic, features: [presence, x, y, vx, vy, heading] }
    - { name: ego_state, type: kinematic }
  actions:
    - { name: lateral, type: discrete-meta, primitives: [change-lane-left, change-lane-right] }
    - { name: longitudinal, type: discrete-meta, primitives: [maintain-speed, accelerate, decelerate] }

preconditions:
  - "ego.lane has a slower lead vehicle"
  - "an adjacent lane exists"

execution:
  entry: check-lead
  steps:
    - id: check-lead
      kind: observe
      transitions:
        - { condition: "lead_speed < target_speed", next: check-gap }
        - { condition: "lead_speed >= target_speed", next: cruise }
    - id: check-gap
      kind: branch
      transitions:
        - { condition: "adjacent_gap >= min_lane_gap", next: change-lane }
        - { condition: "adjacent_gap < min_lane_gap", next: follow }
    - id: follow
      kind: act
      action: decelerate
      transitions:
        - { condition: "always", next: check-lead }
    - id: change-lane
      kind: act
      action: change-lane-left
      transitions:
        - { condition: "ego in target lane", next: pass }
        - { condition: "ego still changing", next: change-lane }
    - id: pass
      kind: act
      action: accelerate
      transitions:
        - { condition: "ego ahead of overtaken vehicle", next: cruise }
        - { condition: "ego not yet clear", next: pass }
    - id: cruise
      kind: act
      action: maintain-speed
      transitions:
        - { condition: "always", next: done }

effects:
  - "ego ahead of the previously blocking lead vehicle"
  - "ego speed at or near target_speed"

success_criteria:
  - "ego overtook the lead vehicle"
  - "collision == false"

failure_criteria:
  - "collision == true"
  - "ego forced back below target_speed behind the same vehicle"

safety_constraints:
  - "no overlap between ego bounding box and any other vehicle"
  - "lane change only into a lane that exists"

failure_modes:
  - "changing into an occupied adjacent gap causing a side collision"
  - "oscillating between lanes without net progress"

termination: "ego past the lead vehicle at target speed OR overtake aborted"
related_skills: [follow-keep-distance]

groundings:
  - backend: highway-env
    backend_version: ">=1.8"
    environment: highway-v0
    observation: { type: Kinematics, features: [presence, x, y, vx, vy, cos_h, sin_h] }
    action: { type: DiscreteMetaAction }
    primitive_map:
      maintain-speed: IDLE
      accelerate: FASTER
      decelerate: SLOWER
      change-lane-left: LANE_LEFT
      change-lane-right: LANE_RIGHT
```

- [ ] **Step 3: 校验 + 人工核对**

Run:
```bash
python3 world2skills/tools/validate_skills.py; echo "exit=$?"
```
Expected: 含 `[OK]   lane-change-overtake`。然后人工核对：`name` 三处一致；图闭合（entry `check-lead`；next 集合 {check-gap, cruise, change-lane, follow, check-lead, pass, done} 均落在存在的 id 或 `done`）；`primitive_map` 覆盖 interface 全部原语（含 change-lane-left/right）；四个必含小节齐全。

- [ ] **Step 4: Commit**

```bash
git add world2skills/skills/lane-change-overtake/
git commit -m "feat(world2skills): add lane-change-overtake seed skill"
```

---

## Task 6: 种子技能 `follow-keep-distance`（纵向）

**Files:**
- Create: `world2skills/skills/follow-keep-distance/SKILL.md`
- Create: `world2skills/skills/follow-keep-distance/skill.yaml`

- [ ] **Step 1: 写 `SKILL.md`**

写入 `world2skills/skills/follow-keep-distance/SKILL.md`：

```markdown
---
name: follow-keep-distance
description: Use when the ego follows a lead vehicle in the same lane and must keep a safe time-gap headway, matching the lead vehicle's speed without collision.
metadata:
  schema_version: "0.1"
  representation: skill.yaml
---

# Follow and Keep Distance

## When to use
The ego is behind a lead vehicle in the same lane with no intent or opportunity
to overtake, and must maintain a safe following distance.

## Procedure
1. Measure the time gap (headway) to the lead vehicle.
2. If the gap is smaller than the target, decelerate.
3. If the gap is larger than the target and below target speed, accelerate.
4. Otherwise hold speed to track the lead vehicle.

## Reasoning cues
- Headway in time, not distance, is what stays safe across speeds.
- React early and gently; late hard braking propagates instability upstream.

## Failure modes & recovery
- Tailgating leaves no room to brake — restore the target gap by decelerating.
- Over-braking on a transient gap dip wastes speed — filter brief fluctuations.
```

- [ ] **Step 2: 写 `skill.yaml`**

写入 `world2skills/skills/follow-keep-distance/skill.yaml`：

```yaml
schema_version: "0.1"
name: follow-keep-distance
version: "1.0"
domain: autonomous-driving
category: longitudinal
tags: [car-following, headway, cruise]

description: >
  Follow a lead vehicle in the same lane while keeping a safe time-gap
  headway and matching its speed.

entities:
  - { name: ego, role: controlled-vehicle }
  - { name: lead_vehicle, role: traffic }

parameters:
  target_speed:
    type: float
    unit: m/s
    range: [0, 30]
    default: 25
    description: desired speed when unobstructed
  target_headway:
    type: float
    unit: s
    range: [1, 3]
    default: 1.5
    description: desired time gap to the lead vehicle

interface:
  observations:
    - { name: lead_vehicle_state, type: kinematic, features: [presence, x, y, vx, vy] }
    - { name: ego_state, type: kinematic }
  actions:
    - { name: longitudinal, type: discrete-meta, primitives: [maintain-speed, accelerate, decelerate] }

preconditions:
  - "a lead vehicle is present in ego.lane"

execution:
  entry: measure-gap
  steps:
    - id: measure-gap
      kind: observe
      transitions:
        - { condition: "always", next: regulate }
    - id: regulate
      kind: branch
      transitions:
        - { condition: "headway < target_headway", next: brake }
        - { condition: "headway > target_headway and ego_speed < target_speed", next: speed-up }
        - { condition: "otherwise", next: hold }
    - id: brake
      kind: act
      action: decelerate
      transitions:
        - { condition: "always", next: measure-gap }
    - id: speed-up
      kind: act
      action: accelerate
      transitions:
        - { condition: "always", next: measure-gap }
    - id: hold
      kind: act
      action: maintain-speed
      transitions:
        - { condition: "lead vehicle still present", next: measure-gap }
        - { condition: "lead vehicle gone", next: done }

effects:
  - "ego headway within tolerance of target_headway"
  - "ego speed tracks lead vehicle speed"

success_criteria:
  - "headway stays >= safe minimum for the episode"
  - "collision == false"

failure_criteria:
  - "collision == true"
  - "headway below safe minimum sustained > 3 s"

safety_constraints:
  - "ego never overlaps the lead vehicle"
  - "headway never negative"

failure_modes:
  - "tailgating below safe headway with no braking margin"
  - "over-braking on a brief transient gap reduction"

termination: "lead vehicle leaves the lane OR episode ends"
related_skills: [lane-change-overtake]

groundings:
  - backend: highway-env
    backend_version: ">=1.8"
    environment: highway-v0
    observation: { type: Kinematics, features: [presence, x, y, vx, vy, cos_h, sin_h] }
    action: { type: DiscreteMetaAction }
    primitive_map:
      maintain-speed: IDLE
      accelerate: FASTER
      decelerate: SLOWER
```

- [ ] **Step 3: 校验 + 人工核对**

Run:
```bash
python3 world2skills/tools/validate_skills.py; echo "exit=$?"
```
Expected: 含 `[OK]   follow-keep-distance`。人工核对：`name` 三处一致；图闭合（entry `measure-gap`；next {regulate, brake, speed-up, hold, measure-gap, done} 均有效）；primitive_map 覆盖 interface 原语；四个必含小节齐全。

- [ ] **Step 4: Commit**

```bash
git add world2skills/skills/follow-keep-distance/
git commit -m "feat(world2skills): add follow-keep-distance seed skill"
```

---

## Task 7: 种子技能 `facilitate-highway-merge`（交互，ego 在主路让行）

> **领域要点：** highway-env `merge-v0` 里 ego 在**主路**行驶，为匝道来车让出空间并保持车速，**不是** ego 自己汇入。技能语义必须反映这一点。

**Files:**
- Create: `world2skills/skills/facilitate-highway-merge/SKILL.md`
- Create: `world2skills/skills/facilitate-highway-merge/skill.yaml`

- [ ] **Step 1: 写 `SKILL.md`**

写入 `world2skills/skills/facilitate-highway-merge/SKILL.md`：

```markdown
---
name: facilitate-highway-merge
description: Use when the ego is on the main highway approaching an on-ramp and should keep speed while making room for a vehicle merging in from the access ramp.
metadata:
  schema_version: "0.1"
  representation: skill.yaml
---

# Facilitate Highway Merge

## When to use
The ego drives on the main highway near an on-ramp where another vehicle is
merging in. The ego should maintain high speed and avoid collision while
creating room for the merging vehicle.

## Procedure
1. Detect a vehicle on the access ramp intending to merge.
2. Estimate whether the ego and the merging vehicle will conflict.
3. If a conflict is likely, create room: decelerate slightly or change to the
   left lane if it is clear.
4. If no conflict, maintain speed.
5. Resume normal cruising once the merging vehicle has settled in.

## Reasoning cues
- A small early speed adjustment opens a gap more smoothly than late braking.
- Only move left if that lane is clearly clear; do not trade one conflict for
  another.

## Failure modes & recovery
- Ignoring the merging vehicle forces a late conflict — adjust early.
- Braking too hard disrupts following traffic — prefer a gentle gap-opening.
```

- [ ] **Step 2: 写 `skill.yaml`**

写入 `world2skills/skills/facilitate-highway-merge/skill.yaml`：

```yaml
schema_version: "0.1"
name: facilitate-highway-merge
version: "1.0"
domain: autonomous-driving
category: interaction
tags: [merge, cooperative, on-ramp]

description: >
  On the main highway, keep speed while making room for a vehicle merging in
  from the access ramp, avoiding collision.

entities:
  - { name: ego, role: controlled-vehicle }
  - { name: merging_vehicle, role: traffic }
  - { name: highway, role: static-topology }

parameters:
  target_speed:
    type: float
    unit: m/s
    range: [0, 30]
    default: 25
    description: desired cruising speed on the main highway
  conflict_gap:
    type: float
    unit: s
    range: [1, 4]
    default: 2
    description: time-to-conflict threshold below which the ego makes room

interface:
  observations:
    - { name: merging_vehicle_state, type: kinematic, features: [presence, x, y, vx, vy, heading] }
    - { name: ego_state, type: kinematic }
  actions:
    - { name: lateral, type: discrete-meta, primitives: [change-lane-left] }
    - { name: longitudinal, type: discrete-meta, primitives: [maintain-speed, accelerate, decelerate] }

preconditions:
  - "ego on main highway"
  - "a vehicle is merging from the access ramp"

execution:
  entry: watch-ramp
  steps:
    - id: watch-ramp
      kind: observe
      transitions:
        - { condition: "always", next: assess-conflict }
    - id: assess-conflict
      kind: branch
      transitions:
        - { condition: "time_to_conflict < conflict_gap and left_lane_clear", next: move-left }
        - { condition: "time_to_conflict < conflict_gap and not left_lane_clear", next: open-gap }
        - { condition: "time_to_conflict >= conflict_gap", next: cruise }
    - id: move-left
      kind: act
      action: change-lane-left
      transitions:
        - { condition: "ego in left lane", next: cruise }
        - { condition: "ego still changing", next: move-left }
    - id: open-gap
      kind: act
      action: decelerate
      transitions:
        - { condition: "merging vehicle settled ahead", next: cruise }
        - { condition: "conflict still possible", next: watch-ramp }
    - id: cruise
      kind: act
      action: maintain-speed
      transitions:
        - { condition: "merging vehicle settled", next: done }
        - { condition: "still near ramp", next: watch-ramp }

effects:
  - "merging vehicle entered the highway without conflict"
  - "ego speed at or near target_speed"

success_criteria:
  - "merging vehicle merged successfully"
  - "collision == false"

failure_criteria:
  - "collision == true"
  - "ego forced to a full stop on the main highway"

safety_constraints:
  - "no overlap between ego bounding box and any other vehicle"
  - "lane change only into a clear existing lane"

failure_modes:
  - "ignoring the merging vehicle until a late unavoidable conflict"
  - "hard braking that disrupts following highway traffic"

termination: "merging vehicle settled on the highway OR ramp passed"
related_skills: [follow-keep-distance, lane-change-overtake]

groundings:
  - backend: highway-env
    backend_version: ">=1.8"
    environment: merge-v0
    observation: { type: Kinematics, features: [presence, x, y, vx, vy, cos_h, sin_h] }
    action: { type: DiscreteMetaAction }
    primitive_map:
      maintain-speed: IDLE
      accelerate: FASTER
      decelerate: SLOWER
      change-lane-left: LANE_LEFT
```

- [ ] **Step 3: 校验 + 人工核对**

Run:
```bash
python3 world2skills/tools/validate_skills.py; echo "exit=$?"
```
Expected: 含 `[OK]   facilitate-highway-merge`。人工核对：`name` 三处一致；图闭合（entry `watch-ramp`；next {assess-conflict, move-left, open-gap, cruise, watch-ramp, done} 均有效）；primitive_map 覆盖 interface 全部原语（含 change-lane-left）；四个必含小节齐全。

- [ ] **Step 4: Commit**

```bash
git add world2skills/skills/facilitate-highway-merge/
git commit -m "feat(world2skills): add facilitate-highway-merge seed skill"
```

---

## Task 8: 种子技能 `roundabout-navigate`（交互，连续让行）

**Files:**
- Create: `world2skills/skills/roundabout-navigate/SKILL.md`
- Create: `world2skills/skills/roundabout-navigate/skill.yaml`

- [ ] **Step 1: 写 `SKILL.md`**

写入 `world2skills/skills/roundabout-navigate/SKILL.md`：

```markdown
---
name: roundabout-navigate
description: Use when the ego must enter and traverse a roundabout, yielding to circulating traffic on entry and exiting at the intended offramp.
metadata:
  schema_version: "0.1"
  representation: skill.yaml
---

# Roundabout Navigate

## When to use
The ego approaches a roundabout and must yield to circulating traffic, enter
when a gap allows, circulate, and leave at the intended exit.

## Procedure
1. At the entry, observe circulating traffic and estimate the entry gap.
2. If the gap is insufficient, wait at the yield line.
3. When the gap is sufficient, enter the roundabout.
4. Circulate, keeping distance from the vehicle ahead.
5. Exit at the intended offramp.

## Reasoning cues
- Circulating traffic has priority; entry is the main conflict point.
- Keep a steady speed while circulating; hesitation inside the ring is riskier
  than a clean commit.

## Failure modes & recovery
- Entering on a marginal gap forces circulating traffic to brake — wait for a
  clear gap.
- Missing the intended exit means another loop — track the exit early.
```

- [ ] **Step 2: 写 `skill.yaml`**

写入 `world2skills/skills/roundabout-navigate/skill.yaml`：

```yaml
schema_version: "0.1"
name: roundabout-navigate
version: "1.0"
domain: autonomous-driving
category: interaction
tags: [roundabout, yield, merge]

description: >
  Enter and traverse a roundabout, yielding to circulating traffic on entry
  and leaving at the intended exit.

entities:
  - { name: ego, role: controlled-vehicle }
  - { name: circulating_vehicles, role: traffic }
  - { name: roundabout, role: static-topology }

parameters:
  target_speed:
    type: float
    unit: m/s
    range: [0, 12]
    default: 6
    description: circulating speed inside the roundabout
  entry_gap:
    type: float
    unit: s
    range: [2, 6]
    default: 3
    description: minimum time gap in circulating traffic to accept entry

interface:
  observations:
    - { name: circulating_vehicles, type: kinematic, features: [presence, x, y, vx, vy, heading] }
    - { name: ego_state, type: kinematic }
  actions:
    - { name: longitudinal, type: discrete-meta, primitives: [maintain-speed, accelerate, decelerate] }

preconditions:
  - "ego approaching a roundabout entry"
  - "ego.intent has an assigned exit"

execution:
  entry: approach
  steps:
    - id: approach
      kind: observe
      transitions:
        - { condition: "always", next: entry-decide }
    - id: entry-decide
      kind: branch
      transitions:
        - { condition: "circulating_gap < entry_gap", next: yield-wait }
        - { condition: "circulating_gap >= entry_gap", next: enter }
    - id: yield-wait
      kind: act
      action: decelerate
      transitions:
        - { condition: "always", next: approach }
    - id: enter
      kind: act
      action: accelerate
      transitions:
        - { condition: "ego inside ring", next: circulate }
        - { condition: "ego still entering", next: enter }
    - id: circulate
      kind: act
      action: maintain-speed
      transitions:
        - { condition: "approaching intended exit", next: exit }
        - { condition: "not yet at exit", next: circulate }
    - id: exit
      kind: act
      action: maintain-speed
      transitions:
        - { condition: "ego left the roundabout", next: done }
        - { condition: "still exiting", next: exit }

effects:
  - "ego left the roundabout at the intended exit"
  - "ego speed at or near target_speed"

success_criteria:
  - "ego took the intended exit"
  - "collision == false"

failure_criteria:
  - "collision == true"
  - "ego missed the intended exit"

safety_constraints:
  - "no overlap between ego bounding box and any other vehicle"
  - "ego yields to circulating traffic on entry"

failure_modes:
  - "entering on a marginal gap forcing circulating traffic to brake"
  - "missing the intended exit and looping again"

termination: "ego exits the roundabout OR episode ends"
related_skills: [unprotected-left-turn]

groundings:
  - backend: highway-env
    backend_version: ">=1.8"
    environment: roundabout-v0
    observation: { type: Kinematics, features: [presence, x, y, vx, vy, cos_h, sin_h] }
    action: { type: DiscreteMetaAction }
    primitive_map:
      maintain-speed: IDLE
      accelerate: FASTER
      decelerate: SLOWER
```

- [ ] **Step 3: 校验 + 人工核对**

Run:
```bash
python3 world2skills/tools/validate_skills.py; echo "exit=$?"
```
Expected: 含 `[OK]   roundabout-navigate`。人工核对：`name` 三处一致；图闭合（entry `approach`；next {entry-decide, yield-wait, enter, circulate, exit, approach, done} 均有效）；primitive_map 覆盖 interface 原语；四个必含小节齐全。

- [ ] **Step 4: Commit**

```bash
git add world2skills/skills/roundabout-navigate/
git commit -m "feat(world2skills): add roundabout-navigate seed skill"
```

---

## Task 9: 元模型规范文档

把 spec 第 4/5/6 节的规范性内容落成 `world2skills/` 内的自包含规范文档（面向后续新增技能的人/agent）。

**Files:**
- Create: `world2skills/docs/skill-representation-spec.md`

- [ ] **Step 1: 写规范文档**

写入 `world2skills/docs/skill-representation-spec.md`：

````markdown
# 自动驾驶 Skill 表示规范 (v0.1)

本规范定义 World2Skills 自动驾驶技能的统一表示形式。每个技能是一个文件夹，
含两个文件：`SKILL.md`（发现层）与 `skill.yaml`（表示层）。

## 1. 粒度

一个 Skill = 一个可复用、有名字的驾驶行为/机动，把「一类场景」映射到「一种
行动方式」（如变道超车、无保护左转、跟车保持车距）。单个元动作粒度过细；
「安全驾驶」粒度过粗——都不是一个 Skill。

## 2. 文件布局

```
skills/<skill-id>/
  SKILL.md      # 标准 Agent Skills：name / description / metadata + 正文
  skill.yaml    # 完整驾驶表示，JSON Schema 校验对象
  references/   # 可选
```

## 3. SKILL.md（发现层）

- frontmatter 只含标准字段 `name`、`description`、`metadata`。
- `description` 必须单行（现有正则加载器会把多行 description 截断为首行）。
- `metadata` 只放字符串索引：`schema_version`、`representation: skill.yaml`。
- 正文必含四个小节：`## When to use`、`## Procedure`、`## Reasoning cues`、
  `## Failure modes & recovery`；`## References` 可选。

## 4. skill.yaml（表示层）字段

| 字段 | 必填 | 说明 |
|------|------|------|
| `schema_version` | 是 | 表示格式版本，如 `"0.1"`。 |
| `name` | 是 | kebab-case，须与文件夹名及 SKILL.md 的 name 一致。 |
| `version` | 是 | 本技能自身版本。 |
| `domain` | 是 | 固定 `autonomous-driving`。 |
| `category` | 是 | `longitudinal` / `lateral` / `interaction`。 |
| `tags` | 否 | 检索标签。 |
| `description` | 是 | 技能用途（可多行）。 |
| `entities` | 是 | 推理对象，每项 `name` + `role`（`controlled-vehicle`/`traffic`/`static-topology`）。 |
| `parameters` | 否 | 可调参数，每项 `type`/`unit`/`range`/`default`/`description`。 |
| `interface` | 是 | 后端无关的抽象接口，见第 5 节。 |
| `preconditions` | 是 | 谓词字符串数组。 |
| `execution` | 是 | 闭合状态图，见第 6 节。 |
| `effects` | 是 | 成功后状态变化（谓词字符串）。 |
| `success_criteria` | 是 | 可观测成功条件。 |
| `failure_criteria` | 是 | 可观测失败条件（区别于叙述性 `failure_modes`）。 |
| `safety_constraints` | 是 | 任何时刻须成立的不变式。 |
| `failure_modes` | 是 | 叙述性失败模式，供人/LLM 阅读。 |
| `termination` | 是 | 终止条件描述。 |
| `related_skills` | 否 | 相关技能 id。 |
| `groundings` | 是 | 后端绑定，见第 7 节。 |

## 5. interface（抽象接口，后端无关）

- `observations`：每项 `name`、`type`（`kinematic`/`image`/`grid`/`track`）、
  可选 `features`。
- `actions`：每项 `name`、`type`（`discrete-meta`/`continuous`/`trajectory`）、
  `primitives`。**原语必须是抽象名**：`maintain-speed`、`accelerate`、
  `decelerate`、`change-lane-left`、`change-lane-right`。不得出现后端标识符。

## 6. execution（闭合状态图）

- `entry`：入口步骤 id。
- `steps`：每步 `id`、`kind`（`observe`/`act`/`branch`）、`action`（仅 `act` 步，
  取值为一个抽象原语）、`transitions`。
- `transitions` 每项 `condition`（谓词字符串）+ `next`（目标 step id 或字面量
  `done`）。
- **闭合要求**：`entry` 与每个 `next` 必须指向存在的 step id 或 `done`；
  `branch` 的条件应互斥且完备。

## 7. groundings（后端绑定）

每项含 `backend`、`backend_version`、`environment`、`observation`（后端观测
类型与特征）、`action`（后端动作类型）、`primitive_map`（抽象原语 → 后端动作）。
`primitive_map` 必须覆盖 `interface.actions[].primitives` 的全部抽象原语。

highway-env 参照映射：`maintain-speed→IDLE`、`accelerate→FASTER`、
`decelerate→SLOWER`、`change-lane-left→LANE_LEFT`、`change-lane-right→LANE_RIGHT`。

## 8. 条件表达约定

`preconditions`、`execution.steps[].transitions[].condition`、`effects`、
`success_criteria`、`failure_criteria`、`safety_constraints` 均为自由谓词
字符串（人与 LLM 可读）。本阶段不写求值器，不要求可执行。

## 9. 校验

- 结构自动校验：`SKILL.md` 过 skill-creator 的 `quick_validate.py`；
  `skill.yaml` 过 `schema/skill.schema.json`（用 `tools/validate_skills.py`）。
- 人工核对（schema 查不了的）：三处 `name` 一致、`execution` 图闭合、
  `primitive_map` 覆盖全部抽象原语、正文四个必含小节。
````

- [ ] **Step 2: Commit**

```bash
git add world2skills/docs/skill-representation-spec.md
git commit -m "docs(world2skills): add skill representation spec v0.1"
```

---

## Task 10: README（阶段说明 + 新增技能指南）

**Files:**
- Create: `world2skills/README.md`

- [ ] **Step 1: 写 README**

写入 `world2skills/README.md`：

````markdown
# World2Skills — 第一阶段：自动驾驶 Skill 构建与表示

本目录是 World2Skills 第一阶段的产物：一套面向自动驾驶的统一 Skill 表示形式，
以及 5 个通过校验的手写种子技能。以 highway-env 为参照后端。

> 范围：仅「构建与表示」。不含数据生成、规则生成、策略学习（后续阶段），
> 也不接入任何仿真器。

## 目录

```
world2skills/
  docs/skill-representation-spec.md   # 元模型规范 (v0.1)
  schema/skill.schema.json            # skill.yaml 的 JSON Schema
  tools/validate_skills.py            # 结构校验驱动
  skills/<skill-id>/SKILL.md          # 发现层
  skills/<skill-id>/skill.yaml        # 表示层
```

## 种子技能

| Skill | 场景 (highway-env) | category |
|-------|--------------------|----------|
| `lane-change-overtake` | highway-v0 | lateral |
| `follow-keep-distance` | highway-v0 | longitudinal |
| `unprotected-left-turn` | intersection-v0 | interaction |
| `facilitate-highway-merge` | merge-v0 | interaction |
| `roundabout-navigate` | roundabout-v0 | interaction |

## 校验

结构自动校验（用现成 `jsonschema` 库与 skill-creator 的 `quick_validate.py`）：

```bash
python3 world2skills/tools/validate_skills.py
```
退出码 0 表示全部通过。

## 如何新增一个技能

1. 建目录 `skills/<skill-id>/`。
2. 写 `SKILL.md`：frontmatter 只放 `name`（= 文件夹名）/单行 `description`/
   `metadata`；正文含四个必含小节。
3. 写 `skill.yaml`：按 `docs/skill-representation-spec.md` 填全字段；
   `interface`/`execution` 只用抽象原语，后端映射写在 `groundings[].primitive_map`。
4. 跑 `python3 world2skills/tools/validate_skills.py` 直到该技能 `[OK]`。
5. 人工核对（schema 查不了的）：
   - `name` 在文件夹名 / `SKILL.md` / `skill.yaml` 三处一致；
   - `execution` 图闭合：`entry` 与每个 `transitions[].next` 指向存在的 step id
     或字面量 `done`；
   - `primitive_map` 覆盖 `interface.actions[].primitives` 全部抽象原语，
     且 `execution` 中 `kind: act` 步骤的 `action` 都在该集合内；
   - 正文四个必含小节齐全。
````

- [ ] **Step 2: Commit**

```bash
git add world2skills/README.md
git commit -m "docs(world2skills): add stage-1 README and how-to"
```

---

## Task 11: 全量校验与最终核对

**Files:**
- 无新增；仅运行校验与人工核对。

- [ ] **Step 1: 全量结构校验**

Run:
```bash
python3 world2skills/tools/validate_skills.py; echo "exit=$?"
```
Expected: 5 行全为 `[OK]`，结尾 `5 skills checked, 0 error(s).`，`exit=0`。

- [ ] **Step 2: 确认 5 个技能齐全、覆盖三类**

Run:
```bash
ls world2skills/skills
```
Expected: `facilitate-highway-merge  follow-keep-distance  lane-change-overtake  roundabout-navigate  unprotected-left-turn`（横向 1 / 纵向 1 / 交互 3）。

- [ ] **Step 3: 逐技能人工核对（schema 查不了的四项）**

对 5 个技能各确认：`name` 三处一致；`execution` 图闭合（entry + 所有 next 落点有效）；`primitive_map` 覆盖 interface 全部抽象原语且 act 步 action 在集合内；`SKILL.md` 含四个必含小节。全部通过。

- [ ] **Step 4: 确认未触碰现有代码**

Run:
```bash
git diff --name-only HEAD~10 HEAD | grep -v '^world2skills/\|^docs/superpowers/' || echo "only world2skills + specs touched"
```
Expected: 输出 `only world2skills + specs touched`（本阶段只新增 `world2skills/` 与规范/计划文档，未改 `spreadsheet_agent/`、`skill_evolver/`、`src/`）。

- [ ] **Step 5: 最终 commit（若前面均已逐任务提交，此步可为空）**

```bash
git add -A world2skills/
git commit -m "chore(world2skills): stage-1 skill representation complete" --allow-empty
```

---

## Self-Review Notes

- **Spec coverage：** 规范第 2 节（交付物：①规范→Task 9、②schema→Task 1–2、③5 示例→Task 4–8）；第 6 节字段集全部落在 schema（Task 1–2）与示例（Task 4）；第 7 节 5 技能→Task 4–8；第 8 节目录→Task 1 + 各技能；第 9 节验收（自动校验→Task 3/11，人工核对→各技能 Step + Task 11）。均有对应任务。
- **抽象原语一致性：** `maintain-speed/accelerate/decelerate/change-lane-left/change-lane-right` 这一枚举在 schema（Task 2 的 interface.actions.primitives 与 execution.action）、5 个 skill.yaml、规范文档、README 中一致。
- **primitive_map 映射一致性：** 五处 groundings 的映射均为 `maintain-speed→IDLE / accelerate→FASTER / decelerate→SLOWER`（含横向的 `change-lane-left→LANE_LEFT / change-lane-right→LANE_RIGHT`），与领域备注一致。
- **校验命令一致性：** 全程用 `python3 world2skills/tools/validate_skills.py`，退出码语义（0 通过 / 非 0 失败）在 Task 3、4–8、11 一致。
- **无占位符：** 每个 skill.yaml、SKILL.md、schema 片段均为完整可写入内容，无 TODO/TBD。
