# World2Skills Phase 1: Skill Construction and Representation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立一套兼容 Agent Skills 目录规范、面向具身智能且可被程序验证、组合和检索的统一 Skill 表示，并交付首批跨领域 Skill 样例、构建工具和评测基线。

**Architecture:** 采用双层表示。每个 Skill 包使用 `SKILL.md` 提供 Agent 发现、触发描述和渐进式工作流说明；使用 `skill.yaml` 作为机器可读的唯一真源，表达类型化接口、状态条件、执行步骤、观测、恢复、安全、依赖和平台绑定。Python 核心库负责加载、规范化、验证、组合和注册，生成确定性的 `skill.lock.json` 与前端 JSON，后续阶段只依赖这些稳定产物。

**Tech Stack:** Python 3.11+, Pydantic 2, PyYAML, JSON Schema Draft 2020-12, Typer, pytest, Agent Skills `SKILL.md`, React + TypeScript 演示层。

---

## 1. Scope and Decisions

### 1.1 第一阶段包含

1. 定义 Skill 的边界、粒度、分类和版本策略。
2. 定义统一机器表示 `skill.yaml` 和对应 JSON Schema。
3. 定义 Agent-facing `SKILL.md` 编写约定。
4. 实现 Skill 脚手架、加载、验证、规范化、注册和组合检查。
5. 将现有 6 个演示 Skill 迁移到正式格式。
6. 增加 2 个跨领域挑战样例，验证表示不只适用于桌面操作。
7. 建立结构、语义、组合、触发和可用性评测。
8. 生成供现有 React 演示使用的只读 JSON。

### 1.2 第一阶段明确不包含

1. 不生成视频、图像、轨迹或标签数据。
2. 不连接 Cosmos、Omniverse、Marble 或真实自动驾驶数据管线。
3. 不生成正式行为规则、行为树或策略网络。
4. 不训练或微调机器人策略模型。
5. 不接入真实 ROS 2 控制器；只定义绑定契约并提供 mock binding。
6. 不自动演化生产 Skill；`Trace2Skill` 只作为后续可接入的实验资产。

### 1.3 推荐方案

采用“Agent Skills 包 + 形式化具身 IR”的混合方案。

| 方案 | 优点 | 缺点 | 结论 |
|---|---|---|---|
| 仅 `SKILL.md` | 实现最快，适合 LLM 阅读 | 条件、单位、状态和依赖难以可靠解析 | 不采用 |
| 仅形式化 DSL/PDDL | 易验证和规划 | Agent 触发、知识解释和扩展成本高 | 不采用 |
| `SKILL.md` + `skill.yaml` | 同时支持 Agent 与程序，便于渐进披露和版本化 | 需要处理双层一致性 | 采用 |

一致性原则：`skill.yaml` 是结构化事实的唯一真源；`SKILL.md` 只描述何时使用、如何读取清单、如何执行工作流和何时加载 references，不复制完整字段。

---

## 2. Canonical Skill Package

每个 Skill 的目标目录如下：

```text
skills/<skill-name>/
├── SKILL.md
├── skill.yaml
├── references/
│   ├── semantics.md
│   └── examples.md
├── scripts/                 # 仅在存在确定性工具时创建
├── bindings/                # 平台/机器人绑定，不进入核心语义
│   └── mock.yaml
└── tests/
    └── scenarios.yaml
```

Agent Skills 兼容层：

```yaml
---
name: pick-up-object
description: Execute or reason about grasping and lifting a reachable object. Use when a task requires selecting a grasp, checking grasp preconditions, lifting an object, diagnosing grasp failure, or composing a pick step into a larger manipulation task.
---
```

`SKILL.md` 正文只保留以下内容：

1. 读取 `skill.yaml` 的顺序。
2. 执行前必须检查的验证级别。
3. 何时读取 `references/semantics.md` 和 `references/examples.md`。
4. 如何选择 binding。
5. 如何报告无法满足的前置条件、约束或依赖。

### 2.1 `skill.yaml` 顶层结构

```yaml
schema_version: 0.1.0
id: manipulation.pick-up-object
name: Pick Up Object
version: 0.1.0
status: draft
description: Grasp and lift a target object from a supporting surface.

taxonomy:
  domain: embodied-robotics
  category: manipulation
  level: primitive
  tags: [grasping, lifting]

provenance:
  authors: [world2skills]
  source_type: human-authored
  source_refs: []
  license: Apache-2.0

intent:
  goal: target object is securely held above its original support
  aliases: [grasp-object, lift-object]
  non_goals: [tool-use, two-arm handover]

interface:
  inputs: []
  outputs: []
  parameters: []

state:
  preconditions: []
  invariants: []
  effects: []

execution:
  mode: sequence
  steps: []

observation:
  success_conditions: []
  failure_conditions: []

recovery:
  strategies: []

safety:
  constraints: []
  stop_conditions: []

requirements:
  capabilities: []
  resources: []
  dependencies: []

grounding:
  abstract_api: pick_up_object
  bindings: []

downstream_contracts:
  data_generation: {}
  rule_generation: {}

evaluation:
  scenario_file: tests/scenarios.yaml
```

### 2.2 必须类型化的字段

#### Entity and value types

每个输入、输出和参数必须声明：

- `name`
- `kind`: `entity`、`scalar`、`vector`、`pose`、`enum`、`boolean`
- `type`: 如 `object`, `container`, `robot.gripper`, `geometry.pose3d`
- `required`
- `description`
- `unit`: 标量和向量需要时填写，首版限制为 SI 常用单位白名单
- `frame`: 位姿和空间向量必须声明坐标系或 `runtime`
- `constraints`: 范围、枚举、正则或实体属性约束

#### Predicate expression

禁止把前置条件和效果保存为不可解析的自然语言字符串。首版使用小型逻辑 AST：

```yaml
preconditions:
  all:
    - predicate: visible
      args: [$target]
    - predicate: reachable
      args: [$target, $robot]
    - predicate: empty
      args: [$gripper]
```

允许节点：

- `predicate`
- `all`
- `any`
- `not`
- `compare`

首版不实现量词、时序逻辑和概率逻辑；需要的复杂语义放入 `references/semantics.md`，同时登记为 schema extension 候选。

#### Effects

效果使用显式操作，避免把赋值和事实混在字符串中：

```yaml
effects:
  - operation: set
    predicate: held_by
    args: [$target, $gripper]
    value: true
  - operation: set
    predicate: on_support
    args: [$target]
    value: false
```

#### Execution steps

每个步骤必须有稳定 `id`，并选择以下一种：

- `invoke`: 调用另一个 Skill。
- `action`: 调用抽象动作 API。
- `check`: 检查状态或观测。
- `branch`: 根据可解析条件选择分支。

首版控制流只支持 `sequence`、`selector` 和有界 `retry`，不支持任意循环。

#### Observation and recovery

失败条件必须有稳定 `code`，恢复策略通过该 code 关联，禁止按数组下标配对：

```yaml
failure_conditions:
  - code: grasp.slip
    when:
      predicate: object_slipping
      args: [$target]

strategies:
  - handles: [grasp.slip]
    max_attempts: 2
    steps:
      - action: reduce_lift_speed
      - action: regrasp
```

#### Safety

每个可执行 Skill 至少包含：

- 力、速度、加速度或空间边界中的适用项。
- 紧急停止条件。
- 禁止接触区域或实体类别。
- 运行时 binding 必须提供的安全参数。

#### Downstream contracts

第一阶段只定义声明式接口：

- `data_generation`: 可变因素、固定因素、所需标签、成功/失败覆盖要求。
- `rule_generation`: 可暴露的谓词、恢复映射、可解释性要求。

这些字段用于验证第二、三阶段能否消费 Skill，不在第一阶段执行生成。

---

## 3. Repository Target Structure

```text
World2Skills/
├── .gitignore
├── pyproject.toml
├── schemas/
│   └── skill.schema.json
├── src/world2skills/
│   ├── __init__.py
│   ├── cli.py
│   ├── compiler.py
│   ├── registry.py
│   ├── models/
│   │   ├── common.py
│   │   ├── expressions.py
│   │   ├── execution.py
│   │   └── skill.py
│   ├── io/
│   │   ├── loader.py
│   │   └── writer.py
│   └── validation/
│       ├── package.py
│       ├── schema.py
│       ├── semantic.py
│       ├── composition.py
│       └── report.py
├── skills/
│   ├── detect-object/
│   ├── move-to-object/
│   ├── pick-up-object/
│   ├── place-object/
│   ├── open-drawer/
│   ├── put-object-into-container/
│   ├── navigate-to-waypoint/
│   └── safe-lane-change/
├── .agents/skills/build-embodied-skill/
│   ├── SKILL.md
│   ├── agents/openai.yaml
│   ├── references/
│   │   ├── schema-guide.md
│   │   ├── granularity-guide.md
│   │   ├── predicate-catalog.md
│   │   └── review-checklist.md
│   └── scripts/
│       └── scaffold_skill.py
├── tests/
│   ├── fixtures/
│   ├── unit/
│   ├── integration/
│   └── evals/
├── scripts/
│   └── export_demo_skills.py
├── src/generated/
│   └── skills.json
└── docs/phase1/
    ├── requirements.md
    ├── skill-representation.md
    ├── adr/
    ├── authoring-guide.md
    ├── evaluation-protocol.md
    └── phase1-report.md
```

`Trace2Skill/` 保持独立，不在第一阶段修改其已有实验代码。需要复用时只通过导出的 Skill 包或后续 adapter 接入。

---

## 4. Validation Levels

CLI 命令统一为：

```bash
w2s-skill validate skills/pick-up-object --level semantic
w2s-skill compile skills/pick-up-object
w2s-skill registry build skills --output build/registry.json
w2s-skill export-demo skills --output src/generated/skills.json
```

验证分级：

| Level | 检查内容 | 阻断发布 |
|---|---|---|
| L0 Package | 目录名、`SKILL.md`、frontmatter、引用文件存在 | 是 |
| L1 Schema | YAML、Pydantic、JSON Schema、枚举和必填字段 | 是 |
| L2 Semantic | ID/版本、变量引用、单位、坐标系、谓词签名、失败恢复映射 | 是 |
| L3 Composition | 依赖解析、输入输出兼容、步骤可达、循环依赖 | 复合 Skill 必须 |
| L4 Grounding | binding 参数覆盖、目标平台能力声明 | 发布到具体平台时必须 |

错误输出必须包含：

```text
<severity> <error-code> <file>:<yaml-path> <message> <suggested-fix>
```

例如：

```text
ERROR W2S-E203 skills/pick-up-object/skill.yaml:state.preconditions.all[1].args[0] unknown variable "$object"; use "$target"
```

---

## 5. Implementation Tasks

### Task 0: Establish the World2Skills Repository Boundary

The current workspace root is not a Git repository, while `Trace2Skill/` is an independent nested repository with local modifications. Phase-1 work must not be committed into or mixed with that nested repository.

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Initialize the root repository**

```bash
git init -b main
```

Expected: `/data/yifan/zrx/World2Skills/.git/` is created.

- [ ] **Step 2: Ignore generated files and the independent research checkout**

Create `.gitignore` with:

```gitignore
node_modules/
dist/
build/
.coverage
.pytest_cache/
.ruff_cache/
__pycache__/
*.py[cod]
.venv/
Trace2Skill/
```

`Trace2Skill/` remains independently versioned and its existing modified files are not touched.

- [ ] **Step 3: Record the existing demo baseline**

```bash
npm test
npm run build
git add .gitignore README.md package.json package-lock.json index.html vite.config.ts tsconfig.json src docs scripts
git commit -m "chore: establish world2skills repository"
```

Expected: the Vitest suite and Vite build pass before Phase-1 changes begin.

### Task 1: Freeze Phase-1 Requirements and ADRs

**Files:**
- Create: `docs/phase1/requirements.md`
- Create: `docs/phase1/skill-representation.md`
- Create: `docs/phase1/adr/0001-dual-layer-representation.md`
- Create: `docs/phase1/adr/0002-yaml-source-of-truth.md`
- Create: `docs/phase1/adr/0003-bounded-expression-language.md`

- [ ] **Step 1: Write measurable requirements**

Record the following requirements exactly:

```text
R1 Every released Skill is a valid Agent Skills package.
R2 Every released Skill contains one valid skill.yaml.
R3 Structured facts are stored only in skill.yaml.
R4 A Skill can be loaded and normalized without importing frontend code.
R5 Primitive and composite Skills use the same top-level schema.
R6 All variable, dependency, predicate, unit, frame, failure and recovery references are validated.
R7 The normalized output is deterministic.
R8 Data-generation and rule-generation contracts are declared but not executed.
R9 At least six manipulation, one navigation and one driving Skill pass release validation.
R10 Existing React demo consumes generated data rather than hand-maintained Skill objects.
```

- [ ] **Step 2: Define non-functional requirements**

Add:

```text
NFR1 A full validation of 100 local Skill packages completes in under 5 seconds on a laptop.
NFR2 The schema is versioned independently from individual Skill versions.
NFR3 Error messages identify the file and YAML path.
NFR4 No required semantic field depends on unrestricted natural-language parsing.
NFR5 The core loader and validator have at least 90% branch coverage.
```

- [ ] **Step 3: Document the three architecture decisions**

Each ADR must include context, decision, alternatives, consequences and reversal conditions.

- [ ] **Step 4: Review scope**

Verify that Cosmos, video generation, behavior-rule generation, policy learning and robot deployment appear only in the out-of-scope section.

- [ ] **Step 5: Commit**

```bash
git add docs/phase1
git commit -m "docs: define phase one skill representation"
```

### Task 2: Create the Python Core and Test Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/world2skills/__init__.py`
- Create: `src/world2skills/cli.py`
- Create: `tests/unit/test_cli.py`

- [ ] **Step 1: Add failing CLI smoke test**

```python
from typer.testing import CliRunner

from world2skills.cli import app


def test_cli_reports_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "w2s-skill 0.1.0"
```

- [ ] **Step 2: Run the test and verify failure**

Run:

```bash
python -m pytest tests/unit/test_cli.py -q
```

Expected: collection fails because `world2skills` does not exist.

- [ ] **Step 3: Configure packaging**

Use a `src/` layout and add runtime dependencies for Pydantic, PyYAML and Typer; add pytest, pytest-cov and Ruff as development dependencies.

- [ ] **Step 4: Implement the minimal CLI**

```python
import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def version() -> None:
    typer.echo("w2s-skill 0.1.0")
```

- [ ] **Step 5: Verify**

```bash
python -m pytest tests/unit/test_cli.py -q
python -m world2skills.cli version
```

Expected: one passing test and `w2s-skill 0.1.0`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/world2skills tests/unit/test_cli.py
git commit -m "build: scaffold world2skills core"
```

### Task 3: Implement the Typed Skill Model

**Files:**
- Create: `src/world2skills/models/common.py`
- Create: `src/world2skills/models/expressions.py`
- Create: `src/world2skills/models/execution.py`
- Create: `src/world2skills/models/skill.py`
- Create: `tests/fixtures/minimal_valid_skill.yaml`
- Create: `tests/unit/test_models.py`

- [ ] **Step 1: Write model acceptance tests**

Tests must assert:

```text
1. A complete primitive Skill validates.
2. Unknown fields fail validation.
3. A pose without frame fails validation.
4. A scalar constraint with min > max fails validation.
5. An execution step cannot define both invoke and action.
6. A failure condition requires a stable code.
7. A recovery strategy requires handles and max_attempts >= 1.
8. A released executable Skill requires safety stop conditions.
```

- [ ] **Step 2: Run tests and verify failure**

```bash
python -m pytest tests/unit/test_models.py -q
```

Expected: import failure for missing model modules.

- [ ] **Step 3: Implement strict shared models**

All Pydantic models must set:

```python
model_config = ConfigDict(extra="forbid")
```

Define reusable models for semantic versions, constraints, typed values, variables, predicate signatures and provenance.

- [ ] **Step 4: Implement the expression union**

Use discriminated models for:

```text
PredicateExpr
AllExpr
AnyExpr
NotExpr
CompareExpr
```

Do not add arbitrary Python expressions or template evaluation.

- [ ] **Step 5: Implement execution and recovery models**

Use explicit step variants and bounded retries.

- [ ] **Step 6: Implement the top-level `SkillManifest`**

Require every top-level section listed in Section 2.1. Allow empty downstream contracts but not absent contracts.

- [ ] **Step 7: Verify and measure coverage**

```bash
python -m pytest tests/unit/test_models.py --cov=world2skills.models --cov-report=term-missing
```

Expected: all tests pass and model branch coverage is at least 90%.

- [ ] **Step 8: Commit**

```bash
git add src/world2skills/models tests/fixtures/minimal_valid_skill.yaml tests/unit/test_models.py
git commit -m "feat: define canonical skill manifest"
```

### Task 4: Add Loader, Normalizer and JSON Schema Export

**Files:**
- Create: `src/world2skills/io/loader.py`
- Create: `src/world2skills/io/writer.py`
- Create: `src/world2skills/compiler.py`
- Create: `schemas/skill.schema.json`
- Create: `tests/unit/test_loader.py`
- Create: `tests/unit/test_compiler.py`

- [ ] **Step 1: Write failing loader tests**

Cover:

```text
1. YAML loads into SkillManifest.
2. Duplicate YAML keys are rejected.
3. Invalid UTF-8 is rejected with a file error.
4. Empty files are rejected.
5. Validation errors preserve YAML paths.
```

- [ ] **Step 2: Write failing compiler tests**

Assert that:

```text
1. Equivalent YAML key ordering produces identical lock JSON.
2. Lock JSON contains schema digest and source digest.
3. Lists whose order is semantic retain order.
4. Sets such as tags are normalized deterministically.
5. JSON Schema generation is reproducible.
```

- [ ] **Step 3: Implement safe loading**

Use `yaml.SafeLoader` with duplicate-key detection. Never instantiate Python objects from YAML tags.

- [ ] **Step 4: Implement deterministic normalization**

Serialize with sorted object keys, stable list policy, UTF-8 and a trailing newline. Calculate SHA-256 digests for source and schema.

- [ ] **Step 5: Export the schema**

Add:

```bash
w2s-skill schema export --output schemas/skill.schema.json
```

- [ ] **Step 6: Verify**

```bash
python -m pytest tests/unit/test_loader.py tests/unit/test_compiler.py -q
w2s-skill schema export --output schemas/skill.schema.json
git diff --exit-code schemas/skill.schema.json
```

Expected: tests pass and a second schema export creates no diff.

- [ ] **Step 7: Commit**

```bash
git add src/world2skills/io src/world2skills/compiler.py schemas tests/unit
git commit -m "feat: load and compile skill manifests"
```

### Task 5: Implement Package and Semantic Validation

**Files:**
- Create: `src/world2skills/validation/package.py`
- Create: `src/world2skills/validation/schema.py`
- Create: `src/world2skills/validation/semantic.py`
- Create: `src/world2skills/validation/report.py`
- Create: `tests/fixtures/invalid_skills/`
- Create: `tests/unit/test_package_validation.py`
- Create: `tests/unit/test_semantic_validation.py`

- [ ] **Step 1: Build one fixture per error class**

Create invalid packages for:

```text
W2S-E001 missing SKILL.md
W2S-E002 folder and frontmatter name mismatch
W2S-E101 invalid schema version
W2S-E201 unknown variable reference
W2S-E202 unknown predicate
W2S-E203 predicate arity mismatch
W2S-E204 unsupported unit
W2S-E205 missing coordinate frame
W2S-E206 recovery references unknown failure code
W2S-E207 duplicate step id
W2S-E208 effect contradicts invariant
W2S-E209 released Skill lacks stop condition
```

- [ ] **Step 2: Write failing validation tests**

Each fixture must produce exactly one primary expected error code and a stable YAML path.

- [ ] **Step 3: Implement Agent Skills package checks**

Check:

1. Folder name is lowercase hyphen-case.
2. `SKILL.md` exists.
3. YAML frontmatter contains `name` and `description`.
4. Frontmatter name equals folder name.
5. Description states capability and trigger context.
6. Linked local resources exist.
7. `SKILL.md` stays below 500 lines.

- [ ] **Step 4: Implement semantic symbol tables**

Build symbol tables for variables, predicates, failures, steps and dependencies before validating references.

- [ ] **Step 5: Implement machine-readable reports**

Support text and JSON:

```bash
w2s-skill validate skills/pick-up-object --format text
w2s-skill validate skills/pick-up-object --format json
```

- [ ] **Step 6: Verify**

```bash
python -m pytest tests/unit/test_package_validation.py tests/unit/test_semantic_validation.py -q
```

Expected: all validation error fixtures pass.

- [ ] **Step 7: Commit**

```bash
git add src/world2skills/validation tests/fixtures/invalid_skills tests/unit
git commit -m "feat: validate skill packages and semantics"
```

### Task 6: Implement Registry and Composition Validation

**Files:**
- Create: `src/world2skills/registry.py`
- Create: `src/world2skills/validation/composition.py`
- Create: `tests/fixtures/registry/`
- Create: `tests/unit/test_registry.py`
- Create: `tests/unit/test_composition.py`

- [ ] **Step 1: Write failing registry tests**

Cover duplicate IDs, duplicate versions, unresolved dependencies, incompatible version ranges and deterministic index ordering.

- [ ] **Step 2: Write failing composition tests**

Cover:

```text
1. Step input is supplied by parent input or prior output.
2. Invoked Skill preconditions are satisfied or explicitly deferred to runtime.
3. Invoked Skill effects can satisfy later steps.
4. Cyclic dependencies fail.
5. Retry count is bounded.
6. Primitive and composite Skills share one manifest type.
```

- [ ] **Step 3: Implement the registry**

The registry index must contain:

```text
id, version, package_path, description, taxonomy, inputs, outputs,
capabilities, dependencies, source_digest, schema_digest
```

- [ ] **Step 4: Implement conservative composition**

The validator may return `valid`, `invalid` or `runtime-dependent`. It must not claim a composition is valid when a precondition cannot be proven.

- [ ] **Step 5: Verify**

```bash
python -m pytest tests/unit/test_registry.py tests/unit/test_composition.py -q
```

- [ ] **Step 6: Commit**

```bash
git add src/world2skills/registry.py src/world2skills/validation/composition.py tests
git commit -m "feat: index and compose skill packages"
```

### Task 7: Build the `build-embodied-skill` Authoring Skill

**Files:**
- Create: `.agents/skills/build-embodied-skill/SKILL.md`
- Create: `.agents/skills/build-embodied-skill/agents/openai.yaml`
- Create: `.agents/skills/build-embodied-skill/references/schema-guide.md`
- Create: `.agents/skills/build-embodied-skill/references/granularity-guide.md`
- Create: `.agents/skills/build-embodied-skill/references/predicate-catalog.md`
- Create: `.agents/skills/build-embodied-skill/references/review-checklist.md`
- Create: `.agents/skills/build-embodied-skill/scripts/scaffold_skill.py`
- Create: `tests/integration/test_scaffold_skill.py`

- [ ] **Step 1: Define concrete trigger examples**

Positive examples:

```text
Create a reusable Skill for opening a drawer.
Turn this robot demonstration into a Skill package.
Validate whether pick-up-object is represented correctly.
Add a safe-lane-change Skill to the registry.
Split this long manipulation workflow into primitive and composite Skills.
```

Hard negatives:

```text
Run the pick-up-object Skill on the real robot.
Generate 10,000 grasping videos.
Train a policy for drawer opening.
Explain what a PDDL precondition is.
Fix a React layout bug in the demo.
```

- [ ] **Step 2: Write the `SKILL.md` workflow**

The workflow must require:

1. Collect three concrete use cases and two non-goals.
2. Select primitive or composite granularity.
3. Reuse predicates before adding new predicates.
4. Scaffold the package.
5. Fill `skill.yaml` before prose references.
6. Run L0-L3 validation.
7. Add positive, negative and recovery scenarios.
8. Compile and register the Skill.
9. Report unresolved runtime-dependent assumptions.

- [ ] **Step 3: Keep progressive disclosure**

Keep `SKILL.md` below 250 lines. Put schema field details, taxonomy, predicate catalog and review criteria in separate references.

- [ ] **Step 4: Implement deterministic scaffolding**

Command:

```bash
python .agents/skills/build-embodied-skill/scripts/scaffold_skill.py \
  pick-up-object \
  --category manipulation \
  --level primitive \
  --output skills
```

The script must create only:

```text
SKILL.md
skill.yaml
references/semantics.md
references/examples.md
tests/scenarios.yaml
```

It must refuse to overwrite an existing package unless `--force` is supplied.

- [ ] **Step 5: Validate the Agent Skill**

Run the official Agent Skills validator when available and the project validator:

```bash
skills-ref validate .agents/skills/build-embodied-skill
w2s-skill validate .agents/skills/build-embodied-skill --level package
```

Expected: both checks pass.

- [ ] **Step 6: Run integration tests**

```bash
python -m pytest tests/integration/test_scaffold_skill.py -q
```

- [ ] **Step 7: Commit**

```bash
git add .agents/skills/build-embodied-skill tests/integration/test_scaffold_skill.py
git commit -m "feat: add embodied skill authoring skill"
```

### Task 8: Migrate the Pilot Skill Library

**Files:**
- Create: `skills/detect-object/`
- Create: `skills/move-to-object/`
- Create: `skills/pick-up-object/`
- Create: `skills/place-object/`
- Create: `skills/open-drawer/`
- Create: `skills/put-object-into-container/`
- Create: `skills/navigate-to-waypoint/`
- Create: `skills/safe-lane-change/`
- Create: `skills/open-drawer-and-store-object/`
- Create: `tests/integration/test_released_skills.py`

- [ ] **Step 1: Create a shared predicate catalog**

Start with stable signatures for:

```text
visible(entity)
reachable(entity, agent)
empty(container_or_gripper)
held_by(entity, gripper)
inside(entity, container)
open(container)
path_clear(agent, target)
localized(agent)
at_pose(entity, pose)
lane_clear(vehicle, lane)
safe_gap(vehicle, lane)
speed_within_limit(vehicle)
```

- [ ] **Step 2: Migrate the six existing demo Skills**

Map every current field in `src/data/skills.ts` to a typed field. Any item that cannot be represented must be recorded in `docs/phase1/schema-gap-log.md` with:

```text
source field
source value
reason it does not fit
proposed schema change
decision: reject, defer or extend
```

- [ ] **Step 3: Add navigation and driving challenge Skills**

`navigate-to-waypoint` validates spatial frames, path conditions and navigation capability requirements.

`safe-lane-change` validates dynamic entities, continuous parameters, safety constraints, runtime-dependent observations and a bounded abort recovery.

- [ ] **Step 4: Add one composite Skill**

`open-drawer-and-store-object` must invoke:

```text
open-drawer
detect-object
move-to-object
pick-up-object
put-object-into-container
```

- [ ] **Step 5: Add scenarios**

Each Skill needs at least:

1. Two nominal success scenarios.
2. Two precondition failures.
3. Two execution failures with recovery.
4. One safety stop scenario.
5. One hard negative scenario where the Skill must not be selected.

- [ ] **Step 6: Validate the full library**

```bash
w2s-skill validate skills --level composition
w2s-skill registry build skills --output build/registry.json
python -m pytest tests/integration/test_released_skills.py -q
```

Expected: all nine packages pass; no unresolved ID, predicate or dependency errors.

- [ ] **Step 7: Commit**

```bash
git add skills docs/phase1/schema-gap-log.md tests/integration/test_released_skills.py
git commit -m "feat: release initial embodied skill library"
```

### Task 9: Replace Hand-Written Frontend Skill Data

**Files:**
- Create: `scripts/export_demo_skills.py`
- Create: `src/generated/skills.json`
- Modify: `src/types.ts`
- Modify: `src/data/skills.ts`
- Modify: `src/components/SkillSchema.tsx`
- Modify: `src/components/SkillLibrary.tsx`
- Modify: `src/utils/generation.test.ts`
- Modify: `src/utils/simulation.test.ts`
- Create: `tests/integration/test_demo_export.py`

- [ ] **Step 1: Write export tests**

Assert that the exporter:

```text
1. Reads compiled Skill manifests only.
2. Produces deterministic JSON.
3. Preserves the six current demo IDs through an explicit legacy-ID map.
4. Includes validation status and schema version.
5. Does not execute downstream data or rule generation.
```

- [ ] **Step 2: Implement the exporter**

Generate a frontend projection rather than exposing the full manifest:

```json
{
  "id": "pick_up_object",
  "canonical_id": "manipulation.pick-up-object",
  "name": "Pick Up Object",
  "version": "0.1.0",
  "task_type": "manipulation",
  "description": "...",
  "preconditions": [],
  "effects": [],
  "success_conditions": [],
  "failure_conditions": [],
  "recovery_rules": [],
  "downstream_contracts": {},
  "validation_status": "valid"
}
```

- [ ] **Step 3: Update TypeScript types**

Keep the existing demo behavior working, but mark `executable_api` and string-derived hooks as compatibility projections rather than canonical fields.

- [ ] **Step 4: Update the UI**

Display:

```text
schema version
Skill version
primitive/composite level
validation status
dependencies
safety stop conditions
```

- [ ] **Step 5: Verify**

```bash
python -m pytest tests/integration/test_demo_export.py -q
npm test
npm run build
```

Expected: Python export test, existing Vitest suite and Vite build pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/export_demo_skills.py src tests/integration/test_demo_export.py
git commit -m "feat: drive demo from canonical skill manifests"
```

### Task 10: Build the Phase-1 Evaluation Suite

**Files:**
- Create: `tests/evals/trigger_cases.yaml`
- Create: `tests/evals/representation_cases.yaml`
- Create: `tests/evals/composition_cases.yaml`
- Create: `tests/evals/test_trigger_eval.py`
- Create: `tests/evals/test_representation_eval.py`
- Create: `tests/evals/test_composition_eval.py`
- Create: `docs/phase1/evaluation-protocol.md`
- Create: `scripts/run_phase1_eval.py`

- [ ] **Step 1: Define evaluation baselines**

Compare:

```text
B0 Existing TypeScript/free-text representation
B1 Agent Skills SKILL.md only
B2 Recommended SKILL.md + typed skill.yaml
```

- [ ] **Step 2: Build trigger evaluation**

For each released Skill, create:

```text
10 positive prompts
5 paraphrased positive prompts
10 hard-negative prompts
5 neighboring-Skill confusion prompts
```

Report precision, recall, F1 and confusion pairs. Release target:

```text
macro precision >= 0.90
macro recall >= 0.90
no Skill pair with confusion rate > 0.15
```

- [ ] **Step 3: Build representation coverage evaluation**

Use 30 scenarios across manipulation, navigation and driving. Measure:

```text
required concept coverage >= 0.90
free-text fallback rate <= 0.10
schema validation pass rate = 1.00 for gold manifests
invalid mutation detection rate >= 0.95
```

- [ ] **Step 4: Build authoring consistency evaluation**

Have two independent authors encode the same five scenarios using the authoring Skill. Measure:

```text
top-level field agreement >= 0.90
predicate signature agreement >= 0.85
failure-to-recovery mapping agreement >= 0.85
median validation-fix cycles <= 2
```

- [ ] **Step 5: Build composition evaluation**

Evaluate at least:

```text
pick-and-place
open-drawer-and-store-object
navigate-and-inspect
abort-lane-change
```

All intentionally valid compositions must pass; all injected missing-input, cyclic-dependency and unsafe-retry variants must fail.

- [ ] **Step 6: Build determinism and performance evaluation**

Run each compile 20 times. Require:

```text
identical output digest across all runs
100-Skill validation time < 5 seconds
registry build time < 2 seconds
```

- [ ] **Step 7: Add one-command evaluation**

```bash
python scripts/run_phase1_eval.py --skills skills --output build/phase1-eval
```

The output directory must contain:

```text
summary.json
trigger-matrix.csv
representation-coverage.csv
composition-results.json
performance.json
report.md
```

- [ ] **Step 8: Commit**

```bash
git add tests/evals scripts/run_phase1_eval.py docs/phase1/evaluation-protocol.md
git commit -m "test: add phase one skill evaluation suite"
```

### Task 11: Freeze Schema v0.1 and Publish the Phase-1 Report

**Files:**
- Create: `docs/phase1/authoring-guide.md`
- Create: `docs/phase1/phase1-report.md`
- Create: `docs/phase1/migration-v0.1.md`
- Modify: `README.md`

- [ ] **Step 1: Run the complete verification gate**

```bash
python -m pytest --cov=world2skills --cov-report=term-missing
ruff check src tests scripts
w2s-skill validate skills --level composition
w2s-skill registry build skills --output build/registry.json
python scripts/run_phase1_eval.py --skills skills --output build/phase1-eval
npm test
npm run build
```

Required result:

```text
all tests pass
core branch coverage >= 90%
all released Skills valid
all release evaluation thresholds met
frontend build succeeds
```

- [ ] **Step 2: Freeze schema artifacts**

Record:

```text
schema version
schema SHA-256
registry SHA-256
released Skill IDs and versions
known limitations
extension requests deferred to v0.2
```

- [ ] **Step 3: Write the phase report**

The report must answer:

1. Why dual-layer representation was selected.
2. Which concepts are formalized and which remain references.
3. Whether the schema generalizes across manipulation, navigation and driving.
4. How much invalid structure the validator catches.
5. Whether independent authors produce consistent Skill packages.
6. Which stable interfaces are handed to phases 2 and 3.
7. Which limitations block real robot deployment.

- [ ] **Step 4: Update README**

Document only installation, validation, authoring, registry build, demo export and evaluation commands.

- [ ] **Step 5: Tag the milestone**

```bash
git add README.md docs/phase1 schemas skills src tests scripts pyproject.toml
git commit -m "release: freeze phase one skill schema v0.1"
git tag phase1-schema-v0.1.0
```

---

## 6. Schedule

Assumption: one full-time engineer/researcher plus one part-time robotics reviewer. Total duration is 8 weeks.

| Week | Main work | Exit criterion |
|---|---|---|
| 1 | Requirements, ontology, ADRs, core scaffold | Scope and representation decisions frozen |
| 2 | Typed model, expression AST, schema export | Minimal valid Skill round-trips |
| 3 | Package and semantic validator | L0-L2 invalid fixtures detected |
| 4 | Registry, dependency and composition validator | Primitive/composite checks pass |
| 5 | Authoring Skill and scaffolder | A fresh Skill can be created and validated |
| 6 | Migrate 6 existing Skills; add 2 cross-domain Skills | Nine packages pass L0-L3 |
| 7 | Frontend projection and full evaluation suite | Demo consumes generated Skill JSON |
| 8 | Threshold fixes, schema freeze, report and tag | `phase1-schema-v0.1.0` released |

For a single part-time contributor, use 12 weeks and keep the same task order.

---

## 7. Milestones and Acceptance Gates

### M1: Representation Draft

Deliverables:

- Requirements and ADRs.
- Pydantic models.
- JSON Schema.
- One valid and twelve invalid fixtures.

Gate:

- The same schema represents one manipulation and one driving Skill without custom top-level fields.

### M2: Construction Toolchain

Deliverables:

- `build-embodied-skill`.
- Scaffolder.
- L0-L3 validation.
- Registry.

Gate:

- A new Skill can be scaffolded, authored, validated, compiled and registered with documented commands.

### M3: Pilot Library

Deliverables:

- Six migrated manipulation Skills.
- Navigation Skill.
- Driving Skill.
- Composite Skill.

Gate:

- All packages pass validation and composition tests.

### M4: Phase-1 Release

Deliverables:

- Evaluation report.
- Generated frontend data.
- Authoring and migration guides.
- Versioned schema and registry.

Gate:

- All thresholds in Task 10 pass and the complete command gate in Task 11 is green.

---

## 8. Risks and Controls

| Risk | Observable symptom | Control |
|---|---|---|
| Schema overfits tabletop manipulation | Driving Skill needs many ad hoc fields | Keep a small core, test navigation and driving before v0.1 freeze |
| Schema becomes too broad | Most fields optional and validation weak | Define a required minimum core and move variants into typed extensions |
| `SKILL.md` and `skill.yaml` diverge | Agent instructions contradict manifest | Make YAML canonical; lint duplicated structured facts |
| Predicates become uncontrolled strings | Same concept appears under many names | Maintain versioned predicate catalog and signature validation |
| Formal language becomes too complex | Authors bypass it with prose | Bound v0.1 AST; record unsupported cases in schema-gap log |
| Platform details leak into core | Skill only works with one robot | Put runtime details under `grounding.bindings` |
| Safety is decorative | Released executable Skill lacks stop semantics | Make stop conditions and limits release-blocking |
| Trigger descriptions overlap | Wrong Skill is loaded | Maintain hard-negative and neighbor-confusion evaluation |
| Trace-derived edits damage Skills | Validation metrics regress after evolution | Require validation and held-out eval before accepting evolved packages |
| Frontend reintroduces a second schema | TypeScript data diverges | Generate frontend projection from compiled manifests only |

---

## 9. Phase-2 and Phase-3 Handoff Contracts

Phase 2 receives:

```text
Skill identity and version
typed inputs and parameters
scene-variable declarations
required labels
success/failure scenario classes
safety constraints
normalized source digest
```

Phase 3 receives:

```text
predicate catalog
preconditions and invariants
effects
failure codes
recovery mappings
composition graph
explainability requirements
```

Handoff rule: later phases may extend their own output schemas, but they must not reinterpret or mutate Phase-1 Skill semantics without a schema-version change.

---

## 10. Definition of Done

第一阶段只有在以下条件全部满足时结束：

1. `skill.yaml` schema、语义和版本策略形成正式文档。
2. Agent Skills 包格式通过官方 validator 和项目 validator。
3. 九个发布 Skill 覆盖操作、导航、驾驶和组合场景。
4. 所有发布 Skill 通过 L0-L3；目标平台发布时通过 L4。
5. 结构化表示不存在必须依赖自然语言解析的关键字段。
6. 构建、验证、编译、注册和导出均可由 CLI 重复执行。
7. 同一输入重复编译得到相同摘要。
8. 触发、覆盖、组合、错误检测和性能指标达到 Task 10 阈值。
9. React 演示不再维护独立的手写 Skill 真源。
10. Phase 2 和 Phase 3 的输入契约已冻结并写入发布报告。
