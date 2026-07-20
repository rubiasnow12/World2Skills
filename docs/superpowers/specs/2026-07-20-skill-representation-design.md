# World2Skills 第一阶段设计：自动驾驶 Skill 构建与表示

- 日期：2026-07-20
- 阶段：World2Skills 第一阶段（Skill 构建与表示）
- 状态：待评审

## 1. 背景与目标

World2Skills 旨在构建一个以 Skill 为中心的具身智能框架：

```
环境经验 → Skill 抽象 → 数据/规则自动生成 → 策略学习 → 新任务泛化
```

第一阶段只做**「构建与表示」**：建立一套统一的技能表示形式。数据生成（第二阶段）、规则生成（第三阶段）、策略学习（第四阶段）、泛化验证（第五阶段）都不在本阶段范围内。

本阶段面向的具身领域为**自动驾驶**（与项目第二阶段的调研方向 Cosmos-Drive-Dreams、GAIA-2、EGO-Planner、Marble 一致）。

### 与现有仓库的关系

当前仓库 Trace2Skill 已经使用 agentskills.io 的 `SKILL.md` 格式表示「文本 Agent 技能」（如 `spreadsheet_agent/skills/xlsx*/SKILL.md`），并有 `skill_evolver/` 从执行 trace 蒸馏技能的管线。World2Skills 复用同一套 `SKILL.md` **发现与目录约定**，把它从「文本/编码 Agent」迁移到「自动驾驶」。

**兼容性边界（重要，勿高估）：**

- 现有加载器 `spreadsheet_agent/agents/cli_skill_preloaded_agent.py` 只用正则从 frontmatter 抽取 `name` 与 `description`（`extract_skill_metadata`，第 34–59 行），按目录发现 `SKILL.md`（`discover_skills`，第 62–75 行）。它**不解析** frontmatter 的其它字段，也不读 `skill.yaml`。
- 现有 `skill_evolver/` 只处理 `SKILL.md` 的 markdown 正文，**不处理** `skill.yaml`。
- 标准 Agent Skills 校验器（skill-creator 的 `quick_validate.py`）只允许 frontmatter 顶层出现 `{name, description, license, allowed-tools, metadata, compatibility}`，**其它顶层键会被直接拒绝**；但 `metadata` 下的嵌套键不受校验。

因此本阶段的表示形式必须满足：**frontmatter 只放标准字段**，完整的驾驶表示放到独立文件 `skill.yaml`。所谓「兼容」是指**能被现有发现机制识别、符合目录约定**，而不是现有工具链会消费驾驶字段。

## 2. 范围（第一阶段交付边界）

采用「选项 B」：交付**表示语言的定义与示例**，不写工具链、不做轨迹抽取。

**交付物：**

1. **元模型规范文档**（人读）：定义一个 Skill 由哪些部分组成、每部分的语义与取值约定、粒度规则。
2. **机读 schema**（JSON Schema 数据文件）：约束 `skill.yaml`（完整驾驶表示）的结构，供将来程序校验使用。本阶段只交付 schema 文件本身，不写运行它的自定义校验器代码。
3. **手写示例 Skill 库**：5 个覆盖纵向/横向/交互三类驾驶行为的种子技能实例，**固定交付 5 个**。

**明确不做（留给后续阶段）：**

- 自定义校验器 / 加载器代码 —— 本阶段验收改用现成工具（见第 9 节），不自研运行时。
- 轨迹 → Skill 自动抽取原型 —— 属于「环境经验 → Skill 抽象」的生成环节，与后续阶段重叠。
- 任何仿真器/数据集/规划栈的接入代码（「先不碰后端」）。

## 3. 关键设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 目标领域 | 仅自动驾驶 | 与第二阶段调研方向一致，聚焦。 |
| 表示形式 | `SKILL.md`（标准 frontmatter + markdown 正文）+ `skill.yaml`（完整结构化驾驶表示） | frontmatter 只放标准字段以通过标准校验器；驾驶语义放 `skill.yaml`，作为 JSON Schema 校验对象；正文供策略/LLM 读。 |
| 交付范围 | 选项 B（规范 + schema + 示例，不含自研工具链） | 轻量，严格锁定在「表示」，不越界到「生成」。 |
| 观测/动作接地 | 抽象接口（`interface`）与后端绑定（`groundings`）分离，以 highway-env 为参照后端 | 避免把 `kinematic` 与 `Kinematics`、`trajectory` 混进同一枚举、随后端增加而膨胀。 |
| 条件表达 | 自由谓词字符串 | 符合 B 的轻量定位，人与 LLM 可读；将来要机器求值时再收紧成语法。 |

### 参照环境：highway-env

选用 highway-env（Farama HighwayEnv）作为参照后端的原因：纯 Python、pip 可装、gym 接口、闭环执行；场景名（highway / merge / roundabout / intersection …）与驾驶技能几乎一一对应；离散元动作 `LANE_LEFT / IDLE / LANE_RIGHT / FASTER / SLOWER` 天然就是技能原语。第一阶段仅将其作为 `groundings` 的参照后端让字段具体准确，**不写任何接入代码**。

从 highway-env 文档/源码核实的标识符（供 `groundings` 取值参照）：

- 观测类型：`Kinematics`、`OccupancyGrid`、`GrayscaleObservation`、`TimeToCollision`、`LidarObservation`
- Kinematics 特征：`presence, x, y, vx, vy, heading, cos_h, sin_h, cos_d, sin_d, long_off, lat_off, ang_off`
- 动作类型：`ContinuousAction`（throttle, steering）、`DiscreteAction`、`DiscreteMetaAction`
- DiscreteMetaAction 元动作：`{0: LANE_LEFT, 1: IDLE, 2: LANE_RIGHT, 3: FASTER, 4: SLOWER}`
- `merge-v0`：ego 在**主路**行驶、为匝道来车让出空间（非 ego 自己汇入）；`intersection-v0` 默认路线为 ego 从南向西的**左转**。

## 4. Skill 粒度定义

一个 Skill = **一个可复用、有名字的驾驶行为/机动**，把「一类场景」映射到「一种行动方式」。

- 例：变道超车、为汇入车让行、无保护左转、环岛通行、跟车保持车距。
- 每个技能对应 highway-env 的一个场景，且由若干元动作组合而成。
- 不是一个 Skill：单个元动作（`FASTER`）粒度过细；「安全驾驶」粒度过粗、不可映射到具体场景。

## 5. 表示形式

一个 Skill = 一个文件夹 `skills/<skill-id>/`，含两个文件：

```
skills/<skill-id>/
  SKILL.md      # 标准 Agent Skills：name / description / metadata + markdown 正文
  skill.yaml    # 完整驾驶表示，JSON Schema 校验对象
  references/   # 可选：细化模式或参数说明
```

### SKILL.md（发现层，必须通过标准校验器）

- frontmatter **只含标准字段**：`name`、`description`、`metadata`。
- `description` **必须单行**（现有正则加载器 `cli_skill_preloaded_agent.py:47` 会把多行 description 截断为首行）。
- `metadata` 只放字符串索引，指向完整表示：

```yaml
---
name: unprotected-left-turn
description: Use when the ego must turn left across oncoming traffic at an unsignalized intersection, yielding to cross and oncoming vehicles.
metadata:
  schema_version: "0.1"
  representation: skill.yaml
---
```

正文固定小节：`## When to use` / `## Procedure` / `## Reasoning cues` / `## Failure modes & recovery`（这四节必含）；`## References`（**可选**，仅当存在 `references/` 时出现）。

### skill.yaml（表示层，JSON Schema 校验对象）

承载完整驾驶语义，分为抽象接口与后端绑定两部分（见第 6 节字段集与示例）。

## 6. Schema 字段集（skill.yaml）

`skill.yaml` 顶层字段：

- `schema_version`（string，必填）：表示格式版本，如 `"0.1"`。
- `name`（string，必填）：须与文件夹名及 `SKILL.md` 的 `name` 一致。
- `version`（string，必填）：本技能自身版本，如 `"1.0"`。
- `domain`（string，必填）：固定 `autonomous-driving`。
- `category`（string，必填）：`longitudinal` / `lateral` / `interaction` 之一。
- `tags`（string 数组，可选）：检索标签。
- `description`（string，必填）：技能用途（可多行，此文件不受首行截断限制）。
- `entities`（数组，必填）：技能所推理的对象，每项含 `name`、`role`（如 `controlled-vehicle` / `traffic` / `static-topology`）。
- `parameters`（映射，可选）：可调参数，每项含 `type`、`unit`、`range`、`default`、`description`。
- `interface`（对象，必填）：**后端无关**的抽象接口。
  - `observations`：每项含 `name`、`type`（抽象类型：`kinematic` / `image` / `grid` / `track`）、可选 `features`。
  - `actions`：每项含 `name`、`type`（抽象类型：`discrete-meta` / `continuous` / `trajectory`）、`primitives`。**原语必须是后端无关的抽象名**（如 `maintain-speed` / `accelerate` / `decelerate` / `change-lane-left` / `change-lane-right`），不得直接使用后端标识符；到后端动作的映射由 `groundings[].primitive_map` 负责。
- `preconditions`（谓词字符串数组，必填）：可应用的前置条件。
- `execution`（对象，必填）：**闭合的状态图**，含：
  - `entry`（string，必填）：入口步骤的 `id`。
  - `steps`（数组，必填）：每步含 `id`（稳定标识）、`kind`（`observe` / `act` / `branch` 之一）、`action`（当 `kind: act` 时为一个抽象原语名）、`transitions`（数组）。
  - `transitions` 每项含 `condition`（谓词字符串）与 `next`（目标步骤 `id`，或字面量 `done` 表示终止）。**图必须闭合**：每个 `next` 必须指向 `steps` 中存在的 `id` 或 `done`；`branch` 步骤的 `transitions` 条件应覆盖互斥且完备的情形。
- `effects`（谓词字符串数组，必填）：成功后的状态变化（后置条件）。
- `success_criteria`（谓词字符串数组，必填）：可观测的成功条件。
- `failure_criteria`（谓词字符串数组，必填）：**可观测的**失败条件（区别于叙述性 `failure_modes`）。
- `safety_constraints`（谓词字符串数组，必填）：任何时刻都须成立的不变式（invariants）。
- `failure_modes`（字符串数组，必填）：叙述性失败模式，供人/LLM 阅读，承接后续规则生成。
- `termination`（string，必填）：终止条件描述。
- `related_skills`（string 数组，可选）：相关技能 id。
- `groundings`（数组，必填，至少一项）：**后端绑定**。每项含 `backend`（如 `highway-env`）、`backend_version`、`environment`（如 `intersection-v0`）、`observation`（后端观测类型与特征）、`action`（后端动作类型）、`primitive_map`（映射：把 `interface` 中的抽象原语名映射到后端具体动作，如 `maintain-speed: IDLE`）。`primitive_map` 必须覆盖 `interface.actions[].primitives` 中出现的全部抽象原语。

### 示例 skill.yaml（unprotected-left-turn，节选）

```yaml
schema_version: "0.1"
name: unprotected-left-turn
version: "1.0"
domain: autonomous-driving
category: interaction
tags: [yield, unsignalized, left-turn]

description: Turn left across oncoming traffic at an unsignalized
  intersection, yielding to cross and oncoming vehicles.

entities:
  - { name: ego, role: controlled-vehicle }
  - { name: oncoming_vehicles, role: traffic }
  - { name: intersection, role: static-topology }

parameters:
  target_speed: { type: float, unit: m/s, range: [0, 10], default: 4,
                  description: cruising speed when the path is clear }
  yield_gap:    { type: float, unit: s,   range: [2, 6],  default: 3,
                  description: minimum time gap to oncoming traffic before committing }

interface:
  observations:
    - { name: surrounding_vehicles, type: kinematic,
        features: [presence, x, y, vx, vy, heading] }
    - { name: ego_state, type: kinematic }
  actions:
    - { name: longitudinal, type: discrete-meta,
        primitives: [maintain-speed, accelerate, decelerate] }

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
        - { condition: "gap < yield_gap",  next: wait }
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
        - { condition: "ego not yet clear",       next: commit }
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

### 条件表达约定

`preconditions`、`execution.steps[].transitions[].condition`、`effects`、`success_criteria`、`failure_criteria`、`safety_constraints` 均使用**自由谓词字符串**（如 `"ego.lane != leftmost"`），人与 LLM 均可读。第一阶段（B）不写求值器，故不要求可执行；将来需要机器求值时再收紧为迷你类型化语法。

## 7. 示例 Skill 库

**固定交付 5 个**种子技能，覆盖纵向 / 横向 / 交互三类，各对应一个 highway-env 场景：

| Skill id | highway-env 场景 | category | 侧重 |
|----------|-----------------|----------|------|
| `lane-change-overtake` | highway-v0 | lateral | 变道 + 间隙推理 |
| `facilitate-highway-merge` | merge-v0 | interaction | ego 在主路为匝道来车让出空间、保持车速 |
| `unprotected-left-turn` | intersection-v0 | interaction | 左转 + 让行 |
| `roundabout-navigate` | roundabout-v0 | interaction | 连续交互 |
| `follow-keep-distance` | highway-v0 | longitudinal | 纵向基线 |

说明：

- `facilitate-highway-merge` 反映 `merge-v0` 的真实语义——ego 行驶在主路、为匝道汇入车辆创造空间并保持高速，而非 ego 自己汇入。
- `unprotected-left-turn` 对应 `intersection-v0` 默认从南向西的左转路线。
- 若需定义最小覆盖子集（三类各一），取 `lane-change-overtake`（横向）、`unprotected-left-turn`（交互）、`follow-keep-distance`（纵向）；但本阶段计划**固定交付全部 5 个**，不设回退。

## 8. 目录布局

新建自包含的 `world2skills/` 顶层目录，不与现有 spreadsheet 代码纠缠：

```
world2skills/
  docs/
    skill-representation-spec.md      # ① 元模型规范（人读）
  schema/
    skill.schema.json                 # ② 机读 JSON Schema（校验 skill.yaml）
  skills/
    lane-change-overtake/
      SKILL.md                        # ③ 标准发现层
      skill.yaml                      #    完整驾驶表示
    facilitate-highway-merge/
      SKILL.md
      skill.yaml
    unprotected-left-turn/
      SKILL.md
      skill.yaml
    roundabout-navigate/
      SKILL.md
      skill.yaml
    follow-keep-distance/
      SKILL.md
      skill.yaml
  README.md                           # 阶段说明 + 如何新增技能
```

说明：B 不含自研校验器/加载器代码，但 `skill.schema.json` 是**数据文件**（JSON Schema 本身），属于交付物。

## 9. 验收标准

采用**现成工具**校验，不自研运行时：

- `world2skills/docs/skill-representation-spec.md` 完整定义元模型（字段语义、粒度规则、条件约定）。
- `world2skills/schema/skill.schema.json` 是合法 JSON Schema（draft 2020-12），能表达第 6 节全部字段与约束。
- **结构自动校验**（现成工具，无自研运行时）：
  - 每个 `SKILL.md` 通过 skill-creator 的 `quick_validate.py`：frontmatter 只含允许字段、`name` 为 kebab-case 且无尖括号、`description` 单行。（注意：该工具**不**校验 `name` 与文件夹名是否一致。）
  - 每个 `skill.yaml` 用现成 JSON Schema 校验器（如 `check-jsonschema` 或 `python -m jsonschema`）对照 `skill.schema.json` 通过。（注意：JSON Schema 只做单文档结构校验，**不**检查跨文件 `name` 一致性、`execution` 图内 `next` 引用、以及 `primitive_map` 对抽象原语的覆盖。）
- **跨文件及引用完整性人工核对**（本阶段不写自定义校验器，故人工逐项核对）：
  - `name` 在文件夹名 / `SKILL.md` / `skill.yaml` 三处一致。
  - `execution` 图闭合：`entry` 与每个 `transitions[].next` 均指向存在的 `steps[].id` 或字面量 `done`。
  - `primitive_map` 覆盖 `interface.actions[].primitives` 的全部抽象原语；`execution` 中 `kind: act` 步骤的 `action` 也在该原语集合内。
  - 正文含四个必含小节（`References` 可选）。
- `world2skills/README.md` 说明阶段目标与如何新增技能（含上述自动校验命令与人工核对清单）。

## 10. 不做清单（防止范围蔓延）

- 不写自研校验器 / 加载器 / 任何运行时代码（验收用现成校验器）。
- 不接入 highway-env 或任何仿真器/数据集。
- 不做轨迹 → Skill 抽取。
- 不改动现有 `spreadsheet_agent/`、`skill_evolver/`、`src/` 代码。
- 不做数据生成、规则生成、策略学习、泛化验证（第二~五阶段）。

## 11. 评审采纳记录

对第一版设计的评审意见已逐条核对代码后采纳，并修正评审中两处措辞不准之处：

- **驾驶字段不能放 frontmatter 顶层**：已核实 `quick_validate.py:42` 的 `ALLOWED_PROPERTIES` 会拒绝未知顶层键（第 45–50 行），`metadata` 嵌套键不受校验。采纳 `SKILL.md` + `skill.yaml` 拆分。更正：该校验器不在本仓库内，是外部 skill-creator 的标准校验器（仍是权威校验器，故按其约束设计）。
- **多行 description**：现有正则 `cli_skill_preloaded_agent.py:47` 会将多行 description **截断为首行**（并非解析为空）；结论不变——强制单行。
- **行为语义缺失**：采纳新增 `execution`（有序/条件/分支）、`effects`、`failure_criteria`（可观测，区别于叙述性 `failure_modes`）、`safety_constraints`、`schema_version`/`version`、`category`/`tags`、`entities`、参数 `description`。
- **抽象与后端分离**：采纳 `interface`（抽象类型）与 `groundings`（后端类型）拆分，消除类型枚举膨胀。
- **技能选型**：已核实 `merge-v0` 语义，将 `merge-onto-highway` 更名为 `facilitate-highway-merge`；`unprotected-left-turn` 经核实为默认左转路线，保留；固定交付 5 个。
- **三处一致性**：状态改为「待评审」；`References` 明确为可选、其余四节必含；验收改用现成校验器（`quick_validate.py` + JSON Schema 校验器），不再仅人工核对。

### 第二轮评审采纳

- **execution 图不闭合**：原示例 `next: decide` 指向不存在的步骤、分支语义不明。采纳改为闭合状态图：`entry` + `steps[{id, kind: observe|act|branch, action, transitions[{condition, next}]}]`，`next` 必须指向存在的 `id` 或字面量 `done`。
- **抽象接口泄漏后端标识符**：`interface`/`execution` 原用 `IDLE/FASTER/SLOWER`（highway-env 专有）。采纳改为抽象原语 `maintain-speed/accelerate/decelerate`，并在 `groundings[].primitive_map` 建立到后端动作的映射。
- **验收措辞更正**：已核实 `quick_validate.py` 不校验 `name` 与文件夹名一致，JSON Schema 也无法校验跨文件一致性、`next` 引用与原语覆盖。故将验收拆为「结构自动校验（现成工具）」与「跨文件及引用完整性人工核对」两类，以保持「不写自定义校验器」的范围。
