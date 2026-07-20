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
- 人工核对（自动工具查不了的）：`SKILL.md` 的 `description` 保持单行、
  三处 `name` 一致、`execution` 图闭合、`primitive_map` 覆盖全部抽象原语、
  正文四个必含小节。
