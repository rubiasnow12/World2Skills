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
   - `SKILL.md` frontmatter 中的 `description` 是单行字符串；
   - `name` 在文件夹名 / `SKILL.md` / `skill.yaml` 三处一致；
   - `execution` 图闭合：`entry` 与每个 `transitions[].next` 指向存在的 step id
     或字面量 `done`；
   - `primitive_map` 覆盖 `interface.actions[].primitives` 全部抽象原语，
     且 `execution` 中 `kind: act` 步骤的 `action` 都在该集合内；
   - 正文四个必含小节齐全。
