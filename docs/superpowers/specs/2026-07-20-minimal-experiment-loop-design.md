# World2Skills 第二阶段：最小实验闭环设计 (v0.1)

> 状态：已通过 brainstorming「有条件批准」，本文件为待评审的规范。
> 关联：第一阶段产物见 `world2skills/README.md` 与
> `world2skills/docs/skill-representation-spec.md`。

## 0. 背景与现状

第一阶段完成「技能表示」：`world2skills/skills/` 下 5 个种子技能，结构校验
全 `[OK]`。表示层用抽象 `interface`/`execution` + `groundings[].primitive_map`
把抽象原语映射到 highway-env `DiscreteMetaAction`。`execution` 图里的条件、
`success_criteria` 等均为**自由文本谓词**，规范明确「本阶段不写求值器」
（`skill-representation-spec.md:83`）。因此当前只能做结构校验，**不能跑实验**。

本阶段目标：打通最小实验闭环
`skill.yaml → 执行器 → highway-env → 指标记录/评测`。

两个已敲定的核心选择：

1. **执行器 = LLM-as-executor**：技能作为 LLM 可读知识，每步由模型读技能 +
   观测，返回一个抽象原语。
2. **粒度 = LLM 直接选原语**：`execution` 图作为 prompt 里的建议流程（非硬约束），
   Python 只做 `原语 → DiscreteMetaAction` 映射并推进环境；天然支持
   「有技能 / 无技能」消融。

## 1. 目标与范围

### 1.1 里程碑

- **M1（本规范交付物）**：`lane-change-overtake` @ `highway-v0`，LLM 驱动跑通，
  在**确定性任务场景**（保证存在慢速前车 + 邻接车道）上跑 N 个固定种子，
  用**场景级 evaluator** 判定「是否完成对同一目标车的超越」，落盘每步轨迹 +
  每 episode 指标 + 聚合汇总。
- **M2（本规范之外）**：加「无技能」LLM 基线消融；铺开到全部 5 个技能/场景；
  异步批量。

### 1.2 非目标

训练/微调、连续控制/轨迹动作、多后端（只 highway-env）、异步并发（M1 顺序执行）。

### 1.3 M1 的成功定义（可执行，替代自由文本 `success_criteria`）

由 `LaneChangeOvertakeScenario` 在 reset 时绑定 `initial_lead_vehicle`，逐步跟踪：

```
success = ego 曾发生换道 (lane_changed)
          AND ego 已超过 initial_lead_vehicle 至少 success_margin 米 (overtaken)
          AND collision == false
```

`lane_changed`、`overtaken`、`collision` 均从**环境 Vehicle 对象的真值**读取，
不依赖观测行号。场景初始化必须主动保证「慢车 + 邻接车道」前置条件成立——
仅用随机 stock `highway-v0` 不能保证，见 §5。

> 5 seeds 是**工程验收**下限（证明闭环可跑通、指标齐全），不足以形成效果结论；
> 效果结论留待 M2 扩大样本与加基线。

## 2. 目录结构

`world2skills/` 从「数据 + 校验脚本目录」升级为 Python 包。新增：

```text
world2skills/
  __init__.py
  runtime/
    __init__.py
    types.py          # 共享 dataclass：SkillCard/Grounding/ObservationContext/
                      #   StepRecord/EpisodeResult/BatchResult
    llm.py            # 本项目内精简 LLM 客户端（OpenAI/AzureResponses/Mock）
    skill_loader.py   # 读 SKILL.md + skill.yaml，选定 highway-env grounding
    scenario.py       # 场景前置条件 + 目标车绑定 + 成功判定（核心新增）
    obs_render.py     # render(observation, context) -> str
    executor.py       # LLMSkillExecutor：建 prompt→调用→结构化解析→映射动作
    env_factory.py    # gymnasium.make + 配置 Kinematics/DiscreteMetaAction + 固定 seed
  evaluation/
    __init__.py
    episode.py        # 跑一个 episode
    metrics.py        # 每 episode + 聚合指标
    run.py            # CLI：多种子批量 → results.json + 轨迹 + config 快照
  experiments/
    configs/          # 实验配置（仅 YAML，不作为 Python 包）
  outputs/            # 运行产物（顶层，git 忽略）
  tests/
  requirements-runtime.txt
  # 既有：docs/ schema/ tools/ skills/ README.md 保持不变
```

约定：`evaluation/`（非 `eval/`，避免与内建 `eval` 混淆）承载 episode/metrics/run；
`experiments/` 只存配置，不含 `.py`；`outputs/` 为运行产物根，git 忽略。

## 3. 组件设计

每个单元职责单一、可独立测。下列「依赖」指模块级依赖。

### 3.1 `runtime/types.py`

纯 dataclass，无行为，供全项目复用：

- `Grounding`：`backend/environment/observation/action/primitive_map`（从 skill.yaml 取）。
- `SkillCard`：`name/description/skill_md_body/parameters/interface/execution/
  preconditions/success_criteria/safety_constraints/...` + `primitives`（该技能
  允许的抽象原语集 = `interface.actions[].primitives` 并集）。
- `ObservationContext`：`available_primitives`（当前可用抽象原语）、`ego_lane`、
  `target_relation`（到绑定目标车的纵向间距/相对速度/在前或在后）、
  `prev_primitive`、`left_lane_exists/right_lane_exists`、`left_gap/right_gap`。
- `StepRecord`：`t/obs_summary/primitive/backend_action/action_index/reward/
  crashed/cache_hit/latency_ms/note`。
- `EpisodeResult`：见 §7 指标 + `status`（`ok`/`error`）+ 错误信息。
- `BatchResult`：见 §7 `results.json` 结构。

### 3.2 `runtime/llm.py`（本项目内实现，修正缓存语义）

在本项目内**重新实现**精简客户端（不跨目录 import Trace2Skill，也不原样复制）。
接口沿用 `Message`/`ModelSettings`/`LLMClient`，提供三种实现：

- `OpenAIClient`：chat.completions（兼容 vLLM/本地 qwen，`api_key="EMPTY"`）。
- `AzureResponsesClient`：Responses API（GPT-5.4 走 `scripts/connect_gpt54.sh`
  的本地 Azure 代理）。
- `MockLLMClient`：预置回复，供单测。

**缓存键修正**（Trace2Skill 的键仅 `(model, messages)`，遗漏生成配置/API 类型/
endpoint，有污染风险，见 `Trace2Skill/src/react_agent/models.py:97`）。本项目缓存键至少含：

```
client_type + model + endpoint_identity(base_url + api_type + api_version)
+ messages + temperature + max_tokens + extra_body + prompt_version
```

每次调用记录 `request_hash`、`cache_hit`(bool)、`latency_ms`。

**确定性说明**：缓存只保证「同输入重放一致」，**不等于模型本身确定**；
`AzureResponsesClient` 还会主动移除 `seed`
（`Trace2Skill/src/react_agent/models.py:696`）。故复现性来自「固定 env seed +
缓存重放」，报告需按多种子分布呈现，不宣称单次确定。

### 3.3 `runtime/skill_loader.py`

`load_skill(skill_id) -> SkillCard`：解析 `skills/<id>/SKILL.md`（frontmatter +
正文四小节）与 `skill.yaml`；`select_grounding(card, backend="highway-env")
-> Grounding`。依赖 `pyyaml`、`types`。

### 3.4 `runtime/scenario.py`（核心新增）

把「场景前置条件 / 目标车绑定 / 成功判定」集中，避免散落到 renderer/episode/metrics。
M1 实现 `LaneChangeOvertakeScenario`：

- `configure() -> dict`：返回 highway-v0 配置（`lanes_count>=2`、单受控车、
  DiscreteMetaAction、Kinematics、`duration`、`policy_frequency`、
  `simulation_frequency`）。
- `reset(env, seed)`：
  1. `env.reset(seed=seed)`；取 `ego = env.unwrapped.vehicle`、记 `initial_ego_lane`。
  2. **强制前置条件**（主策略：reset 后改场景，确定性强）：找 ego 同车道最近前车；
     若不存在则在 ego 前方同车道生成一辆；将其 `speed`/`target_speed` 设为
     `lead_speed_ratio × ego_target_speed`（默认 0.6）。确认存在邻接车道
     （highway-v0 多车道天然满足）。
  3. **绑定** `initial_lead_vehicle = 该车对象`，记初始纵向位置。
  4. 返回 `(obs, info, ObservationContext 初值)`。
  （备策略：seed 拒绝采样，仅在改场景不可行时用；M1 用主策略。）
- `update(env)`：每步后刷新 `lane_changed`（ego `lane_index` 曾≠ `initial_ego_lane`）、
  `overtaken`（`ego.position` 纵向 − `initial_lead_vehicle.position` 纵向 ≥
  `success_margin`）、`collision`（`ego.crashed`）。所有量取自 Vehicle 对象真值。
- `build_context(env, prev_primitive, available_primitives) -> ObservationContext`：
  组装给 renderer 的上下文（目标车关系、可用原语、左右车道存在与间距）。
- `evaluate() -> (success: bool, success_reason: str)`：按 §1.3 判定。

参数：`success_margin`（默认 5 m）、`lead_speed_ratio`（默认 0.6）、`target_speed`
（取技能 `parameters.target_speed.default`）。依赖 highway-env、`types`。

### 3.5 `runtime/obs_render.py`

`render(observation, context: ObservationContext) -> str`：**必须接收环境上下文**，
不能只看 ndarray。Kinematics 数组无车道拓扑与稳定车辆 ID，默认还归一化。
渲染内容：ego 行（米制）+ 最近若干邻车行 + 上下文派生字段（当前可用原语、
ego 所在车道、到目标车的纵向间距/相对速度、左右车道是否存在及间距、上一步原语）。
目标车身份来自 scenario，不靠观测行号。依赖 `numpy`、`types`。

### 3.6 `runtime/executor.py`

`LLMSkillExecutor(skill_card, grounding, llm_client, prompt_version)`：

- `build_messages(obs_text, context) -> list[Message]`：
  - system：策略角色 + 抽象原语语义 + 技能 `safety_constraints` +
    **严格输出契约**「只输出一个 JSON 对象 `{"primitive": "<允许原语之一>"}`」。
  - user：技能卡（SKILL.md 正文 + skill.yaml 的 procedure/execution 作为建议流程、
    parameters、success/failure/safety）+ 观测文本 + **本步允许原语**（技能原语集
    ∩ 当前可用原语）+ 输出格式提醒。
- `decide(obs_text, context, available_primitives) -> (action_index, StepRecord)`：
  1. 调 LLM；异常经客户端重试后仍失败 → 计 `llm_errors`，回退。
  2. **结构化解析**：先剥离 fenced ```json 包裹，再 `json.loads`；取 `primitive`
     字段并校验 ∈ 技能原语集。**不允许在回复里搜子串**（"do not accelerate,
     maintain-speed" 会歧义）。非法 JSON / 未知原语 → 回退 `maintain-speed`
     （若不在集内则取该技能首个原语）+ 计 `parse_failures` + 记原始回复。
  3. **可用性检查**：若所选原语当前不可用（不在 `available_primitives`）→ 显式回退
     + 计 `unavailable_action_attempts`，**不静默交给环境**。
  4. **映射**：原语 →（`grounding.primitive_map`）后端名 →（反转
     `env.unwrapped.action_type.actions` 得 name→index）离散索引。

依赖 `llm`、`types`、`json`。

### 3.7 `runtime/env_factory.py`

`make_env(grounding, scenario_config, seed) -> (env, name_to_index)`：

- `gymnasium.make(grounding.environment, config={...}, render_mode=None)`，
  合并 scenario 配置。
- 观测：`type=Kinematics`，`features` 取自 grounding，**`normalize=False`**
  （保证米制）；`absolute` 见 §10 开放项；`see_behind=True`（供后向间距）；
  `vehicles_count` 足够看清邻车。
- 动作：`type=DiscreteMetaAction`。
- `name_to_index = {v: k for k, v in env.unwrapped.action_type.actions.items()}`
  （**反转 index→label**；不依赖未文档化的 `actions_indexes`）。

依赖 highway-env、gymnasium、`types`。

### 3.8 `evaluation/episode.py`

`run_episode(env, executor, scenario, name_to_index, max_steps) -> EpisodeResult`：
每步 `ctx = scenario.build_context(...) → obs_text = render(obs, ctx) →
executor.decide(obs_text, ctx, ctx.available_primitives) → env.step →
scenario.update`，累积 `StepRecord`，直到 `terminated/truncated/max_steps`。结束时 `scenario.evaluate()` + 基础指标。
**异常处理**：整个循环 try/except，异常 → `EpisodeResult(status="error",
exception_type, steps_done, ...)`；`finally: env.close()`。依赖 §3.4–3.7、`types`。

### 3.9 `evaluation/metrics.py`

基础指标 dataclass + `aggregate(results) -> BatchResult`（success_rate、
collision_rate、mean_return 等）。纯函数，依赖 `types`。

### 3.10 `evaluation/run.py`

CLI：`python -m world2skills.evaluation.run --skill lane-change-overtake
--seeds 0 1 2 3 4 --model gpt-5.4 [--mock] [--out outputs/<run>]`。
装配 loader→scenario→env→executor，逐 seed 跑 episode，聚合，写：
`outputs/<run>/results.json`（§7 结构）、`episode_<seed>.jsonl`（逐步轨迹）、
`config.json`（模型/prompt_version/seeds/**实际安装的 highway-env 与 gymnasium 版本**/
scenario 参数）。

## 4. 数据流

- **一步**：`obs → obs_render(obs, ctx) → executor.decide → action_index →
  env.step → scenario.update → StepRecord`。
- **一集**：`scenario.reset(seed) → 循环步 → scenario.evaluate + 基础指标 →
  EpisodeResult`。
- **一批**：`for seed in seeds: 建 env → run_episode`（M1 顺序）`→ aggregate →
  写盘`。

## 5. 确定性任务场景（为什么不能用 stock highway-v0）

Stock `highway-v0` 随机初始化，不保证 ego 车道前方有慢车、也不保证「超车」有明确
目标对象。M1 由 `LaneChangeOvertakeScenario`（§3.4）在 reset 后主动改场景以满足
`preconditions`（慢速前车 + 邻接车道），并绑定唯一目标车，使 §1.3 的成功判定良定义。

## 6. 动作映射与解析（正确性要点汇总）

- 原语→后端名：`grounding.primitive_map`（如 `accelerate→FASTER`）。
- 后端名→索引：**反转 `action_type.actions`**（§3.7）。
- 可用动作：`env.unwrapped.action_type.get_available_actions()`；不可用原语显式回退
  并计 `unavailable_action_attempts`。
- 解析：严格 JSON（允许 fenced），禁止子串搜索；失败回退 + 计数。

## 7. 指标与产物

每 episode 统一基础指标（**成功判定必须走 scenario evaluator，不假设 env 提供
`arrived/success`**；highway `info` 主要是速度/碰撞/动作/奖励分项，各场景终止语义不一）：

```
episode_return, crashed, steps, mean_speed
parse_failures, unavailable_action_attempts, llm_errors
terminated, truncated, max_steps_reached, termination_reason
lane_changed, overtaken                # 场景特有
success, success_reason
status                                 # ok | error（异常 episode）
```

`results.json`（含 `schema_version`；顶层用 `seeds[]` 而非单个 `seed`）：

```json
{
  "schema_version": "0.1",
  "model": "gpt-5.4",
  "prompt_version": "lco-v1",
  "skill": "lane-change-overtake",
  "environment": "highway-v0",
  "seeds": [0, 1, 2, 3, 4],
  "episodes": 5,
  "success_rate": 0.8,
  "collision_rate": 0.2,
  "mean_return": 23.4,
  "results": [
    {"seed": 0, "status": "ok", "success": true, "success_reason": "overtook lead by 6.2m after left change",
     "episode_return": 25.1, "crashed": false, "steps": 38, "mean_speed": 24.3,
     "parse_failures": 0, "unavailable_action_attempts": 0, "llm_errors": 0,
     "terminated": true, "truncated": false, "max_steps_reached": false,
     "termination_reason": "success", "lane_changed": true, "overtaken": true}
  ]
}
```

## 8. 模型接入与缓存

默认 GPT-5.4（`AzureResponsesClient`，先 `source scripts/connect_gpt54.sh`）；
`OpenAIClient` 支持本地 vLLM/qwen；`MockLLMClient` 供测试。缓存键与语义见 §3.2。

## 9. 错误处理

- LLM 超时/429：客户端重试（沿用 Trace2Skill 的退避策略）。
- 解析失败 / 动作不可用 / LLM 最终失败：各自回退 + 独立计数，不中断 episode。
- 单 episode 异常：捕获 → 写 `status:"error"` + 异常类型 + 已完成步数；`finally`
  关闭环境；不中断整批。M1 顺序执行，异步批量留 M2。

## 10. 依赖与版本

- 新增：`highway-env`、`gymnasium`。写入 `requirements-runtime.txt`。
- **版本固定**（验证于 2026-07-20，PyPI 当前 `highway-env 1.12.0`，要求
  Python≥3.10 与 `gymnasium>=1.0`）：目标 `highway-env==1.12.0`、
  `gymnasium==1.3.0`（安装时以实际为准）。**`config.json` 必须记录实际安装版本**。
- 既有可用：`openai 2.41.1`、`diskcache 5.6.3`、`numpy 1.26.4`、`pyyaml 6.0.3`、
  `jsonschema 4.26.0`、`pytest 8.4.2`。环境：conda `llama_factory`，Python 3.11。

## 11. 测试计划（TDD；无需真模型即可跑）

- `test_skill_loader`：解析 5 技能、选中 highway-env grounding、原语集正确、
  SKILL.md 四小节齐全。
- `test_obs_render`：合成 Kinematics 数组 + `ObservationContext` → 文本含
  ego/邻车/可用原语/目标车间距。
- `test_executor`（`MockLLMClient`）：
  - 合法 JSON 原语 → 正确离散索引；
  - fenced ```json → 正确解析；
  - 非法 JSON → 回退 + `parse_failures`；
  - 未知原语 → 回退 + `parse_failures`；
  - 不可用动作 → 回退 + `unavailable_action_attempts`。
- `test_scenario_evaluator`：绑定同一 lead vehicle，验证「真正超过同一目标 + 换道 +
  无碰撞」才 success；只换道 / 超过的是别的车 / 有碰撞 → 非 success。
- `test_results_schema`：序列化一批（含 1 个 error episode），断言 `schema_version`、
  `seeds[]`、每 episode 必需字段、error episode 的 `status:"error"` 契约。
- `test_episode_smoke`（需装 highway-env，缺失则 skip）：脚本策略（恒定 accelerate）
  在 `highway-v0` 经 scenario 跑 1 集，断言能终止且指标齐全。
- **真实 GPT-5.4 五种子运行属于 M1 acceptance（CLI 手动跑），不进 pytest。**

## 12. M1 验收标准

1. `pytest world2skills/tests` 全绿（单测 + smoke；smoke 需已装 highway-env）。
2. CLI 跑 `lane-change-overtake @ highway-v0` 5 个固定种子（GPT-5.4）完成，产出
   `results.json`（schema v0.1）、逐 episode 轨迹、`config.json`（含实际版本）。
3. 每个种子场景确有慢速前车 + 邻接车道；成功 = 换道 + 超越同一目标车 + 无碰撞。
4. 指标含 `parse_failures / unavailable_action_attempts / llm_errors`。

## 13. 开放假设（实现前如无异议即采用）

- 观测 `absolute`：默认 `False`（米制、ego 相对帧）；目标车跟踪走 env 对象，故帧仅
  影响 LLM 文本表述。
- `success_margin=5 m`、`lead_speed_ratio=0.6`。
- `policy_frequency=1 Hz`、`duration=40 s`（≈40 步，`max_steps` 相应设定）。
- 「无技能」基线与 5 技能铺开留 M2。
