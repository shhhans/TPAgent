# 海外电商智能客服 Agent — 架构设计文档（v1）

> 场景：在国际社媒发布电梯外贸产品宣传视频，客户通过**评论 / 私信**与我们沟通。
> Agent 目标：在这种**长周期、异步、多语言**的触点中，做好**售前服务**（答疑 + 收集报价参数 + 风险兜底）。

---

## 0. 本文档的定位与取舍

本设计**刻意区分两类东西**：

- **v1 工程主线（本项目要做的）**：LangGraph 状态机编排 + 结构化路由 + RAG 售前 + HITL 人工中断 + Mem0 长期记忆。这是几周内能跑通、能产生业务价值的最小闭环。
- **后期研究专题（见第 9 节，本项目 v1 不做）**：私有化 8B 微调、Unlearning/ALKN 逆向学习、Agentic RL(GRPO)、多模态图纸对齐(PathFusion/PMF)、多智能体辩论评分(CoRE)。这些是论文级、长周期、产出不确定的方向，单独立项评估，不拖累主线。

> 核心工程原则：**"防止模型乱报承重/电压/价格" 用确定性工程解决（强制工具调用 + 数值校验 + HITL 审核），而不是用模型微调/unlearning 去"切除"。** 确定性手段可证明、可回滚；模型手段不可控、难验证。

---

## 1. 技术基座

| 层 | 选型 | 说明 |
|---|---|---|
| 大脑（LLM） | **MiniMax**（OpenAI 兼容 API） | `base_url=https://api.minimaxi.com/v1`，默认模型 `MiniMax-M2`（可切 `MiniMax-M3`/`MiniMax-M2.5`）。换供应商只改 `.env` 里的 `base_url+model+key`，不改业务代码 |
| 编排 | **LangGraph** | 显式状态机（StateGraph），条件边路由，原生 HITL 中断（`interrupt`） |
| 短期记忆 | **LangGraph Checkpointer** | 单次会话的消息序列 + 临时参数实时持久化（v1 用 SQLite，生产换 Postgres/Redis） |
| 长期记忆 | **Mem0** | 客户参数表 + 偏好字典的 LLM-based CRUD（合并/覆盖），跨会话召回。**v1 即上**；不可用时降级为本地 JSON 存储（接口一致） |
| 知识库 | **本地向量检索 RAG**（v1 纯文本） | 本地向量库（Chroma），产品规格/FAQ/外贸条款/合规规则。不可用时降级为关键词检索 |
| Embedding | **本地模型**（默认 Chroma 内置 MiniLM / 可配多语言模型） | 不依赖外部 embedding API，契合"本地向量库"要求 |
| 配置 | **`.env` + pydantic-settings** | **所有可配置参数统一放 `.env`**（模型、Mem0、向量库、参数 schema 路径、HITL 阈值等），代码不硬编码 |

> 模型抽象：所有 LLM 调用经过一个 `llm/` 适配层（基于 `openai` SDK 的 `OpenAI(base_url=...)`），上层只依赖 `chat(messages, response_format=...)`。未来若要换本地 8B（vLLM/SGLang 也提供 OpenAI 兼容端点），同样只改 `.env`。

> **部署形态**：本项目作为**服务**存在，社媒/中台对接由他人负责，v1 **不做对外接口**。先用 **Streamlit 简单可视化**（聊天 + 实时展示 State：意图/已收集参数/缺失参数/召回记忆/风险）跑通核心图，便于联调与演示。

> **配置约束（用户要求）**：凡可配置的参数一律落在 `.env`（见 `.env.example`），通过 `app/config.py` 统一加载，业务代码只读 `settings.xxx`，禁止散落硬编码。

---

## 2. 总体架构图

```
        社媒评论 / 私信（多语言、异步）
                    │
        ┌───────────▼─────────────┐
        │   Ingestion 入口适配层    │  归一化为 {channel, user_id, text, ts}
        └───────────┬─────────────┘
                    │
        ┌───────────▼─────────────────────────────┐
        │   Triage Router（前置分诊）               │
        │   1) mem_search 读 Mem0 客户历史 → 注入   │
        │   2) 推演逻辑 + 输出结构化 JSON 决策       │
        │      {intent, extracted_params,           │
        │       missing_params, require_human}      │
        └───────────┬─────────────────────────────┘
                    │ (结构化意图写入 State)
        ┌───────────▼─────────────┐
        │   Supervisor 调度        │  按 intent / 参数完备度 走条件边
        └─┬───────────────────┬───┘
          │                   │
   ┌──────▼───────┐    ┌──────▼──────────┐
   │ Worker1       │    │  HITL 中断节点   │
   │ RAG + 参数收集 │    │  高风险/越界挂起  │
   │ - 知识库检索   │    │  等人工审批       │
   │ - 写 State 参数│    └──────┬──────────┘
   │ - 缺参追问     │           │ resume
   └──────┬────────┘           │
          │                    │
        ┌─▼────────────────────▼─┐
        │  回复生成 / 发送          │
        └─────────┬───────────────┘
                  │ (会话空闲后异步触发)
        ┌─────────▼───────────────┐
        │  记忆固化 Worker         │  清洗压缩多轮历史 → 写 Mem0
        └─────────────────────────┘
```

> v1 **不含**报价合规 Agent (Worker2)、DeepResearch Agent (Worker3)；二者在图中预留路由出口，后续接入。

---

## 3. 全局状态（State）设计

LangGraph `StateGraph` 的共享状态，是各节点通信的唯一真相源。

```python
class AgentState(TypedDict):
    # —— 会话标识 ——
    customer_id: str                  # 跨会话稳定 ID（社媒账号映射）
    thread_id: str                    # 单次会话线程
    channel: Literal["comment", "dm"]

    # —— 消息 ——
    messages: Annotated[list, add_messages]  # 短期上下文

    # —— Triage 产出 ——
    intent_category: str              # 见 §4 意图枚举
    require_human: bool

    # —— 报价参数槽（核心业务数据）——
    collected_params: dict            # 已收集：{destination, voltage, load_capacity, ...}
    missing_params: list[str]         # 报价所需但还缺的

    # —— 长期记忆注入 ——
    customer_memory: dict             # mem_search 召回的历史偏好/参数

    # —— 控制位 ——
    next_worker: str | None           # Supervisor 的路由决策
    risk_flags: list[str]             # 触发 HITL 的原因
```

**参数槽（Schema-driven slot filling）** 是售前的核心，且**采购对象可能是整机，也可能是某个组件**（如曳引机、控制柜、导轨、门机系统）。因此参数 schema **以 ETIM / ECLASS 行业分类标准为驱动**：

1. 先识别客户询问的**对象所属分类**（ETIM class 或 ECLASS class，如 ETIM `EC...` 类目）。
2. 该分类在标准里有一组**已定义的特征/属性（features/properties）**，即"这个组件报价该问哪些参数"。
3. `missing_params` = 该分类必填特征中尚未填充的部分；`collected_params` 用标准特征编码做 key，避免自由发挥。

> v1 实现：`app/tools/params.py` 维护一份**分类→必填特征**的映射（可从 ETIM/ECLASS 拉取/缓存为本地 JSON，路径由 `.env` 配置）。未来可接入 ETIM/ECLASS 官方数据源在线拉取。
>
> 示例（电梯整机 + 曳引机组件，占位，待业务方/标准核准）：
> - 整机：`destination, voltage, load_capacity, floors, shaft_dimensions, trade_term`
> - 曳引机（组件）：`rated_load, rated_speed, voltage, traction_ratio, motor_power`

---

## 4. Triage Router（前置分诊）

**职能**：意图识别 + 参数抽取 + 路由决策。**不直接对客户说话**，只输出结构化 JSON。

**双阶段**：先"推演（thought_process）"，再"决策（结构化标签）"。用 OpenAI 兼容 API 的 `response_format={"type": "json_object"}` 强制 JSON。

输出契约：

```json
{
  "thought_process": "客户询问 Elevator_X 发往迪拜的价格，但未提供当地电压标准与载重，未达报价完备条件。",
  "intent_category": "needs_more_info",
  "extracted_parameters": {"destination": "Dubai", "product": "Elevator_X"},
  "missing_parameters": ["voltage", "load_capacity"],
  "require_human": false
}
```

`intent_category` 枚举（v1）：

| 取值 | 含义 | 路由去向 |
|---|---|---|
| `product_qa` | 纯产品技术/规格问题 | Worker1 (RAG) |
| `needs_more_info` | 有报价意图但参数不全 | Worker1 (追问) |
| `ready_to_quote` | 参数齐全、明确询价 | （v1 预留 → Worker2；当前转 HITL 人工报价）|
| `external_research` | 私库覆盖不了的外部长尾问题 | （v1 预留 → Worker3；当前转 HITL）|
| `chitchat` | 寒暄/无关 | 直接轻量回复 |
| `human_required` | 投诉/纠纷/高风险 | HITL |

**容错**：JSON 解析失败时重试一次（降温/给修复提示），仍失败则降级为 `human_required`，绝不向客户输出半成品。

---

## 5. Supervisor 调度

读取 Triage 写入 State 的结构化意图，通过 LangGraph **条件边**决定下一节点：

```
ready_to_quote 且 missing_params 为空     → (v1) HITL 人工报价
needs_more_info / product_qa             → Worker1
external_research                        → (v1) HITL
require_human / risk_flags 非空           → HITL
chitchat                                 → 轻量回复
```

Supervisor 自身**无业务 API**，只做"交警"。子 Agent 若返回 `{"status":"error"}`，经边退回 Supervisor 重新决策（为后续 Worker2 预留的错误回环模式）。

---

## 6. Worker1 — RAG + 参数收集（售前客服）

**职能**：答产品技术问题 + 主动追问补齐报价参数。

**工具/能力**：
1. `rag_search(query)`：检索企业知识库（v1 纯文本：产品规格、FAQ、外贸条款）。
2. `update_params(params)`：把客户本轮提供的电压/载荷等写入 `State.collected_params`，并重算 `missing_params`。
3. **缺参追问**：若仍有 `missing_params`，生成**一次只问 1~2 个关键参数**的自然语言追问（避免一次轰炸客户）。

**行为约束（防幻觉）**：
- 知识库检索不到的规格/承重/价格，**不得编造**；明确告知"需确认"并走人工或追问。
- 回复带可追溯来源（内部引用 doc id），便于审计。

---

## 7. HITL 人工中断

用 LangGraph `interrupt()` 在图中静态/动态挂起：

**触发条件（v1）**：
- 最终高额报价单生成前（v1 阶段报价本身就走人工）。
- 检测到风险越界：客户需求超产品规格极限、合规敏感（制裁国家/特殊清关）、投诉纠纷。
- Triage/Worker 主动置 `require_human=true` 或 `risk_flags` 非空。

**机制**：图暂停 → 状态经 Checkpointer 持久化 → 外贸经理在人工界面审批/改写 → `resume` 继续。异步特性天然契合社媒"留言后人不在线"的场景。

---

## 8. 记忆体系

### 8.1 短期（Thread-Level）
LangGraph Checkpointer 持久化单次会话的 `messages` + `collected_params`。会话结束清理上下文，但参数已固化前不丢。

### 8.2 长期（Mem0，v1 已落地）
- **写入**：会话结束（设计上为空闲 20~40s 异步）触发"记忆固化"：Mem0 用 LLM 把多轮历史清洗压缩成高纯度事实，做 LLM-based CRUD（自动合并/覆盖旧记忆）。
- **读取**：Triage 入口前的 `load_memory` 节点 `mem_search(customer_id)` 召回，注入 `State.customer_memory` 与 Prompt，复访即唤起上下文。
- **程序记忆**：风险对冲规范、合规流程 —— **不进 Mem0**，作为静态系统级 Prompt 每次注入（`PROCEDURAL_RULES`，不可被对话篡改）。
- **情景记忆**：关键历史节点（人工接管、清关纠纷）v1 以事实条目记录；强时序"前因后果"溯源需求出现后再评估引入 Zep。

> 决策记录：v1 **只用 Mem0 一套**长期记忆；Zep（时序图谱）列入后期可选，避免开局背两套数据一致性 + 故障面。

#### Mem0 接入实现要点（`app/memory/mem0_store.py`）
全本地化，不依赖外部 embedding API：

| 组件 | 选型 | 说明 |
|---|---|---|
| 记忆抽取 LLM | MiniMax（`mem0` 原生 `minimax` provider，默认 `MiniMax-M2`） | 需支持 `json_object`；`MiniMax-M2` 是推理模型会输出 `<think>`，已对 mem0 的 `MiniMaxLLM._parse_response` **打补丁剥离 think**，否则会破坏 mem0 的 JSON 解析 |
| Embedding | 本地 `sentence-transformers`（`huggingface` provider，多语言 MiniLM，384 维） | 首次运行从 HF Hub 下载模型；无需 embedding API |
| 向量库 | 本地 **Chroma**（持久化到 `data/mem0_chroma`） | 与 RAG 同栈 |
| 定制 | `custom_instructions` | 保留客户原文语言、聚焦贸易条款/机电参数/采购对象/关键事件 |

注意点（已踩坑修正）：
- MiniMax 的 embeddings 接口**非 OpenAI 兼容**（需 GroupId + 私有协议），故 embedding 走本地模型。
- mem0 2.x 的 `get_all` 必须用 `filters={"user_id": ...}`，不能用顶层 `user_id=`。
- 任一环节异常 → `search()` 返回空记忆、`finalize()` 降级本地 JSON，**不阻断主流程**。
- 切回无依赖模式：`.env` 设 `MEMORY_BACKEND=json` 即可。

---

## 9. 后期路线图（v1 不做，原始亮点收纳于此）

这些是原方案的"亮点"，价值认可，但属研究/重投入，**单独立项**：

| 方向 | 原方案术语 | 推迟理由 / 替代 |
|---|---|---|
| 私有化模型 | 8B 本地部署 + 微调 | v1 用 OpenAI 兼容 API 即可；有硬合规要求再上，且 vLLM/SGLang 同样提供兼容端点，切换成本低 |
| 风险行为消除 | Unlearning / ALKN 逆向学习 | 用确定性工程（强制工具+校验+HITL）替代，可证可回滚 |
| 策略自优化 | Agentic RL / GRPO 过程奖励 | 需交互环境+奖励工程+大量轨迹；先用"记忆+few-shot 案例库+prompt 迭代"达成"越用越聪明" |
| 多模态 RAG | PathFusion / PMF 图纸对齐 | 独立硬问题，早期数据不足；v1 先纯文本 RAG |
| 报价合规 Agent (Worker2) | ERP/CRM + MCP + JIT 令牌 | 业务上明确推迟；图中已预留路由出口与错误回环 |
| DeepResearch Agent (Worker3) | Web-Search + Firecrawl | 业务上明确推迟；预留 `external_research` 意图出口 |
| 评测层 | CoRE 多 agent 辩论 | v1 简化为 User Simulator + LLM-as-judge rubric（保留 SoP/RV 两个布尔指标，见 §10）|

---

## 10. 评测（轻量版，建议尽早搭）

- **User Simulator**：LLM 扮演不同画像外贸客户（急单/砍价/技术控/参数模糊）做多轮对练。
- **LLM-as-judge rubric**：对每轮输出按"信息准确性 / 参数收集效率 / 是否乱报 / 追问是否到位 / 风险是否正确升级"打分。
- 保留两个高价值**布尔判定**：
  - **SoP（产品满足度）**：推荐/确认的产品是否 100% 满足客户极端工况约束。
  - **RV（逻辑真值合理性）**：参数/报价论证是否符合工程常识，无伪逻辑幻觉。

---

## 11. 建议目录结构（实现阶段）

```
TPAgent/
├── docs/
│   └── ARCHITECTURE.md          # 本文档
├── app/
│   ├── config.py                # pydantic-settings，集中读 .env
│   ├── llm/                     # OpenAI 兼容适配层（MiniMax/DashScope 可切）
│   │   └── client.py
│   ├── state.py                 # AgentState 定义
│   ├── graph.py                 # LangGraph 装配（节点+条件边+checkpointer）
│   ├── nodes/
│   │   ├── triage.py            # Triage Router
│   │   ├── supervisor.py        # 调度
│   │   ├── worker_rag.py        # Worker1
│   │   └── hitl.py              # 人工中断
│   ├── memory/
│   │   └── mem0_store.py        # Mem0 读写 + 异步固化
│   ├── rag/
│   │   └── retriever.py         # 向量检索
│   └── tools/
│       └── params.py            # 参数 schema + 校验 + slot filling
├── eval/
│   ├── user_simulator.py
│   └── judge.py
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

---

## 12. 已确认的决策

1. **模型供应商**：✅ **MiniMax**（环境已配 `MINIMAX_API_KEY`），`base_url=https://api.minimaxi.com/v1`，默认 `MiniMax-M2`。
2. **向量库**：✅ **本地**（Chroma；不可用时降级关键词检索）。
3. **参数 schema**：✅ 由 **ETIM / ECLASS** 分类标准驱动；采购对象可为**整机或组件**。v1 用本地 JSON 映射占位，待接标准数据源。
4. **部署形态**：✅ 作为**服务**，v1 **不做对外接口**；先 **Streamlit 可视化**跑通核心图，对接由他人负责。
5. **配置**：✅ 所有可配置参数统一放 `.env`。

### 仍需业务方提供（不阻塞 v1 骨架）
- 电梯整机及各**组件**对应的 ETIM/ECLASS **分类编码**与**必填特征清单**（当前为占位）。
- 知识库初始语料（产品规格、FAQ、外贸条款、合规红线）。
