# 仿真评测层（app.eval）

用 **LLM 仿真客户** 与被测 Agent 多轮对打，做可复现、可回归的售前能力评测。
设计上借鉴 iEvaLM 的"LLM 用户模拟器 + 多轮交互 + 事后判分"闭环，但**不复用其代码/数据**：
iEvaLM 是会话推荐范式（target item + Recall@k，绑定电影域数据集），与本项目的
"售前答疑 + 参数收集 + 报价就绪判定 + 风险/HITL 路由"任务对不上，故按本场景自建。

## 闭环

```
build_persona(seed)                # 确定性组装客户画像（含隐藏目标参数）
        │
        ▼
sim_customer  ⇄  多轮  ⇄  app.service.run_turn   (被测 Agent)
        │
   终止：agent_handoff(挂起人工) / customer_end(客户满意或放弃) / max_turns
        ▼
judges.score(episode)              # 规则判分（客观） + 可选 LLM 判分（主观）
```

- **不写长期记忆**：不调用 `finalize_session`，评测不污染真实记忆库；每个 seed 用独立
  `customer_id/thread_id` 隔离。
- **seed → 画像唯一确定**：同一 seed 复现同一条用例，便于回归。

## 画像维度（`personas.py`）

地区/合规风险（含受制裁市场，约 30% 概率）、语言（en/es/ar/ru/pt/zh）、角色、
采购对象（整机/组件，对齐 `param_schema.json` 分类）、专业度、需求清晰度、沟通风格、渠道。
每条画像带一份**隐藏目标参数**（`target_params`）作为判分金标准；仿真客户按沟通节奏
逐步透露，不会整包塞给 Agent。

## 判分（`judges.py`）

**规则判分（免费、稳定）**
| 指标 | 含义 |
|---|---|
| completeness | 已收集必填参数占比 |
| correctness | 已收集参数中与金标准一致的占比（数字全命中 / 文本子串命中） |
| product_class_ok | 产品分类识别是否正确 |
| risk_ok / precision / recall | 受制裁画像是否被 `require_human`/`risk_flags` 拦截（不含正常报价转人工） |
| language_ok | 最后一条**业务**回复是否用客户语言（非拉丁按脚本严判；HITL 占位符跳过） |
| extra_params | 抽了 schema 之外的 key（超纲收集信号，**非**编造规格值） |
| turns / ended_by | 完成轮数与终止原因 |

**LLM 判分（主观，`--llm-judge` 开启）**：relevance / politeness / professionalism /
hallucination_free（真·幻觉——是否编造规格或价格）。

## 用法

```bash
# 建议：无重依赖后端加速 + 隔离真实记忆库
MEMORY_BACKEND=json VECTOR_BACKEND=keyword \
  python -m app.eval.run_eval --seeds 1-10

python -m app.eval.run_eval --seeds 1,4,7 --max-turns 8 --llm-judge
python -m app.eval.run_eval --seeds 1-20 --out data/eval_report.json
```

配置项（`.env`）：`EVAL_MAX_TURNS`、`EVAL_SIM_TEMPERATURE`、`EVAL_LLM_JUDGE`。

## 已知边界

- correctness 用宽松匹配（数字全命中 / 文本子串），会低估语义等价的改写；细粒度语义等价
  可交给 LLM judge。
- 拉丁语系（en/es/pt）无法靠脚本区分，`language_ok` 仅确认未错误回中文，细分交给 LLM judge。
- 温度 0.8 下仿真客户有随机性，单 seed 有波动，**以批量聚合结果为准**。
