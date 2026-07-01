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

**确定性自然度（`naturalness.py`，免费，总是计算）**
把"AI 味"里能规则抓的先抓（依据文献：风格/排版偏见 > 冗长偏见）：
| 指标 | 含义 |
|---|---|
| avg_markdown_density | 排版脚手架强度（bullet/编号/表格/加粗/emoji）——评论/私信里堆 Markdown 本身不自然 |
| mechanical_transition_rate | 机械过渡词/客套八股命中率（首先/其次/最后、firstly/in conclusion…） |
| meta_leak_count | 思维/元评论泄漏（对应 issue #2，可当回归探针） |
| over_length_rate | 超 `EVAL_MAX_REPLY_CHARS` 的冗长回复占比 |
| avg_ai_tone_penalty | 上述综合的 0..1 惩罚 |

**LLM 主观判分（`--llm-judge`，四镜头 rubric）**
一次结构化调用（Combined Budget：CoT + rubric），1-3 低基数打分 + 闭合"是否像真人"判定；
**候选送判前先去 Markdown**（否则裁判反而偏爱排版）；**Temp=0** 保可复现；裁判模型可用
`EVAL_JUDGE_MODEL` 覆盖为异构模型以消除自我偏好。四镜头：
fluency(语言学家/AI 味探测) · empathy(普通用户/情感) · coherence(语境连贯) ·
conciseness(HCI/信息密度) · hallucination_free(领域专家/不编造)。

> **两张正交记分卡**：task success 与 naturalness 互不替代——自然度不能盖过参数正确性/风险路由
> 这条红线（复合排名见 `docs/EVAL_ROADMAP.md` #8）。

## 裁判验收（MVVP · `validate_judge.py`）

LLM 裁判"上岗前"必须验收，别把"稳定"当可信（一致性-偏见悖论）：

```bash
python -m app.eval.validate_judge                 # 内置演示集（也是冒烟测试）
python -m app.eval.validate_judge --labeled data/eval_labeled.jsonl
```
报告：**Cohen's κ**（而非原始匹配率——后者相对 κ 通常注水 33-41 个点）、**AB 位置交换**的位置偏见、
在**不同标签分布子集**上的交叉验证。κ≥0.6 方可信任其自然度判分。
标注 JSONL 每行 `{"text","natural":bool,"lang","group"}`（真人语料到位后即插即用）。

## 用法

```bash
# 建议：无重依赖后端加速 + 隔离真实记忆库
MEMORY_BACKEND=json VECTOR_BACKEND=keyword \
  python -m app.eval.run_eval --seeds 1-10

python -m app.eval.run_eval --seeds 1,4,7 --max-turns 8 --llm-judge
python -m app.eval.run_eval --seeds 1-20 --out data/eval_report.json
```

配置项（`.env`）：`EVAL_MAX_TURNS`、`EVAL_SIM_TEMPERATURE`、`EVAL_LLM_JUDGE`、
`EVAL_MAX_REPLY_CHARS`（冗长阈值）。

**异构裁判 / 陪审团**（消除"MiniMax 自己判自己"的自我偏好；配 `DASHSCOPE_API_KEY` 即启用）：
- 换裁判：`EVAL_JUDGE_PROVIDER=dashscope`（裁判用 Qwen，≠ 生成模型）。
- 陪审团：`EVAL_JURY=true`（MiniMax + Qwen 各判一次，二元多数票 / 分值取均值，保留每席明细）。
- 先验收再信任：`python -m app.eval.validate_judge --provider dashscope` 对 Qwen 裁判跑 MVVP。
未配 key 时自动回退单 MiniMax 裁判，不影响运行。

## 已知边界

- correctness 用宽松匹配（数字全命中 / 文本子串），会低估语义等价的改写；细粒度语义等价
  可交给 LLM judge。
- 拉丁语系（en/es/pt）无法靠脚本区分，`language_ok` 仅确认未错误回中文，细分交给 LLM judge。
- 温度 0.8 下仿真客户有随机性，单 seed 有波动，**以批量聚合结果为准**。
- 默认裁判与大脑同为 MiniMax（"自己判自己"，会偏袒低困惑度的八股文），且 MiniMax 不返回
  logprobs（G-EVAL 连续打分不可用）。缓解与后续项见 `docs/EVAL_ROADMAP.md`（真人校准、
  异构陪审团、复合红线记分卡等）。
