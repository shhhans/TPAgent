# TPAgent — 海外电商智能客服 Agent

面向**电梯外贸**的海外社媒售前智能客服。客户通过国际社媒的**评论 / 私信**沟通，
Agent 在这种长周期、异步、多语言的触点中提供售前服务：答疑、收集报价参数、风险兜底。

## 技术基座

- **编排**：LangGraph 显式状态机（`load_memory → triage → supervisor 条件边 → worker_rag / hitl`）
- **大脑**：MiniMax（OpenAI 兼容 API，`base_url=https://api.minimaxi.com/v1`，默认 `MiniMax-M2`，可切 `MiniMax-M3`）
- **长期记忆**：Mem0（不可用时降级本地 JSON，接口一致）—— 客户参数/偏好跨会话召回
- **短期记忆**：LangGraph Checkpointer
- **知识库**：本地向量检索 RAG（Chroma，不可用时降级关键词检索）
- **参数 schema**：ETIM / ECLASS 分类驱动，采购对象可为整机或组件
- **人机协同**：HITL 人工中断（高风险 / 需人工报价时挂起）

> 所有可配置参数统一在 `.env`（见 `.env.example`），由 `app/config.py` 加载。

## v1 范围

Triage 分诊路由 + Supervisor 调度 + Worker1（RAG + 参数收集）+ HITL + Mem0 长期记忆。
报价合规 Agent、DeepResearch Agent、私有化 8B 微调 / RL / 多模态 RAG 列入后期路线图（见设计文档第 9 节）。

本项目作为**服务**存在，社媒/中台对接由他人负责，v1 不做对外接口。

## 快速开始

```bash
# 1) 安装依赖（默认含 mem0 + 本地向量库 + 本地 embedding）
pip install -r requirements.txt
#   若系统自带 PyYAML 导致卸载失败：
#   pip install --ignore-installed PyYAML -r requirements.txt

# 2) 配置（环境变量已有 MINIMAX_API_KEY 则可跳过；否则复制并填写）
cp .env.example .env
#   想免重依赖快速跑通可在 .env 设 MEMORY_BACKEND=json、VECTOR_BACKEND=keyword

# 3a) 命令行烟测
python -m app.cli

# 3b) Streamlit 可视化（聊天 + 实时 State 面板）
streamlit run app/ui/streamlit_app.py

# 3c) 仿真评测（LLM 仿真客户 ↔ Agent 多轮对打 + 判分）
MEMORY_BACKEND=json VECTOR_BACKEND=keyword python -m app.eval.run_eval --seeds 1-10
```

## 目录结构

```
app/
├── config.py            # .env 统一配置
├── llm/client.py        # MiniMax (OpenAI 兼容) 适配 + JSON/think 处理
├── state.py             # AgentState
├── graph.py             # LangGraph 装配
├── service.py           # run_turn / finalize_session（服务入口）
├── cli.py               # 命令行烟测
├── nodes/               # triage / supervisor / worker_rag / hitl
├── memory/mem0_store.py # 长期记忆（Mem0 + JSON 降级）
├── rag/retriever.py     # 本地向量检索（Chroma + 关键词降级）
├── tools/params.py      # ETIM/ECLASS 参数槽
├── ui/streamlit_app.py  # 可视化
└── eval/                # 仿真评测：personas / sim_customer / harness / judges / run_eval
data/
├── kb/                  # 知识库语料（占位）
└── param_schema.json    # ETIM/ECLASS 参数 schema（占位）
docs/ARCHITECTURE.md     # 完整设计文档
docs/EVAL.md             # 仿真评测层说明
```

完整设计见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。
