# TPAgent — 海外电商智能客服 Agent

面向**电梯外贸**的海外社媒售前智能客服。客户通过国际社媒的**评论 / 私信**沟通，
Agent 在这种长周期、异步、多语言的触点中提供售前服务：答疑、收集报价参数、风险兜底。

## 技术基座

- **编排**：LangGraph 显式状态机（Triage Router → Supervisor → Worker → HITL）
- **大脑**：OpenAI 兼容 API（MiniMax / DashScope 通义千问，可切换）
- **长期记忆**：Mem0（客户参数表 + 偏好，跨会话召回）
- **短期记忆**：LangGraph Checkpointer
- **知识库**：向量检索 RAG（v1 纯文本）
- **人机协同**：LangGraph 原生 HITL 中断

## v1 范围

Triage 分诊路由 + Supervisor 调度 + Worker1（RAG + 参数收集）+ HITL 人工中断 + Mem0 长期记忆。

报价合规 Agent、DeepResearch Agent、私有化 8B 微调 / RL / 多模态 RAG 等列入后期路线图。

## 文档

完整设计见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。
