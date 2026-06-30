"""Supervisor 调度：读 Triage 结构化意图，决定走哪条边。本身无业务 API。"""
from __future__ import annotations

from app.state import AgentState


def route(state: AgentState) -> str:
    """LangGraph 条件边：返回下一节点名。"""
    if state.get("require_human") or state.get("risk_flags"):
        return "hitl"

    intent = state.get("intent_category", "")
    if intent in ("product_qa", "needs_more_info"):
        return "worker_rag"
    if intent == "ready_to_quote":
        # v1 报价 Agent 未接入 -> 转人工报价
        return "hitl"
    if intent == "external_research":
        # v1 DeepResearch 未接入 -> 转人工
        return "hitl"
    # chitchat / 其它
    return "worker_rag"
