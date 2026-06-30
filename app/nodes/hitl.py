"""HITL 人工中断：高风险/越界/需人工报价时挂起，等外贸经理审批。

v1 以状态标记 + 占位回复体现；生产可换为 LangGraph interrupt() + checkpointer resume。
"""
from __future__ import annotations

from app.state import AgentState

_REASON_TEXT = {
    "ready_to_quote": "报价需人工外贸经理确认",
    "external_research": "需外部资料核查，已转人工",
    "human_required": "已转人工客服处理",
}


def hitl(state: AgentState) -> dict:
    intent = state.get("intent_category", "")
    flags = state.get("risk_flags", [])
    reason = _REASON_TEXT.get(intent, "已转人工处理")
    if flags:
        reason += f"（风险标记：{', '.join(flags)}）"

    # 面向客户的占位安抚语；真实回复由人工补充
    reply = f"您的需求已收到，{reason}，我们的外贸经理会尽快与您跟进。"
    return {
        "awaiting_human": True,
        "messages": [{"role": "assistant", "content": reply}],
    }
