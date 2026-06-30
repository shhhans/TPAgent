"""LangGraph 全局状态：各节点通信的唯一真相源。"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict


class AgentState(TypedDict, total=False):
    # ---- 会话标识 ----
    customer_id: str
    thread_id: str
    channel: Literal["comment", "dm"]

    # ---- 消息（list[{"role","content"}]）----
    messages: Annotated[list[dict[str, str]], operator.add]

    # ---- Triage 产出 ----
    intent_category: str
    product_class: str | None  # ETIM/ECLASS 分类 key
    require_human: bool
    risk_flags: list[str]

    # ---- 报价参数槽 ----
    collected_params: dict[str, Any]
    missing_params: list[str]

    # ---- 长期记忆注入 ----
    customer_memory: dict[str, Any]

    # ---- 调度控制 ----
    next_worker: str | None
    awaiting_human: bool


def new_state(customer_id: str, thread_id: str, channel: str = "dm") -> AgentState:
    return AgentState(
        customer_id=customer_id,
        thread_id=thread_id,
        channel=channel,  # type: ignore[arg-type]
        messages=[],
        intent_category="",
        product_class=None,
        require_human=False,
        risk_flags=[],
        collected_params={},
        missing_params=[],
        customer_memory={},
        next_worker=None,
        awaiting_human=False,
    )
