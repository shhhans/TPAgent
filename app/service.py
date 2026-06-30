"""服务层：对外暴露 run_turn / finalize_session。本项目作为服务，无对外 HTTP 接口（由他人对接）。"""
from __future__ import annotations

from typing import Any

from app.graph import build_graph
from app.memory import mem0_store


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def run_turn(
    customer_id: str,
    thread_id: str,
    user_text: str,
    channel: str = "dm",
) -> dict[str, Any]:
    """处理一轮客户消息，返回 {reply, state}。"""
    graph = build_graph()
    inp = {
        "customer_id": customer_id,
        "thread_id": thread_id,
        "channel": channel,
        "messages": [{"role": "user", "content": user_text}],
    }
    result = graph.invoke(inp, _config(thread_id))
    reply = ""
    for m in reversed(result.get("messages", [])):
        if m.get("role") == "assistant":
            reply = m.get("content", "")
            break
    return {"reply": reply, "state": result}


def finalize_session(customer_id: str, thread_id: str) -> dict[str, Any]:
    """会话结束：把本线程历史固化进长期记忆。"""
    graph = build_graph()
    snap = graph.get_state(_config(thread_id))
    messages = snap.values.get("messages", []) if snap else []
    return mem0_store.finalize(customer_id, messages)
