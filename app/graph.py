"""LangGraph 装配：load_memory -> triage -> (supervisor 条件边) -> worker_rag / hitl -> END。"""
from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.nodes.hitl import hitl
from app.nodes.supervisor import route
from app.nodes.triage import load_memory, triage
from app.nodes.worker_rag import worker_rag
from app.state import AgentState


def _checkpointer():
    if settings.checkpointer == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            return SqliteSaver.from_conn_string(settings.checkpoint_db)
        except Exception:
            pass
    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()


@lru_cache
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("load_memory", load_memory)
    g.add_node("triage", triage)
    g.add_node("worker_rag", worker_rag)
    g.add_node("hitl", hitl)

    g.add_edge(START, "load_memory")
    g.add_edge("load_memory", "triage")
    # Supervisor 调度（条件边）
    g.add_conditional_edges("triage", route, {"worker_rag": "worker_rag", "hitl": "hitl"})
    g.add_edge("worker_rag", END)
    g.add_edge("hitl", END)

    return g.compile(checkpointer=_checkpointer())
