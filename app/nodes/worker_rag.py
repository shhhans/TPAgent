"""Worker1：RAG + 参数收集（售前客服）。答技术问题 + 缺参追问，并生成客户可见回复。"""
from __future__ import annotations

from app.llm.client import chat
from app.memory.mem0_store import PROCEDURAL_RULES
from app.nodes.triage import _last_user_text
from app.rag import retriever
from app.state import AgentState
from app.tools import params


def worker_rag(state: AgentState) -> dict:
    user_text = _last_user_text(state["messages"])
    hits = retriever.rag_search(user_text)
    context = retriever.format_context(hits)

    missing = state.get("missing_params", [])
    product_class = state.get("product_class")
    intent = state.get("intent_category", "")

    ask_hint = ""
    if intent == "needs_more_info" and missing:
        # 一次只追问 1~2 个关键参数
        focus = missing[:2]
        ask_hint = (
            f"客户有报价意向但还缺这些必填参数：{params.describe_missing(focus)}。"
            "请在回答后，自然地、一次只追问这 1~2 个参数。"
        )

    system = (
        "你是专业、友好的电梯外贸售前客服，可多语言回复（与客户语言一致）。\n"
        f"风控红线（必须遵守）：{PROCEDURAL_RULES}\n"
        "只能依据【知识库】内容回答规格/参数/价格相关问题；知识库没有的，"
        "不要编造，明确说需进一步确认。回复简洁、像真人客服。"
    )
    user = (
        f"【知识库】\n{context}\n\n"
        f"【当前产品分类】{product_class or '未识别'}\n"
        f"【已收集参数】{state.get('collected_params', {})}\n"
        f"【追问指引】{ask_hint or '无需追问'}\n\n"
        f"【客户消息】{user_text}"
    )

    reply = chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
    return {"messages": [{"role": "assistant", "content": reply}]}
