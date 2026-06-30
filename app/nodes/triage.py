"""Triage Router：分诊路由。读记忆 + 推演 + 输出结构化 JSON 决策（不直接对客户说话）。"""
from __future__ import annotations

import json

from app.llm.client import chat_json
from app.memory import mem0_store
from app.state import AgentState
from app.tools import params

INTENTS = [
    "product_qa",        # 纯产品技术/规格问题 -> Worker1(RAG)
    "needs_more_info",   # 有报价意图但参数不全 -> Worker1(追问)
    "ready_to_quote",    # 参数齐全、明确询价 -> (v1) HITL 人工报价
    "external_research", # 私库覆盖不了的外部长尾 -> (v1) HITL
    "chitchat",          # 寒暄/无关
    "human_required",    # 投诉/纠纷/高风险 -> HITL
]


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def load_memory(state: AgentState) -> dict:
    """图入口：召回客户长期记忆注入 State。"""
    mem = mem0_store.search(state["customer_id"])
    # 历史稳定参数预填入 collected_params（不覆盖本会话已有）
    collected = params.update_params(mem.get("params", {}), state.get("collected_params", {}))
    return {"customer_memory": mem, "collected_params": collected}


def triage(state: AgentState) -> dict:
    user_text = _last_user_text(state["messages"])
    classes = params.list_classes()
    class_desc = "\n".join(
        f"- {k}: {v.get('label')}，必填特征 {v.get('required')}" for k, v in classes.items()
    )
    mem_str = mem0_store.format_memory(state.get("customer_memory", {}))

    system = (
        "你是电梯外贸售前的分诊路由器。先推演，再输出结构化 JSON 决策，不要直接回复客户。\n"
        f"可选意图(intent_category): {INTENTS}\n"
        "可选产品分类(product_class，按 ETIM/ECLASS，采购对象可为整机或组件)：\n"
        f"{class_desc}\n"
        "规则：客户想报价但缺必填特征 -> needs_more_info；必填齐全且明确询价 -> ready_to_quote；"
        "纯技术问题 -> product_qa；涉及制裁国家/合规风险/投诉纠纷 -> 设 require_human=true 并填 risk_flags。\n"
        "只输出 JSON: {\"thought_process\":\"...\",\"intent_category\":\"...\","
        "\"product_class\":\"key或null\",\"extracted_parameters\":{},\"require_human\":false,\"risk_flags\":[]}"
    )
    user = (
        f"客户长期记忆:\n{mem_str}\n\n"
        f"本会话已收集参数: {json.dumps(state.get('collected_params', {}), ensure_ascii=False)}\n\n"
        f"客户最新消息:\n{user_text}"
    )

    try:
        out = chat_json([{"role": "system", "content": system}, {"role": "user", "content": user}])
    except Exception:
        # 解析彻底失败 -> 安全降级人工
        return {"intent_category": "human_required", "require_human": True, "risk_flags": ["triage_parse_failed"]}

    intent = out.get("intent_category", "product_qa")
    if intent not in INTENTS:
        intent = "product_qa"
    product_class = out.get("product_class") or None
    if product_class not in classes:
        product_class = None

    collected = params.update_params(state.get("collected_params", {}), out.get("extracted_parameters", {}))
    missing = params.compute_missing(product_class, collected)

    # 若意图为询价但其实缺参，纠偏为 needs_more_info
    if intent == "ready_to_quote" and (not product_class or missing):
        intent = "needs_more_info"

    return {
        "intent_category": intent,
        "product_class": product_class,
        "collected_params": collected,
        "missing_params": missing,
        "require_human": bool(out.get("require_human", False)),
        "risk_flags": out.get("risk_flags", []) or [],
    }
