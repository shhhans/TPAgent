"""长期记忆：客户偏好/参数的跨会话存取。

- MEMORY_BACKEND=mem0 : 使用 Mem0（LLM-based CRUD）。
- MEMORY_BACKEND=json : 无依赖降级，本地 JSON 文件 + LLM 抽取事实。
两种后端对外接口一致：search() / finalize()。
"""
from __future__ import annotations

import json
import os
from typing import Any

from app.config import settings
from app.llm.client import chat_json

# ---------------- 程序记忆（静态注入，不进可写记忆库）----------------
PROCEDURAL_RULES = (
    "风控红线：1) 严禁编造承重/电压/速度等机电参数，不确定一律走人工确认。"
    "2) 制裁国家/受限地区订单需人工合规审查后才可报价。"
    "3) 最终报价/折扣需人工外贸经理审批后才能发送。"
)


# ====================== JSON 降级实现 ======================
def _json_load() -> dict:
    path = settings.memory_json_path
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _json_save(data: dict) -> None:
    os.makedirs(os.path.dirname(settings.memory_json_path) or ".", exist_ok=True)
    with open(settings.memory_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _json_search(customer_id: str) -> dict[str, Any]:
    return _json_load().get(customer_id, {"facts": [], "preferences": {}, "params": {}})


def _extract_facts(messages: list[dict[str, str]], prior: dict) -> dict:
    """用 LLM 把多轮历史清洗压缩为高纯度事实/偏好/参数。"""
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in messages if m.get("content"))
    prompt = [
        {
            "role": "system",
            "content": (
                "你是外贸客服记忆抽取器。从对话中提取关于该客户的、跨会话有价值的长期事实，"
                "剔除寒暄与一次性内容。结合已有记忆做合并/覆盖（新信息优先）。"
                "只输出 JSON: {\"facts\":[\"...\"],\"preferences\":{...},\"params\":{...}}。"
                "facts 为简短事实句（如 偏好DDP条款、目的港迪拜）；"
                "preferences 为偏好键值；params 为客户透露的稳定机电参数键值。"
            ),
        },
        {
            "role": "user",
            "content": f"已有记忆:\n{json.dumps(prior, ensure_ascii=False)}\n\n本次对话:\n{convo}",
        },
    ]
    try:
        out = chat_json(prompt)
        return {
            "facts": list(dict.fromkeys(out.get("facts", []) or [])),
            "preferences": {**prior.get("preferences", {}), **(out.get("preferences", {}) or {})},
            "params": {**prior.get("params", {}), **(out.get("params", {}) or {})},
        }
    except Exception:
        return prior


def _json_finalize(customer_id: str, messages: list[dict[str, str]]) -> dict:
    data = _json_load()
    prior = data.get(customer_id, {"facts": [], "preferences": {}, "params": {}})
    updated = _extract_facts(messages, prior)
    data[customer_id] = updated
    _json_save(data)
    return updated


# ====================== Mem0 实现（best-effort）======================
_mem0 = None
_mem0_patched = False


def _patch_mem0_minimax_think() -> None:
    """给 mem0 的 MiniMax provider 打补丁：剥离 <think>…</think>，否则推理模型输出会破坏 mem0 的 JSON 解析。"""
    global _mem0_patched
    if _mem0_patched:
        return
    import re as _re

    from mem0.llms.minimax import MiniMaxLLM

    _orig = MiniMaxLLM._parse_response

    def _patched(self, response, tools):
        out = _orig(self, response, tools)
        if isinstance(out, str):
            return _re.sub(r"<think>.*?</think>", "", out, flags=_re.S).strip()
        return out

    MiniMaxLLM._parse_response = _patched
    _mem0_patched = True


def _get_mem0():
    global _mem0
    if _mem0 is not None:
        return _mem0
    from mem0 import Memory

    _patch_mem0_minimax_think()

    cfg = {
        # 内部记忆抽取/CRUD 用非推理模型，输出干净 JSON（避免 MiniMax-M2 的 <think> 噪声）
        "llm": {
            "provider": "minimax",
            "config": {
                "model": settings.mem0_llm_model,
                "api_key": settings.minimax_api_key,
                "minimax_base_url": settings.minimax_base_url,
                "temperature": 0.1,
            },
        },
        # 本地 embedding，不依赖外部 embedding API
        "embedder": {
            "provider": "huggingface",
            "config": {"model": settings.embedding_model},
        },
        # 本地向量库
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": settings.mem0_collection,
                "path": settings.mem0_vector_path,
            },
        },
        # 外贸场景定制：保留客户原文语言，聚焦贸易/机电要素
        "custom_instructions": (
            "你在为电梯外贸售前客服维护客户长期记忆。"
            "请用客户原文所用语言记录，事实要简短。"
            "重点记录：贸易条款偏好(FOB/CIF/DDP)、目的国/港口、机电参数(电压/载重/速度/层站)、"
            "采购对象(整机或具体组件)、关键历史事件(人工接管/清关纠纷)。忽略寒暄与一次性闲聊。"
        ),
    }
    _mem0 = Memory.from_config(cfg)
    return _mem0


def _mem0_search(customer_id: str) -> dict[str, Any]:
    m = _get_mem0()
    # mem0 2.x: 用 filters 传 user_id
    res = m.get_all(filters={"user_id": customer_id})
    items = res.get("results", res) if isinstance(res, dict) else res
    facts = [it.get("memory", "") for it in (items or []) if it.get("memory")]
    return {"facts": facts, "preferences": {}, "params": {}}


def _mem0_finalize(customer_id: str, messages: list[dict[str, str]]) -> dict:
    m = _get_mem0()
    m.add(messages, user_id=customer_id)
    return _mem0_search(customer_id)


# ====================== 对外统一接口 ======================
def search(customer_id: str) -> dict[str, Any]:
    """召回客户长期记忆，失败降级为空记忆。"""
    try:
        if settings.memory_backend == "mem0":
            return _mem0_search(customer_id)
        return _json_search(customer_id)
    except Exception:
        return {"facts": [], "preferences": {}, "params": {}}


def finalize(customer_id: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    """会话结束后固化记忆。失败降级为 JSON。"""
    try:
        if settings.memory_backend == "mem0":
            return _mem0_finalize(customer_id, messages)
        return _json_finalize(customer_id, messages)
    except Exception:
        return _json_finalize(customer_id, messages)


def format_memory(mem: dict[str, Any]) -> str:
    parts = []
    if mem.get("facts"):
        parts.append("已知事实: " + "；".join(mem["facts"]))
    if mem.get("preferences"):
        parts.append("偏好: " + json.dumps(mem["preferences"], ensure_ascii=False))
    if mem.get("params"):
        parts.append("历史参数: " + json.dumps(mem["params"], ensure_ascii=False))
    return "\n".join(parts) if parts else "（无历史记忆，可能是新客户）"
