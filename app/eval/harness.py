"""仿真闭环：仿真客户 ↔ 被测 Agent 多轮对打，直到达成/挂起/超轮。

不写入长期记忆（不调用 finalize_session），因此评测不污染真实记忆库。
每个 seed 用独立 customer_id / thread_id 隔离。
"""
from __future__ import annotations

from typing import Any

from app.config import settings
from app.eval import sim_customer
from app.eval.personas import Persona, build_persona
from app.service import run_turn


def _snapshot(state: dict) -> dict:
    """从 Agent State 中抽取评测关心的字段。"""
    return {
        "intent_category": state.get("intent_category"),
        "product_class": state.get("product_class"),
        "collected_params": dict(state.get("collected_params", {})),
        "missing_params": list(state.get("missing_params", [])),
        "require_human": bool(state.get("require_human", False)),
        "risk_flags": list(state.get("risk_flags", [])),
        "awaiting_human": bool(state.get("awaiting_human", False)),
    }


def run_episode(persona: Persona, max_turns: int | None = None) -> dict[str, Any]:
    """跑一条画像的完整仿真对话，返回轨迹与终态。"""
    max_turns = max_turns or settings.eval_max_turns
    customer_id = f"sim_{persona.seed}"
    thread_id = f"sim_{persona.seed}"

    buyer_msg = sim_customer.open_message(persona)
    # 仿真客户视角历史：buyer 自己的话是 assistant，seller 的话是 user
    sim_history: list[dict[str, str]] = [{"role": "assistant", "content": buyer_msg}]

    transcript: list[dict[str, Any]] = []
    final_state: dict[str, Any] = {}
    ended_by = "max_turns"

    for turn in range(max_turns):
        result = run_turn(customer_id, thread_id, buyer_msg, channel=persona.channel)
        reply = result.get("reply", "")
        final_state = _snapshot(result.get("state", {}))
        transcript.append({"turn": turn, "buyer": buyer_msg, "seller": reply, "state": final_state})

        # Agent 侧终止：已挂起人工（报价就绪转人工 or 风险拦截）
        if final_state["awaiting_human"]:
            ended_by = "agent_handoff"
            break

        sim_history.append({"role": "user", "content": reply})
        nxt = sim_customer.next_message(persona, sim_history)
        buyer_msg = nxt["message"]
        if nxt["end"] or not buyer_msg:
            ended_by = "customer_end"
            transcript[-1]["customer_end_note"] = nxt.get("note", "")
            break
        sim_history.append({"role": "assistant", "content": buyer_msg})

    return {
        "seed": persona.seed,
        "persona": persona,
        "transcript": transcript,
        "final_state": final_state,
        "ended_by": ended_by,
        "turns": len(transcript),
    }


def run_seed(seed: int, max_turns: int | None = None) -> dict[str, Any]:
    return run_episode(build_persona(seed), max_turns)
