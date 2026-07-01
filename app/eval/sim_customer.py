"""LLM 仿真客户：按画像扮演真实买家，与被测 Agent 多轮对话。

关键约束：仿真客户只"知道"自己的画像与隐藏需求，按沟通节奏逐步透露参数，
用画像指定的语言书写，保持角色性格，不会一次把所有参数报全——逼被测 Agent
主动询问、收集，才测得出真实能力。
"""
from __future__ import annotations

from app.eval.personas import Persona
from app.llm.client import chat_json
from app.config import settings

_STYLE_HINT = {
    "terse": "Keep messages short, a few words to one sentence.",
    "chatty": "Be friendly and a bit talkative.",
    "demanding": "Be pushy about price and delivery time.",
    "skeptical": "Question quality/certifications, ask for proof.",
    "bargainer": "Keep pressing for discounts and better terms.",
}
_KNOWLEDGE_HINT = {
    "novice": "You are not technical; describe needs in plain words, sometimes vaguely.",
    "intermediate": "You know the basics and some spec terms.",
    "expert": "You are technical; use precise industry terms.",
}
_CLARITY_HINT = {
    "vague": "Your requirements are fuzzy at first; you firm them up only when asked.",
    "moderate": "You give partial specs, need some prompting for the rest.",
    "precise": "You know exactly what you want and answer clearly when asked.",
}


def _system_prompt(persona: Persona) -> str:
    return (
        "You are role-playing a real overseas buyer contacting an elevator exporter's "
        "pre-sales agent through social media. Stay fully in character.\n\n"
        f"{persona.needs_brief()}\n\n"
        f"Personality: {_STYLE_HINT.get(persona.style, '')} "
        f"{_KNOWLEDGE_HINT.get(persona.knowledge, '')} "
        f"{_CLARITY_HINT.get(persona.clarity, '')}\n\n"
        f"IMPORTANT rules:\n"
        f"- Write EVERY message in {persona.lang_name} (the buyer's language).\n"
        f"- Reveal at most 1-2 requirements per message, and only what the seller asks for; "
        f"do NOT list all your specs at once.\n"
        f"- If the seller has gathered your key specs and offers to prepare a quote / hand you "
        f"to a sales rep, you may end the conversation.\n"
        f"- If you get stuck or frustrated, you may also end.\n"
        f"- You are the BUYER, never act as the seller.\n\n"
        'Respond ONLY as a JSON object: {"message": "<your next message to the seller>", '
        '"end": <true|false>, "note": "<brief reason if ending, else empty>"}'
    )


def open_message(persona: Persona) -> str:
    """仿真客户的开场白。"""
    out = chat_json(
        [
            {"role": "system", "content": _system_prompt(persona)},
            {
                "role": "user",
                "content": (
                    "Start the conversation. Send your FIRST short message to the seller "
                    "expressing interest. Do not reveal all specs yet."
                ),
            },
        ],
        temperature=settings.eval_sim_temperature,
    )
    return (out.get("message") or "").strip() or "Hi, I'm interested in your elevators."


def next_message(persona: Persona, history: list[dict[str, str]]) -> dict:
    """给定对话历史（buyer=user / seller=assistant 视角），产出仿真客户下一条消息。

    返回 {"message": str, "end": bool, "note": str}。
    """
    # history 里 seller 的话对仿真客户而言是 user 输入，buyer 自己的话是 assistant。
    convo = [{"role": "system", "content": _system_prompt(persona)}]
    convo.extend(history)
    convo.append(
        {
            "role": "user",
            "content": "Reply to the seller's latest message. Return the JSON object.",
        }
    )
    out = chat_json(convo, temperature=settings.eval_sim_temperature)
    return {
        "message": (out.get("message") or "").strip(),
        "end": bool(out.get("end", False)),
        "note": (out.get("note") or "").strip(),
    }
