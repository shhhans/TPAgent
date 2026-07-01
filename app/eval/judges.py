"""判分层：规则判分（客观、免费、稳定）+ 可选 LLM 判分（主观）。

规则判分覆盖：参数完整率、参数正确率(vs 隐藏 ground-truth)、幻觉参数、
产品分类是否识别正确、风险路由是否正确、回复语言是否匹配。
LLM 判分覆盖：得体度/相关性/专业可信度（默认关闭，用 --llm-judge 开启）。
"""
from __future__ import annotations

import re
from typing import Any

from app.eval.personas import Persona
from app.llm.client import chat_json
from app.tools import params as param_tool

# ---- 值归一化与匹配 ----

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _value_match(gt: str, got: Any) -> bool:
    """判断 Agent 抽取值 got 是否与金标准 gt 一致（宽松但有意义）。"""
    if got in (None, "", []):
        return False
    got_s = str(got)
    gt_nums = _NUM_RE.findall(str(gt))
    got_nums = _NUM_RE.findall(got_s)
    if gt_nums:
        # 金标准含数字：要求其全部数字都出现在抽取值中
        return set(gt_nums) <= set(got_nums)
    # 纯文本：归一化后子串命中，或有效字母词命中
    gn, on = _norm(gt), _norm(got_s)
    if gn and gn in on:
        return True
    for tok in re.split(r"[\s,/]+", str(gt).lower()):
        if len(tok) >= 3 and _norm(tok) and _norm(tok) in on:
            return True
    return False


# ---- 语言脚本检测 ----

def _has_cjk(t: str) -> bool:
    return bool(re.search(r"[一-鿿]", t))


def _has_arabic(t: str) -> bool:
    return bool(re.search(r"[؀-ۿ]", t))


def _has_cyrillic(t: str) -> bool:
    return bool(re.search(r"[Ѐ-ӿ]", t))


def _language_ok(lang: str, reply: str) -> bool | None:
    """回复语言是否匹配客户语言。

    非拉丁语系(zh/ar/ru)按脚本严格判定；拉丁语系(en/es/pt)无法靠脚本区分，
    仅确认没有错误地用中文回复（返回 True/False），细分交由 LLM 判分。
    """
    if not reply:
        return None
    if lang == "zh":
        return _has_cjk(reply)
    if lang == "ar":
        return _has_arabic(reply)
    if lang == "ru":
        return _has_cyrillic(reply)
    # 拉丁语系：至少不该默认回中文
    return not _has_cjk(reply)


# ---- 规则判分 ----

def score_rules(episode: dict[str, Any]) -> dict[str, Any]:
    persona: Persona = episode["persona"]
    state = episode["final_state"]
    transcript = episode["transcript"]

    target = persona.target_params
    collected = state.get("collected_params", {})
    required = list(target.keys())

    # 完整率 / 正确率
    present = [k for k in required if collected.get(k) not in (None, "", [])]
    correct = [k for k in present if _value_match(target[k], collected[k])]
    completeness = len(present) / len(required) if required else 1.0
    correctness = len(correct) / len(present) if present else 0.0

    # 超纲收集：抽了目标 schema 之外的 key（LLM 自造键/过度抽取）。
    # 注意：这不等于"编造规格值"，真·幻觉由 LLM judge 的 hallucination_free 判定。
    extra = [k for k in collected if k not in required and collected.get(k) not in (None, "", [])]

    # 产品分类识别
    product_ok = state.get("product_class") == persona.product_class

    # 风险路由：受制裁画像应被风险信号拦截。
    # 只看 require_human/risk_flags——awaiting_human 在"报价就绪转人工"时也会置位，
    # 那属于正常业务转交，不能算成风险误报。
    risk_routed = bool(state.get("require_human") or state.get("risk_flags"))
    expected_human = persona.expects_human
    risk_ok = risk_routed == expected_human

    # 语言匹配：取最后一条"业务"回复（跳过 HITL 转人工占位符——其语言 v1 尚未跟随客户）
    last_reply = ""
    for t in reversed(transcript):
        if not t["state"].get("awaiting_human"):
            last_reply = t["seller"]
            break
    if not last_reply and transcript:
        last_reply = transcript[-1]["seller"]
    lang_ok = _language_ok(persona.lang, last_reply)

    return {
        "completeness": round(completeness, 3),
        "correctness": round(correctness, 3),
        "correct_params": correct,
        "missing_or_wrong": [k for k in required if k not in correct],
        "extra_params": extra,
        "product_class_ok": product_ok,
        "risk_ok": risk_ok,
        "expected_human": expected_human,
        "risk_routed": risk_routed,
        "language_ok": lang_ok,
        "turns": episode.get("turns"),
        "ended_by": episode.get("ended_by"),
    }


# ---- LLM 判分（可选） ----

def score_llm(episode: dict[str, Any]) -> dict[str, Any]:
    """主观维度打分（1-5）。仅在开启时调用，会消耗一次 API。"""
    persona: Persona = episode["persona"]
    convo = "\n".join(
        f"BUYER: {t['buyer']}\nSELLER: {t['seller']}" for t in episode["transcript"]
    )
    system = (
        "You are a strict QA reviewer for a pre-sales customer-service agent (elevator export). "
        "Score the SELLER's performance on the conversation. "
        "Return ONLY JSON: {\"relevance\":1-5, \"politeness\":1-5, \"professionalism\":1-5, "
        "\"hallucination_free\":1-5, \"comment\":\"one short sentence\"}. "
        "hallucination_free=5 means the seller never invented specs/prices."
    )
    user = (
        f"Buyer speaks {persona.lang_name}, wants {persona.product_label}.\n\n"
        f"Conversation:\n{convo}"
    )
    try:
        out = chat_json([{"role": "system", "content": system}, {"role": "user", "content": user}])
    except Exception as e:  # noqa: BLE001
        return {"llm_error": str(e)[:120]}
    keep = {}
    for k in ("relevance", "politeness", "professionalism", "hallucination_free"):
        try:
            keep[k] = int(out.get(k))
        except (TypeError, ValueError):
            keep[k] = None
    keep["comment"] = (out.get("comment") or "").strip()
    return keep


def score(episode: dict[str, Any], use_llm: bool = False) -> dict[str, Any]:
    result = {"seed": episode["seed"], "persona": episode["persona"].summary()}
    result["rules"] = score_rules(episode)
    if use_llm:
        result["llm"] = score_llm(episode)
    return result
