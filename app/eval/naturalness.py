"""自然度的确定性(规则)维度：不花 API 就能抓的"AI 味"。

依据文献要点：
- 风格/排版偏见 > 冗长偏见：客服（评论/私信）里堆 Markdown 本身就是不自然，需惩罚；
  且送进 LLM 裁判前必须先去格式化，否则裁判反而奖励 Markdown。
- 冗余/过度思考：把"思维/元评论泄漏"(issue #2)做成可验证的负向惩罚项。
这些轴与 LLM 主观判分互补：能规则抓的绝不花钱让模型抓。
"""
from __future__ import annotations

import re
from typing import Any

from app.config import settings

# ---- Markdown 脚手架 ----

_MD_BOLD = re.compile(r"\*\*.+?\*\*|__.+?__")
_MD_HEADER = re.compile(r"^\s{0,3}#{1,6}\s", re.M)
_MD_BULLET = re.compile(r"^\s*[-*•·✅✓☑▪●]\s+", re.M)
_MD_NUMBERED = re.compile(r"^\s*\d+[\.\)]\s+", re.M)
_MD_TABLE = re.compile(r"^\s*\|.*\|\s*$", re.M)
_MD_FENCE = re.compile(r"```")
_EMOJI = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")


def strip_markdown(text: str) -> str:
    """去 Markdown 归一化：送 LLM 裁判前用，避免裁判偏爱排版。"""
    t = text
    t = _MD_FENCE.sub("", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"__(.+?)__", r"\1", t)
    t = re.sub(r"[*_`#>]", "", t)
    t = re.sub(r"^\s*\|.*$", "", t, flags=re.M)          # 表格行
    t = re.sub(r"^\s*[-*•·✅✓☑▪●]\s+", "", t, flags=re.M)  # bullet 记号
    t = re.sub(r"^\s*\d+[\.\)]\s+", "", t, flags=re.M)     # 编号
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def markdown_density(text: str) -> float:
    """0..1：排版脚手架的强度（bullet/编号/表格/标题/加粗/emoji 命中密度）。"""
    if not text.strip():
        return 0.0
    lines = [ln for ln in text.splitlines() if ln.strip()]
    n = max(len(lines), 1)
    scaffold = (
        len(_MD_BULLET.findall(text))
        + len(_MD_NUMBERED.findall(text))
        + len(_MD_TABLE.findall(text))
        + len(_MD_HEADER.findall(text))
    )
    marks = scaffold + len(_MD_BOLD.findall(text)) + len(_EMOJI.findall(text)) + len(_MD_FENCE.findall(text))
    return round(min(marks / n, 1.0), 3)


# ---- 机械过渡词 / 八股 ----

_MECH_PATTERNS = [
    # 中文
    r"首先[，,、]", r"其次[，,、]", r"再次[，,、]", r"最后[，,、]", r"综上所述", r"总而言之",
    r"总的来说", r"需要注意的是", r"值得一提的是", r"希望(以上|这些)?(信息|回答|内容)?对您有(所)?帮助",
    r"如(有|您有)(任何)?(其他|其它)?(问题|疑问)", r"感谢您的(咨询|理解|耐心)",
    # 英文
    r"\bfirstly\b", r"\bsecondly\b", r"\bthirdly\b", r"\bin conclusion\b", r"\bto summarize\b",
    r"\bfurthermore\b", r"\bmoreover\b", r"\bit(?:'| i)s (?:worth|important) (?:noting|to note)\b",
    r"\bi hope this helps\b", r"\bfeel free to\b", r"\bplease don'?t hesitate\b",
]
_MECH_RE = re.compile("|".join(_MECH_PATTERNS), re.I)


def mechanical_transitions(text: str) -> list[str]:
    """命中的机械过渡词/客套八股（去重）。"""
    hits = [m.group(0) for m in _MECH_RE.finditer(text)]
    seen, out = set(), []
    for h in hits:
        k = h.lower()
        if k not in seen:
            seen.add(k)
            out.append(h)
    return out


# ---- 思维/元评论泄漏（对应 issue #2）----

_META_PATTERNS = [
    r"<\s*/?\s*think", r"<\s*/?\s*reasoning",
    # 中文自述流程
    r"这样(我)?(就|便)", r"接下来我(会|将|需要|应该)", r"我(需要|应该|可以|将)(先|再|继续|接着)",
    r"符合.{0,6}(要求|指引|规则|指南)", r"根据.{0,6}(追问|提示|指引|指南|规则)",
    r"(以下|下面)是我(的)?(回复|回答|思路|分析)", r"我的(回复|回答|思路|分析)(如下|是)",
    r"(作为|扮演).{0,6}(客服|助手|AI|模型)", r"(现在|那么)我来", r"让我(先|来|们)",
    # 英文自述流程
    r"\bas an ai\b", r"\bas (?:a|your) (?:customer service|assistant|sales)\b",
    r"\bhere'?s my (?:response|reply|answer)\b", r"\blet me (?:now )?(?:confirm|ask|summarize|check)\b",
    r"\bi (?:will|should|need to) (?:now )?(?:ask|confirm|check|proceed)\b", r"^\s*note\s*:",
]
_META_RE = re.compile("|".join(_META_PATTERNS), re.I | re.M)


def meta_leak(text: str) -> list[str]:
    """检测模型把'自述/思维过程'漏进客户可见回复的片段。"""
    return list({m.group(0).strip() for m in _META_RE.finditer(text)})


# ---- 综合 ----

def reply_penalties(reply: str) -> dict[str, Any]:
    """对单条客服回复计算确定性自然度惩罚项。"""
    md = markdown_density(reply)
    mech = mechanical_transitions(reply)
    meta = meta_leak(reply)
    length = len(reply)
    over_len = length > settings.eval_max_reply_chars

    # 综合 AI 味惩罚 0..1（meta 泄漏最严重）
    penalty = min(
        0.35 * (1.0 if md >= 0.34 else md / 0.34 if md else 0.0)
        + 0.25 * (1.0 if mech else 0.0)
        + 0.40 * (1.0 if meta else 0.0)
        + 0.15 * (1.0 if over_len else 0.0),
        1.0,
    )
    return {
        "markdown_density": md,
        "mechanical_transitions": mech,
        "meta_leak": meta,
        "length_chars": length,
        "over_length": over_len,
        "ai_tone_penalty": round(penalty, 3),
    }


def score_replies(replies: list[str]) -> dict[str, Any]:
    """对一组客服回复聚合确定性自然度指标（供 judges 调用）。"""
    if not replies:
        return {
            "n_replies": 0, "avg_markdown_density": 0.0, "mechanical_transition_rate": 0.0,
            "meta_leak_count": 0, "over_length_rate": 0.0, "avg_ai_tone_penalty": 0.0,
            "meta_leak_examples": [],
        }
    per = [reply_penalties(r) for r in replies]
    n = len(per)
    meta_examples: list[str] = []
    for p in per:
        meta_examples.extend(p["meta_leak"])
    return {
        "n_replies": n,
        "avg_markdown_density": round(sum(p["markdown_density"] for p in per) / n, 3),
        "mechanical_transition_rate": round(sum(1 for p in per if p["mechanical_transitions"]) / n, 3),
        "meta_leak_count": sum(len(p["meta_leak"]) for p in per),
        "over_length_rate": round(sum(1 for p in per if p["over_length"]) / n, 3),
        "avg_ai_tone_penalty": round(sum(p["ai_tone_penalty"] for p in per) / n, 3),
        "meta_leak_examples": meta_examples[:5],
    }
