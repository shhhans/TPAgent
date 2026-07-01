"""MVVP —— LLM 裁判"上岗前"验收协议（Minimal Viable Validation Protocol）。

依据文献：别把"裁判很稳定"当可信（一致性-偏见悖论）。上岗前必须：
  1) 报"校正后的 Cohen's κ"而非原始匹配率（原始匹配率相对 κ 通常注水 33-41 个点）；
  2) 用 AB 位置交换量化"位置偏见"；
  3) 在 ≥2 个标签分布不同的子集上交叉验证。

标注数据：优先用真人标注的 JSONL（--labeled），每行
  {"text": "...", "natural": true/false, "lang": "English", "group": "human|ai|..."}。
无标注数据时用内置演示集，先把协议本身跑起来（也可当冒烟测试）。
"""
from __future__ import annotations

import argparse
import json
from typing import Any

from app.eval import judges

# 内置演示标注集：明显"像真人 (natural=True)" vs 明显"AI 味 (natural=False)"。
# group 用于交叉验证（不同标签分布的子集）。
_DEMO: list[dict[str, Any]] = [
    {"text": "Sure, 1000kg works fine for an 8-floor building. What's your target delivery port?",
     "natural": True, "lang": "English", "group": "human"},
    {"text": "Got it — for that shaft size we'd usually go 1.5 m/s. Want me to check CIF pricing to Santos?",
     "natural": True, "lang": "English", "group": "human"},
    {"text": "No worries, take your time. Ping me the voltage whenever you have it and I'll sort the rest.",
     "natural": True, "lang": "English", "group": "human"},
    {"text": "Totally understand the frustration — let me get this fixed for you right away.",
     "natural": True, "lang": "English", "group": "human"},
    {"text": "Firstly, thank you for your inquiry. Secondly, please kindly provide the following parameters: "
             "1. Voltage 2. Load 3. Floors. Finally, we hope this information is helpful to you.",
     "natural": False, "lang": "English", "group": "ai"},
    {"text": "As an AI customer service assistant, I have confirmed your requirements. This is my response: "
             "**Voltage**: 380V. Next I will ask about the trade term as required by the guidelines.",
     "natural": False, "lang": "English", "group": "ai"},
    {"text": "Your feedback has been received. Please note that the following specifications are required. "
             "It is worth noting that we hope this helps. Feel free to contact us.",
     "natural": False, "lang": "English", "group": "ai"},
    {"text": "Thank you for your valuable inquiry. In conclusion, to summarize, we hope this information "
             "is helpful and please don't hesitate to reach out with any further questions.",
     "natural": False, "lang": "English", "group": "ai"},
]


def cohen_kappa(a: list[Any], b: list[Any]) -> float | None:
    """两组分类标签的 Cohen's κ（偶然性校正后的一致度）。"""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return None
    labels = sorted({x for p in pairs for x in p}, key=str)
    po = sum(1 for x, y in pairs if x == y) / n
    pe = 0.0
    for lab in labels:
        pa = sum(1 for x, _ in pairs if x == lab) / n
        pb = sum(1 for _, y in pairs if y == lab) / n
        pe += pa * pb
    if pe >= 1.0:
        return 1.0
    return round((po - pe) / (1 - pe), 3)


def _raw_match(a: list[Any], b: list[Any]) -> float:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    return round(sum(1 for x, y in pairs if x == y) / len(pairs), 3) if pairs else 0.0


def run_pointwise(items: list[dict[str, Any]]) -> dict[str, Any]:
    """逐条二元判定 vs 金标签：报原始匹配率、κ、注水差(Kappa deflation)。"""
    gold = [bool(it["natural"]) for it in items]
    pred = [judges.judge_natural_binary(it["text"], it.get("lang", "English")) for it in items]
    raw = _raw_match(gold, pred)
    kappa = cohen_kappa(gold, pred)
    deflation = round((raw - kappa) * 100, 1) if kappa is not None else None
    return {"n": len(items), "raw_match": raw, "cohen_kappa": kappa, "kappa_deflation_pts": deflation,
            "gold": gold, "pred": pred}


def run_position_swap(items: list[dict[str, Any]]) -> dict[str, Any]:
    """AB 位置交换探位置偏见：对每个 (natural, ai) 配对跑两种顺序。

    - 无偏理想：内容一致地选中"更自然"那条，与位置无关。
    - order_agreement 低 = 判定随位置翻转；position_A_rate 远离 0.5 = 偏好某个位置。
    """
    nat = [it for it in items if it["natural"]]
    ai = [it for it in items if not it["natural"]]
    pairs = list(zip(nat, ai))  # (更自然, AI味)
    consistent = 0            # 两种顺序都判"更自然"那条胜
    pos_a_wins = 0
    total = 0
    for good, bad in pairs:
        w1 = judges.judge_pairwise(good["text"], bad["text"], good.get("lang", "English"))  # good=A
        w2 = judges.judge_pairwise(bad["text"], good["text"], good.get("lang", "English"))  # good=B
        if w1 is None or w2 is None:
            continue
        total += 1
        # 内容一致 = 顺序1选A 且 顺序2选B（都选中 good）
        if w1 == "A" and w2 == "B":
            consistent += 1
        if w1 == "A":
            pos_a_wins += 1
        if w2 == "A":
            pos_a_wins += 1
    order_agreement = round(consistent / total, 3) if total else None
    position_a_rate = round(pos_a_wins / (2 * total), 3) if total else None
    return {"pairs": total, "order_agreement": order_agreement,
            "position_A_rate": position_a_rate,
            "position_bias": None if position_a_rate is None else round(abs(position_a_rate - 0.5) * 2, 3)}


def cross_validate(items: list[dict[str, Any]]) -> dict[str, Any]:
    """在不同 group（标签分布不同）子集上分别报 κ。"""
    groups: dict[str, list[dict]] = {}
    for it in items:
        groups.setdefault(it.get("group", "all"), []).append(it)
    out = {}
    for g, sub in groups.items():
        gold = [bool(it["natural"]) for it in sub]
        pred = [judges.judge_natural_binary(it["text"], it.get("lang", "English")) for it in sub]
        out[g] = {"n": len(sub), "raw_match": _raw_match(gold, pred), "cohen_kappa": cohen_kappa(gold, pred)}
    return out


def load_items(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return _DEMO
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="MVVP：LLM 裁判上岗前验收")
    ap.add_argument("--labeled", default=None, help="标注 JSONL 路径（缺省用内置演示集）")
    ap.add_argument("--skip-swap", action="store_true", help="跳过位置交换探测（省 API）")
    args = ap.parse_args()

    items = load_items(args.labeled)
    src = args.labeled or "内置演示集"
    print(f"MVVP 验收 · 数据={src} · 样本={len(items)}\n")

    pw = run_pointwise(items)
    print("① 逐条二元判定 vs 金标签")
    print(f"   原始匹配率 raw_match = {pw['raw_match']}")
    print(f"   校正一致度 Cohen's κ = {pw['cohen_kappa']}   （κ<0.6 慎用，<0.4 不可用）")
    print(f"   注水差 kappa_deflation = {pw['kappa_deflation_pts']} 个百分点（原始匹配率虚高幅度）\n")

    print("③ 交叉验证（不同标签分布子集）")
    for g, r in cross_validate(items).items():
        print(f"   [{g:>6}] n={r['n']}  raw={r['raw_match']}  κ={r['cohen_kappa']}")
    print()

    if not args.skip_swap:
        sw = run_position_swap(items)
        print("② AB 位置交换（位置偏见）")
        print(f"   顺序一致率 order_agreement = {sw['order_agreement']}   （越高越好，理想=1.0）")
        print(f"   选 A 位比率 position_A_rate = {sw['position_A_rate']}   （理想≈0.5）")
        print(f"   位置偏见 position_bias = {sw['position_bias']}   （0=无偏，1=完全按位置）")
        print()

    verdict = "通过" if (pw["cohen_kappa"] or 0) >= 0.6 else "存疑/不通过"
    print(f"结论：裁判 κ={pw['cohen_kappa']} → {verdict}（κ≥0.6 方可信任其自然度判分）")


if __name__ == "__main__":
    main()
