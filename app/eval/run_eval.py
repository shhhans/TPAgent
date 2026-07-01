"""批量评测入口：跑一批 seed，聚合规则指标，打印报表 + 落 JSON。

用法：
    python -m app.eval.run_eval --seeds 1-10
    python -m app.eval.run_eval --seeds 1,4,7 --max-turns 8 --llm-judge
    python -m app.eval.run_eval --seeds 1-5 --out data/eval_report.json

建议评测时用无重依赖后端加速、并隔离真实记忆库：
    MEMORY_BACKEND=json VECTOR_BACKEND=keyword python -m app.eval.run_eval --seeds 1-10
"""
from __future__ import annotations

import argparse
import json
import statistics
from typing import Any

from app.config import settings
from app.eval import judges
from app.eval.harness import run_seed


def parse_seeds(spec: str) -> list[int]:
    """支持 '1-10' / '1,4,7' / '1-3,8,10-12' 混合写法。"""
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


def _mean(xs: list[float]) -> float:
    return round(statistics.mean(xs), 3) if xs else 0.0


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rules = [r["rules"] for r in rows]
    # 风险路由的准/召（正类=应转人工）
    tp = sum(1 for x in rules if x["expected_human"] and x["risk_routed"])
    fp = sum(1 for x in rules if not x["expected_human"] and x["risk_routed"])
    fn = sum(1 for x in rules if x["expected_human"] and not x["risk_routed"])
    precision = round(tp / (tp + fp), 3) if (tp + fp) else None
    recall = round(tp / (tp + fn), 3) if (tp + fn) else None

    lang_checked = [x["language_ok"] for x in rules if x["language_ok"] is not None]
    nat = [r["naturalness"] for r in rows if "naturalness" in r]

    agg = {
        "n": len(rows),
        "avg_completeness": _mean([x["completeness"] for x in rules]),
        "avg_correctness": _mean([x["correctness"] for x in rules]),
        "product_class_acc": _mean([1.0 if x["product_class_ok"] else 0.0 for x in rules]),
        "risk_routing_acc": _mean([1.0 if x["risk_ok"] else 0.0 for x in rules]),
        "risk_precision": precision,
        "risk_recall": recall,
        "language_match": _mean([1.0 if ok else 0.0 for ok in lang_checked]) if lang_checked else None,
        "over_collection_rate": _mean([1.0 if x["extra_params"] else 0.0 for x in rules]),
        "avg_turns": _mean([float(x["turns"]) for x in rules]),
        # 确定性自然度（免费，总是有）
        "avg_ai_tone_penalty": _mean([n["avg_ai_tone_penalty"] for n in nat]) if nat else None,
        "avg_markdown_density": _mean([n["avg_markdown_density"] for n in nat]) if nat else None,
        "meta_leak_total": sum(n["meta_leak_count"] for n in nat) if nat else 0,
    }
    # LLM 主观判分（可选）
    llm = [r["llm"] for r in rows if r.get("llm") and "llm_error" not in r["llm"]]
    if llm:
        agg["avg_naturalness_overall"] = _mean(
            [l["naturalness_overall"] for l in llm if l.get("naturalness_overall") is not None]
        )
        agg["verdict_natural_rate"] = _mean([1.0 if l.get("verdict_natural") else 0.0 for l in llm])
    return agg


def print_report(rows: list[dict[str, Any]], agg: dict[str, Any]) -> None:
    print("\n" + "=" * 96)
    print(f"{'seed':>4} {'compl':>6} {'corr':>6} {'cls':>4} {'risk':>5} {'lang':>5} {'xtra':>4} "
          f"{'aitone':>6} {'meta':>4} {'turns':>5}  persona")
    print("-" * 108)
    for r in rows:
        x = r["rules"]
        n = r.get("naturalness", {})
        cls = "✓" if x["product_class_ok"] else "✗"
        risk = "✓" if x["risk_ok"] else "✗"
        lang = "-" if x["language_ok"] is None else ("✓" if x["language_ok"] else "✗")
        xtra = str(len(x["extra_params"]))
        aitone = n.get("avg_ai_tone_penalty", 0.0)
        meta = str(n.get("meta_leak_count", 0))
        print(
            f"{r['seed']:>4} {x['completeness']:>6.2f} {x['correctness']:>6.2f} "
            f"{cls:>4} {risk:>5} {lang:>5} {xtra:>4} {aitone:>6.2f} {meta:>4} {x['turns']:>5}  {r['persona']}"
        )
    print("-" * 108)
    print(
        f"n={agg['n']}  完整率={agg['avg_completeness']}  正确率={agg['avg_correctness']}  "
        f"分类准确={agg['product_class_acc']}  风险路由准确={agg['risk_routing_acc']} "
        f"(P={agg['risk_precision']} R={agg['risk_recall']})"
    )
    print(
        f"语言匹配={agg['language_match']}  超纲收集率={agg['over_collection_rate']}  平均轮数={agg['avg_turns']}"
    )
    print(
        f"[自然度·规则] AI味惩罚={agg['avg_ai_tone_penalty']}  Markdown密度={agg['avg_markdown_density']}  "
        f"思维泄漏总数={agg['meta_leak_total']}"
    )
    if "avg_naturalness_overall" in agg:
        print(
            f"[自然度·LLM] 综合(1-3)={agg['avg_naturalness_overall']}  像真人比率={agg['verdict_natural_rate']}"
        )
    print("=" * 108 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="仿真客户批量评测")
    ap.add_argument("--seeds", default="1-5", help="如 1-10 / 1,4,7 / 1-3,8")
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--llm-judge", action="store_true", help="额外跑 LLM 主观判分（更慢/耗 API）")
    ap.add_argument("--out", default=None, help="报表 JSON 输出路径")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    max_turns = args.max_turns or settings.eval_max_turns
    print(f"评测 {len(seeds)} 个 seed，max_turns={max_turns}，llm_judge={args.llm_judge}")

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"  ▶ seed={seed} ...", flush=True)
        episode = run_seed(seed, max_turns)
        rows.append(judges.score(episode, use_llm=args.llm_judge))

    agg = aggregate(rows)
    print_report(rows, agg)

    if args.out:
        report = {"config": {"seeds": seeds, "max_turns": max_turns, "llm_judge": args.llm_judge},
                  "aggregate": agg, "rows": rows}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"报表已写入 {args.out}")


if __name__ == "__main__":
    main()
