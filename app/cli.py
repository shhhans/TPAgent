"""命令行烟测：python -m app.cli  （输入 :q 退出并固化记忆）。"""
from __future__ import annotations

import json
import sys
import uuid

from app.service import finalize_session, run_turn


def main() -> None:
    customer_id = sys.argv[1] if len(sys.argv) > 1 else "demo_customer"
    thread_id = f"thread-{uuid.uuid4().hex[:8]}"
    print(f"== 电梯外贸售前客服 (customer={customer_id}, thread={thread_id}) ==")
    print("输入消息开始对话，:q 退出。\n")
    while True:
        try:
            text = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            text = ":q"
        if text in (":q", ":quit", "exit"):
            mem = finalize_session(customer_id, thread_id)
            print("\n[记忆已固化] " + json.dumps(mem, ensure_ascii=False))
            break
        if not text:
            continue
        out = run_turn(customer_id, thread_id, text)
        st = out["state"]
        print(f"客服> {out['reply']}")
        print(
            f"   [intent={st.get('intent_category')} class={st.get('product_class')} "
            f"missing={st.get('missing_params')} human={st.get('awaiting_human')} "
            f"risk={st.get('risk_flags')}]\n"
        )


if __name__ == "__main__":
    main()
