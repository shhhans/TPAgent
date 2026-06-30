"""Streamlit 简单可视化：聊天 + 实时展示 Agent State（意图/参数/记忆/风险）。

启动: streamlit run app/ui/streamlit_app.py
"""
from __future__ import annotations

import uuid

import streamlit as st

from app.memory.mem0_store import format_memory
from app.service import finalize_session, run_turn

st.set_page_config(page_title="电梯外贸售前客服 Agent", layout="wide")
st.title("🛗 海外电商智能客服 Agent — v1 可视化")

# ---- 会话初始化 ----
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"thread-{uuid.uuid4().hex[:8]}"
    st.session_state.history = []  # [(role, content)]
    st.session_state.last_state = {}

with st.sidebar:
    st.subheader("会话设置")
    customer_id = st.text_input("客户 ID", value="demo_customer")
    st.caption(f"thread: {st.session_state.thread_id}")
    if st.button("结束会话并固化记忆"):
        mem = finalize_session(customer_id, st.session_state.thread_id)
        st.success("记忆已固化")
        st.json(mem)
    if st.button("新建会话"):
        st.session_state.thread_id = f"thread-{uuid.uuid4().hex[:8]}"
        st.session_state.history = []
        st.session_state.last_state = {}
        st.rerun()

col_chat, col_state = st.columns([3, 2])

with col_chat:
    st.subheader("对话")
    for role, content in st.session_state.history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.write(content)

    prompt = st.chat_input("输入客户消息…")
    if prompt:
        st.session_state.history.append(("user", prompt))
        with st.chat_message("user"):
            st.write(prompt)
        with st.spinner("Agent 处理中…"):
            out = run_turn(customer_id, st.session_state.thread_id, prompt)
        st.session_state.last_state = out["state"]
        st.session_state.history.append(("assistant", out["reply"]))
        with st.chat_message("assistant"):
            st.write(out["reply"])

with col_state:
    st.subheader("🧠 Agent State（实时）")
    s = st.session_state.last_state
    if not s:
        st.info("发送一条消息后这里会显示 Triage 决策与状态。")
    else:
        st.metric("意图 intent", s.get("intent_category", "-"))
        c1, c2 = st.columns(2)
        c1.metric("产品分类", s.get("product_class") or "未识别")
        c2.metric("待人工", "是" if s.get("awaiting_human") else "否")
        if s.get("risk_flags"):
            st.error("风险标记: " + ", ".join(s["risk_flags"]))
        st.markdown("**已收集参数**")
        st.json(s.get("collected_params", {}))
        st.markdown("**缺失必填参数**")
        st.write(s.get("missing_params", []) or "（无 / 已齐全）")
        st.markdown("**召回的长期记忆**")
        st.text(format_memory(s.get("customer_memory", {})))
