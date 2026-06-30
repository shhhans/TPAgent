"""LLM 适配层：统一走 OpenAI 兼容接口（MiniMax）。

上层只依赖 chat() / chat_json()，换供应商只改 .env 的 base_url/model/key。
"""
from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from app.config import settings

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not settings.minimax_api_key:
            raise RuntimeError("MINIMAX_API_KEY 未配置，请在 .env 中设置。")
        _client = OpenAI(
            api_key=settings.minimax_api_key,
            base_url=settings.minimax_base_url,
        )
    return _client


def chat(
    messages: list[dict[str, str]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    json_mode: bool = False,
) -> str:
    """返回模型文本输出。json_mode=True 时请求结构化 JSON。"""
    kwargs: dict[str, Any] = {
        "model": settings.minimax_model,
        "messages": messages,
        "temperature": settings.llm_temperature if temperature is None else temperature,
        "max_tokens": settings.llm_max_tokens if max_tokens is None else max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _get_client().chat.completions.create(**kwargs)
    content = resp.choices[0].message.content or ""
    return _strip_think(content).strip()


def _strip_think(text: str) -> str:
    """剥离推理模型（如 MiniMax-M2）的 <think>...</think> 块。"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _extract_json(text: str) -> dict:
    """从文本里稳健地抠出 JSON 对象（容错 markdown 代码块/前后缀）。"""
    text = text.strip()
    # 去掉 ```json ... ``` 围栏
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 兜底：取第一个 { 到最后一个 }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"无法解析为 JSON: {text[:200]}")


def chat_json(
    messages: list[dict[str, str]],
    *,
    temperature: float | None = None,
    retries: int = 1,
) -> dict:
    """要求模型输出 JSON 并解析为 dict；解析失败重试 retries 次。"""
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        raw = chat(messages, temperature=temperature, json_mode=True)
        try:
            return _extract_json(raw)
        except ValueError as e:
            last_err = e
            # 重试时降温并追加修复指令
            messages = messages + [
                {"role": "user", "content": "上次输出不是合法 JSON，请只返回一个合法 JSON 对象，不要任何多余文字。"}
            ]
            temperature = 0.0
    raise ValueError(f"chat_json 多次解析失败: {last_err}")
