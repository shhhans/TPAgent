"""报价参数槽工具：ETIM/ECLASS 分类驱动的 slot filling 与校验。"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from app.config import settings


@lru_cache
def _schema() -> dict:
    with open(settings.param_schema_path, encoding="utf-8") as f:
        return json.load(f)


def list_classes() -> dict[str, dict]:
    """返回 {class_key: {label, required, ...}}，供 Triage 提示词列出可选分类。"""
    return _schema().get("classes", {})


def label_of(param_key: str) -> str:
    return _schema().get("param_labels", {}).get(param_key, param_key)


def required_params(product_class: str | None) -> list[str]:
    if not product_class:
        return []
    return _schema().get("classes", {}).get(product_class, {}).get("required", [])


def update_params(
    collected: dict[str, Any], new_params: dict[str, Any]
) -> dict[str, Any]:
    """把客户本轮提供的参数并入已收集参数（None/空值不覆盖已有）。"""
    merged = dict(collected)
    for k, v in (new_params or {}).items():
        if v in (None, "", []):
            continue
        merged[k] = v
    return merged


def compute_missing(
    product_class: str | None, collected: dict[str, Any]
) -> list[str]:
    """该分类必填特征中尚未填充的部分。"""
    return [p for p in required_params(product_class) if p not in collected or collected[p] in (None, "", [])]


def is_complete(product_class: str | None, collected: dict[str, Any]) -> bool:
    return bool(product_class) and not compute_missing(product_class, collected)


def describe_missing(missing: list[str]) -> str:
    """把缺失参数 key 转成人类可读的中文清单，供追问。"""
    return "、".join(f"{label_of(p)}" for p in missing)
