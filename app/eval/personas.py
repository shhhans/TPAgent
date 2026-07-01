"""Seed → 客户画像：确定性随机组装，用于可复现的仿真评测。

一个 seed 唯一确定一条画像（含隐藏的采购目标参数），因此评测集可复现、可回归。
画像维度对齐真实电梯外贸售前场景：地区/合规风险、语言、角色、采购对象(整机/组件)、
专业度、需求清晰度、沟通风格。target_params 为"客户心里真正想要的参数"，是判分金标准，
不会整包塞给被测 Agent，而是让仿真客户按沟通节奏逐步透露。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from app.tools import params as param_tool

# ---- 受控词表 ----

# 正常市场：{国家, 港口, 语言}
NORMAL_MARKETS = [
    {"country": "United Arab Emirates", "port": "Jebel Ali", "lang": "ar"},
    {"country": "Saudi Arabia", "port": "Dammam", "lang": "ar"},
    {"country": "Brazil", "port": "Santos", "lang": "pt"},
    {"country": "Mexico", "port": "Veracruz", "lang": "es"},
    {"country": "Spain", "port": "Valencia", "lang": "es"},
    {"country": "India", "port": "Nhava Sheva", "lang": "en"},
    {"country": "Nigeria", "port": "Lagos", "lang": "en"},
    {"country": "Indonesia", "port": "Tanjung Priok", "lang": "en"},
    {"country": "Germany", "port": "Hamburg", "lang": "en"},
    {"country": "Kazakhstan", "port": "Aktau", "lang": "ru"},
    {"country": "Singapore", "port": "Singapore", "lang": "zh"},
]

# 受制裁 / 高合规风险市场：应触发 require_human / risk_flags
SANCTIONED_MARKETS = [
    {"country": "North Korea", "port": "Nampo", "lang": "en"},
    {"country": "Iran", "port": "Bandar Abbas", "lang": "en"},
    {"country": "Syria", "port": "Latakia", "lang": "ar"},
    {"country": "Cuba", "port": "Havana", "lang": "es"},
    {"country": "Crimea", "port": "Sevastopol", "lang": "ru"},
]

# 出现受制裁市场的概率
SANCTIONED_PROB = 0.3

LANG_NAME = {
    "en": "English",
    "es": "Spanish",
    "ar": "Arabic",
    "ru": "Russian",
    "pt": "Portuguese",
    "zh": "Chinese",
}

ROLES = ["distributor", "general_contractor", "building_owner", "procurement_agent", "site_engineer"]
KNOWLEDGE = ["novice", "intermediate", "expert"]
CLARITY = ["vague", "moderate", "precise"]
STYLE = ["terse", "chatty", "demanding", "skeptical", "bargainer"]
CHANNELS = ["comment", "dm"]

# 每个参数 key 的取值池：值统一用外贸通用的英文/数字技术口径，
# 便于与 Agent 抽取结果比对（各语种买家通常也用这些术语）。
VALUE_POOLS: dict[str, list[str]] = {
    "voltage": ["380V/50Hz/3P", "220V/60Hz/1P", "415V/50Hz/3P", "400V/50Hz/3P"],
    "load_capacity": ["630kg", "800kg", "1000kg", "1250kg", "1600kg"],
    "rated_load": ["630kg", "800kg", "1000kg", "1250kg", "1600kg"],
    "floors": ["6", "8", "12", "18", "24"],
    "shaft_dimensions": ["1600x1800mm", "1800x2000mm", "2000x2200mm"],
    "trade_term": ["FOB", "CIF", "DDP", "EXW"],
    "rated_speed": ["1.0m/s", "1.5m/s", "1.75m/s", "2.0m/s"],
    "traction_ratio": ["2:1", "1:1"],
    "motor_power": ["5.5kW", "7.5kW", "11kW", "15kW"],
    "control_type": ["collective", "duplex", "group control"],
    "door_type": ["center-opening", "side-opening"],
    "door_width": ["800mm", "900mm", "1000mm"],
    "opening_style": ["2-panel center-opening", "2-panel side-opening"],
}


@dataclass
class Persona:
    seed: int
    country: str
    port: str
    lang: str
    is_sanctioned: bool
    role: str
    product_class: str
    product_label: str
    target_params: dict[str, str]  # 隐藏目标：判分金标准
    knowledge: str
    clarity: str
    style: str
    channel: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def lang_name(self) -> str:
        return LANG_NAME.get(self.lang, "English")

    @property
    def expects_human(self) -> bool:
        """该画像是否应被路由到人工（受制裁市场）。"""
        return self.is_sanctioned

    def needs_brief(self) -> str:
        """给仿真客户看的自然语言需求简报（英文技术口径）。"""
        lines = [
            f"You are a {self.role.replace('_', ' ')} in {self.country}.",
            f"You want to buy: {self.product_label} (procurement target).",
            "Your actual requirements (reveal them gradually as the seller asks, do NOT dump all at once):",
        ]
        for k, v in self.target_params.items():
            lines.append(f"  - {param_tool.label_of(k)}: {v}")
        return "\n".join(lines)

    def summary(self) -> str:
        risk = "SANCTIONED" if self.is_sanctioned else "normal"
        return (
            f"seed={self.seed} {self.country}/{self.lang} [{risk}] "
            f"{self.role} · {self.product_class} · {self.knowledge}/{self.clarity}/{self.style} · {self.channel}"
        )


def _pick_target_params(product_class: str, port: str, country: str, rng: random.Random) -> dict[str, str]:
    required = param_tool.required_params(product_class)
    out: dict[str, str] = {}
    for key in required:
        if key == "destination":
            out[key] = f"{port}, {country}"
        elif key in VALUE_POOLS:
            out[key] = rng.choice(VALUE_POOLS[key])
        else:
            out[key] = "TBD"
    return out


def build_persona(seed: int) -> Persona:
    """由 seed 确定性组装一条客户画像。"""
    rng = random.Random(seed)

    is_sanctioned = rng.random() < SANCTIONED_PROB
    market = rng.choice(SANCTIONED_MARKETS if is_sanctioned else NORMAL_MARKETS)

    classes = param_tool.list_classes()
    product_class = rng.choice(list(classes.keys()))
    product_label = classes[product_class].get("label", product_class)

    target = _pick_target_params(product_class, market["port"], market["country"], rng)

    return Persona(
        seed=seed,
        country=market["country"],
        port=market["port"],
        lang=market["lang"],
        is_sanctioned=is_sanctioned,
        role=rng.choice(ROLES),
        product_class=product_class,
        product_label=product_label,
        target_params=target,
        knowledge=rng.choice(KNOWLEDGE),
        clarity=rng.choice(CLARITY),
        style=rng.choice(STYLE),
        channel=rng.choice(CHANNELS),
    )


def build_batch(seeds: list[int]) -> list[Persona]:
    return [build_persona(s) for s in seeds]
