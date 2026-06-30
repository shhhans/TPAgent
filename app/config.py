"""集中配置：所有可配置参数统一从 .env 读取，业务代码只读 settings.xxx。"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- LLM (MiniMax / OpenAI 兼容) ----
    minimax_api_key: str = ""
    minimax_base_url: str = "https://api.minimaxi.com/v1"
    minimax_model: str = "MiniMax-M2"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048

    # ---- 向量库 / RAG ----
    vector_backend: str = "chroma"  # chroma | keyword
    chroma_persist_dir: str = "./data/chroma"
    kb_dir: str = "./data/kb"
    rag_top_k: int = 4
    embedding_backend: str = "chroma_default"  # chroma_default | sentence_transformers
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # ---- 长期记忆 ----
    memory_backend: str = "json"  # mem0 | json
    memory_json_path: str = "./data/memory_store.json"

    # ---- 短期记忆 / Checkpointer ----
    checkpointer: str = "memory"  # memory | sqlite
    checkpoint_db: str = "./data/checkpoints.sqlite"

    # ---- 业务参数 schema ----
    param_schema_path: str = "./data/param_schema.json"

    # ---- HITL ----
    hitl_high_value_amount: float = 50000

    # ---- 杂项 ----
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
