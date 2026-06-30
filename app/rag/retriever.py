"""本地向量检索 RAG：优先 Chroma；不可用时降级为关键词检索。

KB 为 data/kb 下的 markdown，按段落切块。
"""
from __future__ import annotations

import glob
import os
import re

from app.config import settings

_collection = None
_chunks: list[dict] | None = None  # 关键词降级用


def _load_chunks() -> list[dict]:
    """读取 KB markdown，按二级标题/空行切块。"""
    chunks: list[dict] = []
    for path in sorted(glob.glob(os.path.join(settings.kb_dir, "**/*.md"), recursive=True)):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        # 以二级标题或空行分段
        for i, block in enumerate(re.split(r"\n\s*\n", text)):
            block = block.strip()
            if block:
                chunks.append({"id": f"{os.path.basename(path)}#{i}", "text": block, "source": os.path.basename(path)})
    return chunks


def _init_chroma():
    global _collection
    if _collection is not None:
        return _collection
    import chromadb

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    col = client.get_or_create_collection("kb")
    if col.count() == 0:
        chunks = _load_chunks()
        if chunks:
            col.add(
                ids=[c["id"] for c in chunks],
                documents=[c["text"] for c in chunks],
                metadatas=[{"source": c["source"]} for c in chunks],
            )
    _collection = col
    return col


def _keyword_search(query: str, top_k: int) -> list[dict]:
    global _chunks
    if _chunks is None:
        _chunks = _load_chunks()
    q_terms = set(re.findall(r"\w+", query.lower()))
    scored = []
    for c in _chunks:
        text_terms = set(re.findall(r"\w+", c["text"].lower()))
        score = len(q_terms & text_terms)
        # 中文无空格，补一个子串命中分
        score += sum(1 for t in q_terms if t and t in c["text"].lower())
        if score > 0:
            scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


def rag_search(query: str, top_k: int | None = None) -> list[dict]:
    """返回 [{text, source}]。Chroma 失败自动降级关键词。"""
    k = top_k or settings.rag_top_k
    if settings.vector_backend == "chroma":
        try:
            col = _init_chroma()
            res = col.query(query_texts=[query], n_results=k)
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            return [{"text": d, "source": (m or {}).get("source", "")} for d, m in zip(docs, metas)]
        except Exception:
            pass  # 降级
    return _keyword_search(query, k)


def format_context(hits: list[dict]) -> str:
    if not hits:
        return "（知识库未检索到相关内容）"
    return "\n\n".join(f"[来源:{h['source']}]\n{h['text']}" for h in hits)
