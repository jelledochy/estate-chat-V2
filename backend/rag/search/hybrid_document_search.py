from __future__ import annotations

from typing import Any

import chromadb
from openai import OpenAI

from backend.rag.constants import DEFAULT_TOP_K
from backend.rag.helpers import load_cross_encoder
from backend.rag.search.keyword_search import keyword_search
from backend.rag.search.vector_search import search as vector_search


def search_source_documents(
    *,
    openai_client: OpenAI,
    collection: chromadb.Collection,
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    vector_results = vector_search(
        openai_client=openai_client,
        collection=collection,
        query=query,
        top_k=top_k,
    )
    keyword_results = keyword_search(collection=collection, query=query, top_k=top_k)
    candidates = merge_document_results(
        vector_results=vector_results,
        keyword_results=keyword_results,
    )
    return rerank_source_documents(query=query, rows=candidates, top_k=top_k)


def hybrid_document_search(
    *,
    openai_client: OpenAI,
    collection: chromadb.Collection,
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    return search_source_documents(
        openai_client=openai_client,
        collection=collection,
        query=query,
        top_k=top_k,
    )


def rerank_source_documents(
    *,
    query: str,
    rows: list[dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    if not rows or top_k < 1:
        return []

    cross_encoder = load_cross_encoder()
    scores = cross_encoder.predict([(query, row.get("text", "")) for row in rows])
    for row, score in zip(rows, scores, strict=False):
        row["source_rerank_score"] = float(score)

    return sorted(
        rows,
        key=lambda row: float(row.get("source_rerank_score", float("-inf"))),
        reverse=True,
    )[:top_k]


def merge_document_results(
    *,
    vector_results: list[dict[str, Any]],
    keyword_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for rank, row in enumerate(vector_results, start=1):
        merge_document_result(merged, row, method="vector", rank=rank)
    for rank, row in enumerate(keyword_results, start=1):
        merge_document_result(merged, row, method="keyword", rank=rank)
    return list(merged.values())


def merge_document_result(
    merged: dict[str, dict[str, Any]],
    row: dict[str, Any],
    *,
    method: str,
    rank: int,
) -> None:
    key = str(row.get("chunk_id") or row.get("text") or rank)
    current = merged.setdefault(key, {**row, "retrieval_methods": []})
    methods = current.setdefault("retrieval_methods", [])
    if method not in methods:
        methods.append(method)
    current[f"{method}_rank"] = rank

    for field in ("bm25_score", "distance"):
        if field in row and current.get(field) is None:
            current[field] = row[field]


__all__ = [
    "hybrid_document_search",
    "merge_document_result",
    "merge_document_results",
    "rerank_source_documents",
    "search_source_documents",
]
