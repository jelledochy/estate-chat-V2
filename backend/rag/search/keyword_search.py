from __future__ import annotations

import re
from typing import Any

import chromadb

from backend.rag.constants import DEFAULT_TOP_K

TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def keyword_search(
    *,
    collection: chromadb.Collection,
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    query_tokens = _tokenize(query)
    if not query_tokens or top_k < 1:
        return []

    rows = _collection_rows(collection)
    if not rows:
        return []

    try:
        from langchain_community.retrievers import BM25Retriever
        from langchain_core.documents import Document
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "BM25 keyword search requires langchain-community and rank-bm25. "
            "Run `poetry lock` and `poetry install` after adding the dependencies."
        ) from exc

    documents = [
        Document(page_content=row.get("text", ""), metadata={"row_index": idx})
        for idx, row in enumerate(rows)
    ]
    try:
        retriever = BM25Retriever.from_documents(
            documents,
            k=top_k,
            preprocess_func=_tokenize,
        )
    except ImportError as exc:
        raise RuntimeError(
            "BM25 keyword search requires rank-bm25. Run `poetry install`."
        ) from exc

    keyword_rows: list[dict[str, Any]] = []
    for rank, document in enumerate(retriever.invoke(query), start=1):
        row_index = int(document.metadata["row_index"])
        keyword_rows.append({**rows[row_index], "keyword_rank": rank})
    return keyword_rows


def _collection_rows(collection: chromadb.Collection) -> list[dict[str, Any]]:
    count = collection.count()
    if count < 1:
        return []

    result = collection.get(
        limit=count,
        include=["documents", "metadatas"],
    )
    ids = result.get("ids", [])
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])

    rows: list[dict[str, Any]] = []
    for idx, chunk_id in enumerate(ids):
        rows.append(
            {
                "chunk_id": chunk_id,
                "text": documents[idx] if idx < len(documents) else "",
                "metadata": metadatas[idx] or {} if idx < len(metadatas) else {},
            }
        )
    return rows


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())

