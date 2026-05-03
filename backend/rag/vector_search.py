from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from backend.rag.constants import (
    CHROMA_PATH,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    PROJECT_ROOT,
)


@lru_cache(maxsize=1)
def load_collection() -> tuple[OpenAI, chromadb.Collection]:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        raise ValueError(
            "OPENAI_API_KEY is missing. Add it to /workspace/.env or your environment."
        )
    if not CHROMA_PATH.exists():
        raise FileNotFoundError(f"Chroma path does not exist: {CHROMA_PATH}")

    openai_client = OpenAI()
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    return openai_client, collection


def embed_query(openai_client: OpenAI, query: str) -> list[float]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL_NAME,
        input=[query],
    )
    return response.data[0].embedding


def search(
    *,
    openai_client: OpenAI,
    collection: chromadb.Collection,
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    query_embedding = embed_query(openai_client, query)
    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }

    result = collection.query(**query_kwargs)

    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    rows: list[dict[str, Any]] = []
    for idx, chunk_id in enumerate(ids):
        metadata = metadatas[idx] or {}
        rows.append(
            {
                "chunk_id": chunk_id,
                "text": documents[idx] if idx < len(documents) else "",
                "distance": float(distances[idx]) if idx < len(distances) else 0.0,
                "metadata": metadata,
            }
        )
    return rows
