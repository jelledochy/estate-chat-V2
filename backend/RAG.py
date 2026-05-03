from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.rag.constants import (  # noqa: E402
    COLLECTION_NAME,
    CROSS_ENCODER_MODEL_NAME,
    DEFAULT_LLM_MODEL,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    GRAPH_MODEL,
    RETRIEVAL_CANDIDATE_MULTIPLIER,
)
from backend.rag.graph_search import graph_search  # noqa: E402
from backend.rag.helpers import (  # noqa: E402
    _low_confidence_warning,
    _triplet_display,
    build_prompt,
    llm,
    rerank_context,
)
from backend.rag.vector_search import load_collection, search  # noqa: E402


def rag(
    *,
    query: str,
    model: str = DEFAULT_LLM_MODEL,
    graph_model: str = GRAPH_MODEL,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    start_time = time()
    openai_client, collection = load_collection()
    candidate_k = max(top_k, top_k * RETRIEVAL_CANDIDATE_MULTIPLIER)

    graph_error = None
    with ThreadPoolExecutor(max_workers=2) as executor:
        document_future = executor.submit(
            search,
            openai_client=openai_client,
            collection=collection,
            query=query,
            top_k=candidate_k,
        )
        graph_future = executor.submit(
            graph_search,
            query=query,
            model=graph_model,
        )

        document_results = document_future.result()
        try:
            graph_results = graph_future.result()
        except Exception as exc:
            graph_results = []
            graph_error = str(exc)

    document_results, graph_results = rerank_context(
        query=query,
        document_results=document_results,
        graph_results=graph_results,
        top_k=top_k,
    )
    prompt = build_prompt(query, document_results, graph_results)
    answer, token_stats = llm(openai_client=openai_client, prompt=prompt, model=model)
    confidence_warning = _low_confidence_warning(document_results, graph_results)

    return {
        "answer": answer,
        "model_used": model,
        "graph_model_used": graph_model,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reranker_model": CROSS_ENCODER_MODEL_NAME,
        "collection": COLLECTION_NAME,
        "top_k": top_k,
        "response_time_seconds": round(time() - start_time, 3),
        "sources": _document_sources(document_results),
        "graph_sources": _graph_sources(graph_results),
        "graph_error": graph_error,
        "confidence_warning": confidence_warning,
        "low_confidence": confidence_warning is not None,
        "prompt_tokens": token_stats["prompt_tokens"],
        "completion_tokens": token_stats["completion_tokens"],
        "total_tokens": token_stats["total_tokens"],
        "prompt": prompt,
    }


def _document_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = []
    for rank, row in enumerate(rows, start=1):
        metadata = row.get("metadata", {})
        sources.append(
            {
                "rank": rank,
                "chunk_id": row.get("chunk_id"),
                "document_id": metadata.get("document_id"),
                "document_type": metadata.get("document_type"),
                "page_number": metadata.get("page_number"),
                "distance": row.get("distance"),
                "rerank_score": row.get("rerank_score"),
            }
        )
    return sources


def _graph_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = []
    for rank, row in enumerate(rows, start=1):
        sources.append(
            {
                "rank": rank,
                "fact_id": row.get("fact_id"),
                "triplet": _triplet_display(row.get("triplet")),
                "text": row.get("text"),
                "cypher_query": (row.get("metadata") or {}).get("query"),
                "rerank_score": row.get("rerank_score"),
            }
        )
    return sources


def main(query: str) -> dict[str, Any]:
    return rag(query=query)
