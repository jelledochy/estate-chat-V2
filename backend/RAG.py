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
    _document_citation_label,
    _low_confidence_warning,
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

    neo4j_graph_results = graph_results
    document_results, prompt_graph_results = rerank_context(
        query=query,
        document_results=document_results,
        graph_results=neo4j_graph_results,
        top_k=top_k,
    )
    prompt = build_prompt(query, document_results, prompt_graph_results)
    answer, token_stats = llm(openai_client=openai_client, prompt=prompt, model=model)
    confidence_warning = _low_confidence_warning(document_results, prompt_graph_results)

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
        "neo4j_graph_context": _graph_context_items(neo4j_graph_results),
        "prompt_graph_context": _graph_context_items(prompt_graph_results),
        "graph_error": graph_error,
        "confidence_warning": confidence_warning,
        "low_confidence": confidence_warning is not None,
        "prompt_tokens": token_stats["prompt_tokens"],
        "completion_tokens": token_stats["completion_tokens"],
        "total_tokens": token_stats["total_tokens"],
        "prompt": prompt,
    }


def _document_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for rank, row in enumerate(rows, start=1):
        metadata = row.get("metadata", {})
        document_id = str(metadata.get("document_id") or "").strip()
        if not document_id:
            continue
        page_number = _positive_int_or_none(metadata.get("page_number"))
        group_key = f"{document_id}:p{page_number or 0}"

        source = grouped.setdefault(
            group_key,
            {
                "rank": rank,
                "chunk_id": row.get("chunk_id"),
                "chunk_ids": [],
                "document_id": metadata.get("document_id"),
                "document_type": metadata.get("document_type"),
                "page_number": None,
                "page_numbers": [],
                "distance": row.get("distance"),
                "rerank_score": row.get("rerank_score"),
                "citation_label": _document_citation_label(row),
            },
        )
        chunk_id = row.get("chunk_id")
        if chunk_id and chunk_id not in source["chunk_ids"]:
            source["chunk_ids"].append(chunk_id)

        if page_number is not None and page_number not in source["page_numbers"]:
            source["page_numbers"].append(page_number)
            source["page_numbers"].sort()
            source["page_number"] = source["page_numbers"][0]

        if source.get("distance") is None or (
            row.get("distance") is not None and row["distance"] < source["distance"]
        ):
            source["distance"] = row.get("distance")
        if source.get("rerank_score") is None or (
            row.get("rerank_score") is not None
            and row["rerank_score"] > source["rerank_score"]
        ):
            source["rerank_score"] = row.get("rerank_score")

    return sorted(grouped.values(), key=lambda source: int(source.get("rank") or 0))


def _graph_context_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        items.append(
            {
                "rank": rank,
                "triplet": _graph_triplet_parts(row.get("triplet")),
                "text": str(row.get("text") or "").strip(),
                "cypher_query": str(metadata.get("query") or "").strip() or None,
                "cypher_row": _json_safe_dict(metadata.get("cypher_row")),
                "rerank_score": _float_or_none(row.get("rerank_score")),
            }
        )
    return items


def _graph_triplet_parts(value: Any) -> list[str]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        return []
    parts = [str(part).strip() for part in value]
    return parts if all(parts) else []


def _json_safe_dict(value: Any) -> dict[str, Any]:
    return _json_safe(value) if isinstance(value, dict) else {}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    return str(value)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _positive_int_or_none(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def main(query: str) -> dict[str, Any]:
    return rag(query=query)
