from __future__ import annotations

from functools import lru_cache
from typing import Any

from backend.rag.constants import (
    GRAPH_EMBEDDING_MODEL_NAME,
    GRAPH_EXPANSION_DEPTH,
    GRAPH_RESULT_LIMIT,
    GRAPH_VECTOR_TOP_K,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URL,
    NEO4J_USER,
)
from backend.rag.helpers import _node_to_graph_facts, _triplet_display


@lru_cache(maxsize=4)
def load_graph_retriever(*, embedding_model: str) -> Any:
    try:
        from llama_index.core.indices.property_graph import VectorContextRetriever
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "llama-index is required for property-graph retrieval. "
            "Install the project dependencies before using graph context."
        ) from exc

    try:
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "llama-index Neo4j/OpenAI integrations are required for graph retrieval."
        ) from exc

    graph_store = Neo4jPropertyGraphStore(
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        url=NEO4J_URL,
        database=NEO4J_DATABASE,
    )
    return VectorContextRetriever(
        graph_store,
        embed_model=OpenAIEmbedding(model_name=embedding_model),
        include_text=False,
        similarity_top_k=GRAPH_VECTOR_TOP_K,
        path_depth=GRAPH_EXPANSION_DEPTH,
        limit=GRAPH_RESULT_LIMIT,
    )


def graph_search(
    *,
    query: str,
    embedding_model: str = GRAPH_EMBEDDING_MODEL_NAME,
) -> list[dict[str, Any]]:
    retriever = load_graph_retriever(embedding_model=embedding_model)
    graph_store = getattr(retriever, "_graph_store", None)
    if graph_store is not None:
        node_count = graph_store.structured_query("MATCH (n) RETURN count(n) AS count")
        if node_count and int(node_count[0].get("count", 0)) == 0:
            raise RuntimeError(
                "Neo4j graph is empty. Build the property graph before expecting graph context."
            )
    nodes = retriever.retrieve(query)

    facts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node_with_score in nodes[:GRAPH_RESULT_LIMIT]:
        remaining = GRAPH_RESULT_LIMIT - len(facts)
        for fact in _node_to_graph_facts(node_with_score, max_facts=remaining):
            triplet_key = _triplet_display(fact.get("triplet"))
            key = (triplet_key or str(fact.get("text") or "")).casefold()
            if key in seen:
                continue
            seen.add(key)
            facts.append(fact)
            if len(facts) >= GRAPH_RESULT_LIMIT:
                return facts
    return facts
