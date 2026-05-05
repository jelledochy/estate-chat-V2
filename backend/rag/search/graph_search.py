from __future__ import annotations

from functools import lru_cache
from typing import Any

from backend.rag.constants import (
    GRAPH_MODEL,
    GRAPH_RESULT_LIMIT,
    GRAPH_TEXT_TO_CYPHER_TEMPLATE,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URL,
    NEO4J_USER,
)
from backend.rag.helpers import _cypher_validator, _node_to_graph_facts, _triplet_display


@lru_cache(maxsize=4)
def load_graph_retriever(*, model: str = GRAPH_MODEL) -> Any:
    try:
        from llama_index.core.indices.property_graph import TextToCypherRetriever
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "llama-index is required for property-graph retrieval. "
            "Install the project dependencies before using graph context."
        ) from exc
    except ImportError:
        from llama_index.core.retrievers import TextToCypherRetriever

    try:
        from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
        from llama_index.llms.openai import OpenAI as LlamaOpenAI
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
    return TextToCypherRetriever(
        graph_store,
        llm=LlamaOpenAI(model=model, temperature=0.0),
        text_to_cypher_template=GRAPH_TEXT_TO_CYPHER_TEMPLATE,
        cypher_validator=_cypher_validator,
        include_raw_response_as_metadata=True,
    )


def graph_search(
    *,
    query: str,
    model: str = GRAPH_MODEL,
) -> list[dict[str, Any]]:
    retriever = load_graph_retriever(model=model)
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

