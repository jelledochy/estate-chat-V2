from __future__ import annotations

from typing import Any

import streamlit as st

try:
    from streamlit_agraph import Config, Edge, Node, agraph
except ModuleNotFoundError:
    Config = Edge = Node = agraph = None


def render_graph_context(graph_context: dict[str, Any], key_prefix: str) -> None:
    if not isinstance(graph_context, dict):
        graph_context = {}

    neo4j_items = _graph_context_items(graph_context.get("neo4j"))
    prompt_items = _graph_context_items(graph_context.get("prompt"))
    graph_error = str(graph_context.get("error") or "").strip()

    with st.expander("Knowledge graph context"):
        view_key = f"{key_prefix}_graph_context_view"
        if view_key not in st.session_state:
            st.session_state[view_key] = "prompt"

        current_view = st.session_state[view_key]
        if current_view not in {"neo4j", "prompt"}:
            current_view = "prompt"
            st.session_state[view_key] = current_view

        prompt_column, neo4j_column = st.columns(2)
        with prompt_column:
            if st.button(
                "Prompt graph context",
                key=f"{key_prefix}_graph_context_prompt",
                type="primary" if current_view == "prompt" else "secondary",
                use_container_width=True,
                disabled=current_view == "prompt",
            ):
                current_view = "prompt"
                st.session_state[view_key] = current_view
        with neo4j_column:
            if st.button(
                "Neo4j returned graph",
                key=f"{key_prefix}_graph_context_neo4j",
                type="primary" if current_view == "neo4j" else "secondary",
                use_container_width=True,
                disabled=current_view == "neo4j",
            ):
                current_view = "neo4j"
                st.session_state[view_key] = current_view

        st.markdown("**Knowledge graph context used for this answer**")
        st.caption(
            f"{len(neo4j_items)} graph facts returned from Neo4j; "
            f"{len(prompt_items)} graph facts used in the prompt."
        )
        if graph_error:
            st.warning(f"Graph retrieval was unavailable: {graph_error}")

        if current_view == "neo4j":
            _render_graph_context_view(
                neo4j_items,
                heading="Neo4j returned graph",
                caption="Facts returned by vector graph expansion before reranking.",
            )
            return

        _render_graph_context_view(
            prompt_items,
            heading="Prompt graph context",
            caption="Graph facts retained after reranking and sent to the model.",
        )


def _graph_context_items(value: Any) -> list[dict[str, Any]]:
    return value if isinstance(value, list) else []


def _render_graph_context_view(
    items: list[dict[str, Any]],
    *,
    heading: str,
    caption: str,
) -> None:
    st.markdown(f"**{heading}**")
    st.caption(caption)
    if not items:
        st.info("No graph context is available for this view.")
        return

    _render_graph_network(items)


def _render_graph_network(items: list[dict[str, Any]]) -> None:
    triplet_items = []
    for item in items:
        triplet = _triplet_parts(item)
        if triplet:
            triplet_items.append((item, triplet))

    if not triplet_items:
        st.info("The returned graph context did not include triplets to visualize.")
        return

    if agraph is None:
        for _, triplet in triplet_items:
            st.write(f"{triplet[0]} -> {_relation_label(triplet[1])} -> {triplet[2]}")
        return

    nodes_by_id = {}
    edges = []
    for item, triplet in triplet_items:
        source, relation, target = triplet
        source_id = _node_id(source)
        target_id = _node_id(target)
        nodes_by_id.setdefault(
            source_id,
            Node(id=source_id, label=source, title=source, color="#2563eb", size=24),
        )
        nodes_by_id.setdefault(
            target_id,
            Node(id=target_id, label=target, title=target, color="#0f766e", size=24),
        )
        edges.append(
            Edge(
                source=source_id,
                target=target_id,
                label=_relation_label(relation),
                title=_graph_item_title(item),
                color="#64748b",
            )
        )

    config = Config(width=900, height=420, directed=True, physics=True, hierarchical=False)
    agraph(nodes=list(nodes_by_id.values()), edges=edges, config=config)


def _triplet_parts(item: dict[str, Any]) -> list[str]:
    triplet = item.get("triplet")
    if not isinstance(triplet, list | tuple) or len(triplet) != 3:
        return []
    parts = [str(part).strip() for part in triplet]
    return parts if all(parts) else []


def _relation_label(relation: str) -> str:
    return str(relation).replace("_", " ").lower()


def _node_id(label: str) -> str:
    return "node:" + str(label).casefold()


def _graph_item_title(item: dict[str, Any]) -> str:
    text = str(item.get("text") or "").strip()
    query = str(item.get("cypher_query") or "").strip()
    if query:
        return f"{text}\n\nCypher:\n{query}".strip()
    return text
