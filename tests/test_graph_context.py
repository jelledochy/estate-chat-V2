from backend.app.main import _rag_graph_context_to_chat_context
from backend.RAG import _graph_context_items
from backend.rag.helpers import _infer_triplet_from_cypher_row, build_prompt


def test_graph_context_items_include_cypher_metadata() -> None:
    row = {
        "triplet": ("Alice", "PARENT_OF", "Bob"),
        "text": "Alice parent of Bob.",
        "metadata": {
            "query": "MATCH (a)-[:PARENT_OF]->(b) RETURN a.name, b.name",
            "cypher_row": {"a.name": "Alice", "b.name": "Bob"},
        },
        "rerank_score": 1.25,
    }

    assert _graph_context_items([row]) == [
        {
            "rank": 1,
            "triplet": ["Alice", "PARENT_OF", "Bob"],
            "text": "Alice parent of Bob.",
            "cypher_query": "MATCH (a)-[:PARENT_OF]->(b) RETURN a.name, b.name",
            "cypher_row": {"a.name": "Alice", "b.name": "Bob"},
            "rerank_score": 1.25,
        }
    ]


def test_build_prompt_contains_graph_context_block() -> None:
    row = {
        "triplet": ("Alice", "PARENT_OF", "Bob"),
        "text": "Alice parent of Bob.",
    }

    prompt = build_prompt("How are Alice and Bob related?", [], [row])

    assert "GRAPH CONTEXT:" in prompt
    assert "Alice -> PARENT_OF -> Bob" in prompt
    assert "Alice parent of Bob." in prompt


def test_infers_triplet_from_case_insensitive_where_name() -> None:
    query = """
    MATCH (parent:PERSON)-[:PARENT_OF]->(child:PERSON)
    WHERE toLower(child.name) = toLower('thomas janssen')
    RETURN parent.name AS parent_name, parent.id AS parent_id
    LIMIT 25
    """
    row = {"parent_name": "Hendrik Janssen", "parent_id": "Hendrik Janssen"}

    assert _infer_triplet_from_cypher_row(query, row) == (
        "Hendrik Janssen",
        "PARENT_OF",
        "thomas janssen",
    )


def test_chat_graph_context_preserves_returned_and_prompt_views() -> None:
    rag_result = {
        "neo4j_graph_context": [
            {
                "rank": 1,
                "triplet": ["Alice", "PARENT_OF", "Bob"],
                "text": "Alice parent of Bob.",
                "cypher_query": "MATCH (a)-[:PARENT_OF]->(b) RETURN a.name, b.name",
                "cypher_row": {"a.name": "Alice", "b.name": "Bob"},
            }
        ],
        "prompt_graph_context": [
            {
                "rank": 1,
                "triplet": ["Alice", "PARENT_OF", "Bob"],
                "text": "Alice parent of Bob.",
                "rerank_score": 1.25,
            }
        ],
        "graph_error": "Neo4j unavailable",
    }

    graph_context = _rag_graph_context_to_chat_context(rag_result)

    assert graph_context.neo4j[0].cypher_row == {"a.name": "Alice", "b.name": "Bob"}
    assert graph_context.prompt[0].rerank_score == 1.25
    assert graph_context.error == "Neo4j unavailable"
