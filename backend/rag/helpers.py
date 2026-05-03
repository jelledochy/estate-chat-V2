from __future__ import annotations

import ast
import json
import re
from typing import Any

from openai import OpenAI

from backend.rag.constants import (
    CROSS_ENCODER_MODEL_NAME,
    DEFAULT_LLM_MODEL,
    DEFAULT_TOP_K,
    DOCUMENT_ENTRY_TEMPLATE,
    GRAPH_ENTRY_TEMPLATE,
    GRAPH_RESULT_LIMIT,
    PROMPT_TEMPLATE,
)

_CROSS_ENCODER: Any | None = None

_CYPHER_CODE_BLOCK_RE = re.compile(
    r"```(?:\s*cypher)?\s*(.*?)\s*```",
    flags=re.IGNORECASE | re.DOTALL,
)
_CYPHER_START_RE = re.compile(
    r"\b(?:MATCH|OPTIONAL MATCH|WITH|RETURN|CALL|UNWIND|MERGE|CREATE|DELETE|SET|REMOVE|"
    r"FOREACH|LOAD CSV|USE|SHOW|START DATABASE|STOP DATABASE|ALTER|DROP|LIMIT|ORDER BY|"
    r"SKIP|OFFSET|WHERE)\b",
    flags=re.IGNORECASE,
)
_CYPHER_LIMIT_RE = re.compile(r"\bLIMIT\s+(?P<limit>\d+)\b", flags=re.IGNORECASE)
_GENERATED_CYPHER_RE = re.compile(
    r"Generated Cypher query:\s*(?P<query>.*?)(?:\n\s*Cypher Response:|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)
_CYPHER_RESPONSE_RE = re.compile(
    r"Cypher Response:\s*(?P<response>\[.*?\])\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)
_CYPHER_RELATION_RE = re.compile(
    r"(?P<left>\([^)]*\))\s*(?P<left_connector><-|-)\s*"
    r"\[\s*:\s*(?P<relation>[A-Z_]+)[^\]]*\]\s*"
    r"(?P<right_connector>->|-)\s*(?P<right>\([^)]*\))",
    flags=re.IGNORECASE | re.DOTALL,
)
_CYPHER_NODE_RE = re.compile(
    r"\(\s*(?P<variable>[A-Za-z_][A-Za-z0-9_]*)?\s*"
    r"(?::\s*(?P<label>[A-Za-z_][A-Za-z0-9_]*))?\s*"
    r"(?P<properties>\{[^)]*\})?\s*\)",
    flags=re.DOTALL,
)
_CYPHER_NAME_PROPERTY_RE = re.compile(
    r"\bname\s*:\s*(['\"])(?P<name>.*?)\1",
    flags=re.IGNORECASE | re.DOTALL,
)


def _parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _parse_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _clean_cypher_query(cypher: str) -> str:
    text = (cypher or "").strip()
    if not text:
        return ""

    fence_match = _CYPHER_CODE_BLOCK_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    text = "\n".join(lines).strip("`").strip()
    start_match = _CYPHER_START_RE.search(text)
    if start_match:
        text = text[start_match.start() :].strip()
    return text


def _cypher_validator(cypher: str) -> str:
    """Clean generated Cypher and enforce a safe minimum row limit."""
    cleaned = _clean_cypher_query(cypher)
    if not cleaned:
        raise ValueError("Graph retriever produced an empty Cypher query.")
    return _ensure_cypher_limit(cleaned)


def _ensure_cypher_limit(cypher: str) -> str:
    """Append or widen LIMIT so multi-answer relationship queries are not cut off."""
    limit_match = _CYPHER_LIMIT_RE.search(cypher)
    if limit_match:
        limit = int(limit_match.group("limit"))
        if limit >= GRAPH_RESULT_LIMIT:
            return cypher
        return (
            cypher[: limit_match.start("limit")]
            + str(GRAPH_RESULT_LIMIT)
            + cypher[limit_match.end("limit") :]
        )

    suffix = ""
    query = cypher.rstrip()
    if query.endswith(";"):
        query = query[:-1].rstrip()
        suffix = ";"
    return f"{query}\nLIMIT {GRAPH_RESULT_LIMIT}{suffix}"


def load_cross_encoder() -> Any:
    global _CROSS_ENCODER
    if _CROSS_ENCODER is not None:
        return _CROSS_ENCODER
    try:
        from sentence_transformers import CrossEncoder
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "sentence-transformers is required for CrossEncoder reranking. "
            "Install it or set up the project environment with this dependency."
        ) from exc

    _CROSS_ENCODER = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    return _CROSS_ENCODER


def _node_to_graph_facts(
    node_with_score: Any,
    *,
    max_facts: int | None = None,
) -> list[dict[str, Any]]:
    """Convert one graph retriever node into one or more atomic graph facts."""
    if max_facts is not None and max_facts <= 0:
        return []

    node = getattr(node_with_score, "node", node_with_score)
    metadata = getattr(node, "metadata", {}) or {}
    text = _node_content(node)
    if _is_empty_cypher_response(text):
        return []

    cypher_query = _extract_cypher_query(text, metadata)
    cypher_rows = _extract_cypher_response_rows(text, metadata)
    if cypher_rows:
        facts = _cypher_rows_to_graph_facts(
            query=cypher_query,
            rows=cypher_rows,
            metadata=metadata,
            max_facts=max_facts,
        )
        if facts:
            return facts

    triplet = _extract_triplet(metadata.get("triplet")) or _extract_triplet(node)
    if triplet is None:
        triplet = _extract_triplet_from_text(text)

    fact_text = _triplet_to_text(triplet) if triplet else text.strip()
    if not fact_text:
        return []

    return [
        {
            "kind": "graph",
            "fact_id": "",
            "triplet": triplet,
            "text": fact_text,
            "metadata": metadata,
        }
    ]


def _extract_cypher_query(text: str, metadata: dict[str, Any]) -> str:
    """Read the generated Cypher query from metadata or rendered node text."""
    query = metadata.get("query")
    if isinstance(query, str) and query.strip():
        return _clean_cypher_query(query)

    match = _GENERATED_CYPHER_RE.search(text)
    if not match:
        return ""
    return _clean_cypher_query(match.group("query"))


def _extract_cypher_response_rows(text: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Read Cypher result rows from metadata or the rendered response text."""
    response = metadata.get("response")
    rows = _coerce_cypher_rows(response)
    if rows:
        return rows

    match = _CYPHER_RESPONSE_RE.search(text)
    if not match:
        return []

    try:
        parsed = ast.literal_eval(match.group("response"))
    except (SyntaxError, ValueError):
        return []
    return _coerce_cypher_rows(parsed)


def _coerce_cypher_rows(value: Any) -> list[dict[str, Any]]:
    """Normalize Cypher responses to a list of row dictionaries."""
    if isinstance(value, str) and value.strip():
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value)
            except (SyntaxError, ValueError, json.JSONDecodeError):
                continue
            return _coerce_cypher_rows(parsed)

    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        return []

    rows: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _cypher_rows_to_graph_facts(
    *,
    query: str,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    max_facts: int | None = None,
) -> list[dict[str, Any]]:
    """Split Cypher result rows into one graph fact per returned row."""
    facts: list[dict[str, Any]] = []
    for row in rows:
        triplet = _infer_triplet_from_cypher_row(query, row)
        text = _triplet_to_text(triplet) if triplet else _format_cypher_row(row)
        if not text:
            continue
        facts.append(
            {
                "kind": "graph",
                "fact_id": "",
                "triplet": triplet,
                "text": text,
                "metadata": {
                    **metadata,
                    "query": query or metadata.get("query"),
                    "cypher_row": row,
                },
            }
        )
        if max_facts is not None and len(facts) >= max_facts:
            break
    return facts


def _infer_triplet_from_cypher_row(
    query: str,
    row: dict[str, Any],
) -> tuple[str, str, str] | None:
    """Infer subject-relation-object from a single-hop Cypher query and row."""
    if not query:
        return None

    match = _CYPHER_RELATION_RE.search(query)
    if not match:
        return None

    left = _parse_cypher_node(match.group("left"))
    right = _parse_cypher_node(match.group("right"))
    relation = match.group("relation").upper()

    if match.group("left_connector") == "<-" and match.group("right_connector") == "-":
        subject_node, object_node = right, left
    else:
        subject_node, object_node = left, right

    subject = _cypher_node_name(subject_node, row)
    object_ = _cypher_node_name(object_node, row)
    if not subject or not object_:
        return None
    return (subject, relation, object_)


def _parse_cypher_node(text: str) -> dict[str, str]:
    """Extract a Cypher node variable and inline name property."""
    match = _CYPHER_NODE_RE.fullmatch(text.strip())
    if not match:
        return {}

    properties = match.group("properties") or ""
    name_match = _CYPHER_NAME_PROPERTY_RE.search(properties)
    return {
        "variable": match.group("variable") or "",
        "label": match.group("label") or "",
        "name": name_match.group("name").strip() if name_match else "",
    }


def _cypher_node_name(node: dict[str, str], row: dict[str, Any]) -> str:
    """Resolve a node name from inline Cypher properties or returned row keys."""
    if node.get("name"):
        return node["name"]

    variable = node.get("variable", "")
    if variable:
        candidate_keys = [
            f"{variable}_name",
            f"{variable}.name",
            f"{variable}Name",
            variable,
        ]
        candidate_keys.extend(
            key
            for key in row
            if key.lower().startswith(variable.lower()) and "name" in key.lower()
        )
        for key in candidate_keys:
            value = row.get(key)
            if value not in (None, ""):
                return str(value).strip()

    if len(row) == 1:
        value = next(iter(row.values()))
        return "" if value in (None, "") else str(value).strip()
    return ""


def _format_cypher_row(row: dict[str, Any]) -> str:
    """Fallback text for Cypher rows that cannot be shaped as triplets."""
    parts = [f"{key}: {value}" for key, value in row.items() if value not in (None, "")]
    return "; ".join(parts).strip()


def _is_empty_cypher_response(text: str) -> bool:
    return bool(re.search(r"Cypher Response:\s*\[\s*\]\s*$", text.strip(), flags=re.DOTALL))


def _node_content(node: Any) -> str:
    get_content = getattr(node, "get_content", None)
    if callable(get_content):
        return str(get_content()).strip()
    return str(getattr(node, "text", "") or "").strip()


def _extract_triplet(value: Any) -> tuple[str, str, str] | None:
    if isinstance(value, list | tuple) and len(value) == 3:
        return tuple(str(part).strip() for part in value)  # type: ignore[return-value]

    if isinstance(value, str):
        parsed = _parse_json_list(value)
        if len(parsed) == 3:
            return tuple(str(part).strip() for part in parsed)  # type: ignore[return-value]
        parsed_dict = _parse_json_dict(value)
        if parsed_dict:
            return _extract_triplet(parsed_dict)

    metadata = getattr(value, "metadata", None)
    if isinstance(metadata, dict) and metadata.get("triplet"):
        return _extract_triplet(metadata["triplet"])

    if isinstance(value, dict):
        subject = value.get("subject") or value.get("source") or value.get("head")
        predicate = value.get("predicate") or value.get("relation") or value.get("edge")
        object_ = value.get("object") or value.get("target") or value.get("tail")
        if subject and predicate and object_:
            return (str(subject).strip(), str(predicate).strip(), str(object_).strip())
        if value.get("triplet"):
            return _extract_triplet(value["triplet"])

    return None


def _extract_triplet_from_text(text: str) -> tuple[str, str, str] | None:
    for pattern in (
        r"^\s*\(?\s*([^,\n()]+)\s*,\s*([^,\n()]+)\s*,\s*([^,\n()]+)\s*\)?\s*$",
        r"^\s*(.+?)\s*[-=]+>\s*(.+?)\s*[-=]+>\s*(.+?)\s*$",
    ):
        match = re.search(pattern, text.strip(), flags=re.MULTILINE)
        if match:
            return tuple(part.strip() for part in match.groups())  # type: ignore[return-value]
    return None


def _triplet_to_text(triplet: tuple[str, str, str]) -> str:
    subject, predicate, object_ = triplet
    predicate_text = predicate.replace("_", " ").lower()
    return f"{subject} {predicate_text} {object_}."


def _triplet_display(triplet: tuple[str, str, str] | None) -> str:
    if triplet is None:
        return ""
    return f"{triplet[0]} -> {triplet[1]} -> {triplet[2]}"


def rerank_context(
    *,
    query: str,
    document_results: list[dict[str, Any]],
    graph_results: list[dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for row in document_results:
        candidates.append({**row, "kind": "document", "rerank_text": row.get("text", "")})
    for row in graph_results:
        triplet = _triplet_display(row.get("triplet"))
        rerank_text = f"{triplet}\n{row.get('text', '')}".strip()
        candidates.append({**row, "kind": "graph", "rerank_text": rerank_text})

    if not candidates:
        return [], []

    cross_encoder = load_cross_encoder()
    scores = cross_encoder.predict(
        [(query, candidate.get("rerank_text", "")) for candidate in candidates]
    )
    for candidate, score in zip(candidates, scores, strict=False):
        candidate["rerank_score"] = float(score)

    ranked = sorted(
        candidates,
        key=lambda row: float(row.get("rerank_score", float("-inf"))),
        reverse=True,
    )[:top_k]
    documents = [row for row in ranked if row.get("kind") == "document"]
    facts = [row for row in ranked if row.get("kind") == "graph"]
    for rank, row in enumerate(facts, start=1):
        row["fact_id"] = f"graph-{rank}"
    return documents, facts


def _low_confidence_warning(
    search_results: list[dict[str, Any]],
    graph_results: list[dict[str, Any]],
) -> str | None:
    """Warn when every retained context item has a negative reranker score."""
    scores = [
        float(row["rerank_score"])
        for row in [*search_results, *graph_results]
        if row.get("rerank_score") is not None
    ]
    if scores and all(score < 0 for score in scores):
        return (
            "Low confidence: all retrieved context has negative reranker scores, "
            "so the answer may be incomplete or unsupported."
        )
    return None


def build_prompt(
    query: str,
    search_results: list[dict[str, Any]],
    graph_results: list[dict[str, Any]] | None = None,
) -> str:
    document_entries: list[str] = []
    graph_entries: list[str] = []
    for rank, row in enumerate(search_results, start=1):
        metadata = row.get("metadata", {})
        document_entries.append(
            DOCUMENT_ENTRY_TEMPLATE.format(
                rank=rank,
                chunk_id=row.get("chunk_id", ""),
                document_id=metadata.get("document_id", "unknown"),
                document_type=metadata.get("document_type", "unknown"),
                page_number=metadata.get("page_number", 0),
                distance=float(row.get("distance", 0.0)),
                rerank_score=float(row.get("rerank_score", 0.0)),
                people=_parse_json_list(metadata.get("person_names_mentioned")),
                dates=_parse_json_list(metadata.get("mentioned_dates")),
                years=_parse_json_list(metadata.get("mentioned_years")),
                amounts_eur=_parse_json_list(metadata.get("amounts_eur")),
                roles=_parse_json_list(metadata.get("legal_roles_mentioned")),
                text=(row.get("text") or "").strip(),
            )
        )

    for rank, row in enumerate(graph_results or [], start=1):
        graph_entries.append(
            GRAPH_ENTRY_TEMPLATE.format(
                rank=rank,
                fact_id=row.get("fact_id", ""),
                rerank_score=float(row.get("rerank_score", 0.0)),
                triplet=_triplet_display(row.get("triplet")),
                text=(row.get("text") or "").strip(),
            )
        )

    context_sections: list[str] = []
    if document_entries:
        context_sections.append("DOCUMENT RETRIEVAL:\n" + "\n\n".join(document_entries))
    if graph_entries:
        context_sections.append("GRAPH RETRIEVAL:\n" + "\n\n".join(graph_entries))

    context = "\n\n".join(context_sections) if context_sections else "No relevant context found."
    return PROMPT_TEMPLATE.format(question=query, context=context).strip()


def llm(
    *,
    openai_client: OpenAI,
    prompt: str,
    model: str = DEFAULT_LLM_MODEL,
) -> tuple[str, dict[str, int]]:
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = (response.choices[0].message.content or "").strip()
    usage = response.usage
    token_stats = {
        "prompt_tokens": int(usage.prompt_tokens if usage else 0),
        "completion_tokens": int(usage.completion_tokens if usage else 0),
        "total_tokens": int(usage.total_tokens if usage else 0),
    }
    return answer, token_stats
