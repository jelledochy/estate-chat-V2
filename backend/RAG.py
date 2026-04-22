from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from time import time
from typing import Any

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_PATH = PROJECT_ROOT / "backend" / "data" / "chroma_db"
COLLECTION_NAME = "estate_documents"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
DEFAULT_TOP_K = 5
DEFAULT_LLM_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
CROSS_ENCODER_MODEL_NAME = os.getenv(
    "RAG_CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
RETRIEVAL_CANDIDATE_MULTIPLIER = int(os.getenv("RAG_RETRIEVAL_CANDIDATE_MULTIPLIER", "4"))
GRAPH_RESULT_LIMIT = int(os.getenv("RAG_GRAPH_RESULT_LIMIT", "25"))
NEO4J_URL = os.getenv("NEO4J_URL") or os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4jpassword")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

_CROSS_ENCODER: Any | None = None


prompt_template = """
You are an estate-document assistant.
Answer the QUESTION based only on the CONTEXT from retrieved notarial documents and graph facts.
If the context is insufficient, clearly say what is missing and do not invent details.
When you cite facts, mention source ids in square brackets like [Document 1] or [Graph 1].

QUESTION:
{question}

CONTEXT:
{context}
""".strip()


document_entry_template = """
[Document {rank}]
chunk_id: {chunk_id}
document_id: {document_id}
document_type: {document_type}
page_number: {page_number}
distance: {distance:.6f}
rerank_score: {rerank_score:.6f}
people: {people}
dates: {dates}
years: {years}
amounts_eur: {amounts_eur}
roles: {roles}
text:
{text}
""".strip()


graph_entry_template = """
[Graph {rank}]
fact_id: {fact_id}
rerank_score: {rerank_score:.6f}
triplet: {triplet}
text:
{text}
""".strip()


GRAPH_ALLOWED_OUTPUT_FIELDS = [
    "id",
    "label",
    "name",
    "text",
    "triplet",
    "type",
    "document_id",
    "document_type",
    "page_number",
]


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


def load_collection() -> tuple[OpenAI, chromadb.Collection]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
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


def load_graph_retriever(*, model: str = DEFAULT_LLM_MODEL) -> Any:
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
        allowed_output_field=GRAPH_ALLOWED_OUTPUT_FIELDS,
    )


def graph_search(*, query: str, model: str = DEFAULT_LLM_MODEL) -> list[dict[str, Any]]:
    retriever = load_graph_retriever(model=model)
    nodes = retriever.retrieve(query)

    facts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, node_with_score in enumerate(nodes[:GRAPH_RESULT_LIMIT], start=1):
        fact = _node_to_graph_fact(node_with_score, idx)
        if fact is None:
            continue
        key = fact["text"].lower()
        if key in seen:
            continue
        seen.add(key)
        facts.append(fact)
    return facts


def _node_to_graph_fact(node_with_score: Any, rank: int) -> dict[str, Any] | None:
    node = getattr(node_with_score, "node", node_with_score)
    metadata = getattr(node, "metadata", {}) or {}
    text = _node_content(node)
    triplet = _extract_triplet(metadata.get("triplet")) or _extract_triplet(node)
    if triplet is None:
        triplet = _extract_triplet_from_text(text)

    fact_text = _triplet_to_text(triplet) if triplet else text.strip()
    if not fact_text:
        return None

    return {
        "kind": "graph",
        "fact_id": f"graph-{rank}",
        "triplet": triplet,
        "text": fact_text,
        "metadata": metadata,
        "graph_score": getattr(node_with_score, "score", None),
    }


def _node_content(node: Any) -> str:
    get_content = getattr(node, "get_content", None)
    if callable(get_content):
        return str(get_content()).strip()
    return str(getattr(node, "text", "") or "").strip()


def _extract_triplet(value: Any) -> tuple[str, str, str] | None:
    if isinstance(value, (list, tuple)) and len(value) == 3:
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
    return documents, facts


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
            document_entry_template.format(
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
            graph_entry_template.format(
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

    context = (
        "\n\n".join(context_sections)
        if context_sections
        else "No relevant context found."
    )
    return prompt_template.format(question=query, context=context).strip()


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


def rag(
    *,
    query: str,
    model: str = DEFAULT_LLM_MODEL,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    t0 = time()
    openai_client, collection = load_collection()
    candidate_k = max(top_k, top_k * RETRIEVAL_CANDIDATE_MULTIPLIER)

    search_results = search(
        openai_client=openai_client,
        collection=collection,
        query=query,
        top_k=candidate_k,
    )
    graph_error = None
    try:
        graph_results = graph_search(query=query, model=model)
    except Exception as exc:
        graph_results = []
        graph_error = str(exc)

    search_results, graph_results = rerank_context(
        query=query,
        document_results=search_results,
        graph_results=graph_results,
        top_k=top_k,
    )
    prompt = build_prompt(query, search_results, graph_results)
    answer, token_stats = llm(openai_client=openai_client, prompt=prompt, model=model)

    took = time() - t0
    sources = []
    for rank, row in enumerate(search_results, start=1):
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
    graph_sources = []
    for rank, row in enumerate(graph_results, start=1):
        graph_sources.append(
            {
                "rank": rank,
                "fact_id": row.get("fact_id"),
                "triplet": _triplet_display(row.get("triplet")),
                "rerank_score": row.get("rerank_score"),
                "graph_score": row.get("graph_score"),
            }
        )

    return {
        "answer": answer,
        "model_used": model,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reranker_model": CROSS_ENCODER_MODEL_NAME,
        "collection": COLLECTION_NAME,
        "top_k": top_k,
        "response_time_seconds": round(took, 3),
        "sources": sources,
        "graph_sources": graph_sources,
        "graph_error": graph_error,
        "prompt_tokens": token_stats["prompt_tokens"],
        "completion_tokens": token_stats["completion_tokens"],
        "total_tokens": token_stats["total_tokens"],
        "prompt": prompt,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run RAG over the best-performing embedding index "
            "(text-embedding-3-small on estate_documents)."
        )
    )
    parser.add_argument("query", help="User question")
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL, help="OpenAI chat model")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of reranked document chunks / graph facts to include",
    )
    parser.add_argument(
        "--where",
        default=None,
        help='Optional Chroma metadata filter as JSON, e.g. \'{"document_type":"mortgage_deed"}\'',
    )
    args = parser.parse_args()

    if args.top_k < 1:
        print("--top-k must be >= 1", file=sys.stderr)
        return 1

    try:
        result = rag(
            query=args.query,
            model=args.model,
            top_k=args.top_k,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
