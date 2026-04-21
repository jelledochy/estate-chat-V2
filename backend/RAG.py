from __future__ import annotations

import argparse
import json
import os
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


prompt_template = """
You are an estate-document assistant.
Answer the QUESTION based only on the CONTEXT from retrieved notarial documents.
If the context is insufficient, clearly say what is missing and do not invent details.
When you cite facts, mention source ids in square brackets like [1], [2].

QUESTION:
{question}

CONTEXT:
{context}
""".strip()


entry_template = """
[Source {rank}]
chunk_id: {chunk_id}
document_id: {document_id}
document_type: {document_type}
page_number: {page_number}
distance: {distance:.6f}
people: {people}
dates: {dates}
years: {years}
amounts_eur: {amounts_eur}
roles: {roles}
text:
{text}
""".strip()


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


def build_prompt(query: str, search_results: list[dict[str, Any]]) -> str:
    context_entries: list[str] = []
    for rank, row in enumerate(search_results, start=1):
        metadata = row.get("metadata", {})
        context_entries.append(
            entry_template.format(
                rank=rank,
                chunk_id=row.get("chunk_id", ""),
                document_id=metadata.get("document_id", "unknown"),
                document_type=metadata.get("document_type", "unknown"),
                page_number=metadata.get("page_number", 0),
                distance=float(row.get("distance", 0.0)),
                people=_parse_json_list(metadata.get("person_names_mentioned")),
                dates=_parse_json_list(metadata.get("mentioned_dates")),
                years=_parse_json_list(metadata.get("mentioned_years")),
                amounts_eur=_parse_json_list(metadata.get("amounts_eur")),
                roles=_parse_json_list(metadata.get("legal_roles_mentioned")),
                text=(row.get("text") or "").strip(),
            )
        )

    context = "\n\n".join(context_entries) if context_entries else "No relevant context found."
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

    search_results = search(
        openai_client=openai_client,
        collection=collection,
        query=query,
        top_k=top_k,
    )
    prompt = build_prompt(query, search_results)
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
            }
        )

    return {
        "answer": answer,
        "model_used": model,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "collection": COLLECTION_NAME,
        "top_k": top_k,
        "response_time_seconds": round(took, 3),
        "sources": sources,
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
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks")
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
