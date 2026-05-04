from __future__ import annotations

import datetime as dt
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
EXTRACTED_DATA_DIR = BACKEND_ROOT / "data" / "extracted"

for import_path in (PROJECT_ROOT, BACKEND_ROOT):
    path_text = str(import_path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from backend.RAG import DEFAULT_LLM_MODEL, rag  # noqa: E402

try:
    from app.models.chat import (
        ChatRequest,
        ChatResponse,
        GraphContext,
        GraphContextItem,
        SourceDocument,
    )
except ModuleNotFoundError:
    from backend.app.models.chat import (  # type: ignore[no-redef]
        ChatRequest,
        ChatResponse,
        GraphContext,
        GraphContextItem,
        SourceDocument,
    )


class HealthResponse(BaseModel):
    status: str = "ok"


class GraphSubgraphRequest(BaseModel):
    query: str = ""
    depth: int = Field(default=2, ge=1, le=5)
    max_nodes: int = Field(default=80, ge=1, le=200)
    max_seed_people: int = Field(default=3, ge=1, le=200)
    max_seed_properties: int = Field(default=3, ge=1, le=200)


app = FastAPI(title="Estate Chat API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _parse_date_from_text(text: str) -> dt.date | None:
    patterns = (
        (r"\b(\d{4})-(\d{2})-(\d{2})\b", "%Y-%m-%d"),
        (r"\b(\d{1,2} [A-Z][a-z]+ \d{4})\b", "%d %B %Y"),
        (r"\b([A-Z][a-z]+ \d{1,2}, \d{4})\b", "%B %d, %Y"),
    )
    for pattern, date_format in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        try:
            return dt.datetime.strptime(match.group(1), date_format).date()
        except ValueError:
            continue
    return None


def _parse_amount_from_text(text: str) -> float | None:
    match = re.search(r"\bEUR\s+([0-9][0-9,]*(?:\.[0-9]+)?)\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _parse_property_from_text(text: str) -> dict[str, str] | None:
    match = re.search(
        r"(?:located at|Address:\*\*|Address:)\s*([^.\n]+Belgium)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return {"address": _clean_text(match.group(1).strip(" -*"))}


def _document_summary_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    full_text = str(payload.get("full_text") or "")
    document_id = str(payload.get("document_id") or "").strip()
    filename = str(payload.get("filename") or f"{document_id}.pdf").strip()

    return {
        "document_id": document_id,
        "filename": filename,
        "document_type": str(payload.get("document_type") or "unknown").strip() or "unknown",
        "date": _parse_date_from_text(full_text),
        "parties": payload.get("parties") if isinstance(payload.get("parties"), dict) else {},
        "property": _parse_property_from_text(full_text),
        "amount": _parse_amount_from_text(full_text),
        "references": (
            payload.get("references") if isinstance(payload.get("references"), list) else []
        ),
        "notes": str(payload.get("notes") or "").strip() or None,
    }


@lru_cache(maxsize=1)
def _load_extracted_documents() -> tuple[dict[str, Any], ...]:
    if not EXTRACTED_DATA_DIR.exists():
        return ()

    documents: list[dict[str, Any]] = []
    for path in sorted(EXTRACTED_DATA_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("document_id"):
            documents.append(payload)
    return tuple(documents)


def _documents_by_id() -> dict[str, dict[str, Any]]:
    return {
        str(document.get("document_id")): document
        for document in _load_extracted_documents()
        if document.get("document_id")
    }


def _source_excerpt(
    document_id: str,
    page_numbers: list[int],
    *,
    max_chars: int = 260,
) -> str | None:
    document = _documents_by_id().get(document_id)
    if not document:
        return None

    wanted_pages = {page for page in page_numbers if page > 0}
    snippets: list[str] = []
    pages = document.get("pages") or []
    for page in pages:
        if not isinstance(page, dict):
            continue
        try:
            current_page_number = int(page.get("page_number"))
        except (TypeError, ValueError):
            current_page_number = 0
        if wanted_pages and current_page_number not in wanted_pages:
            continue
        text = _clean_text(page.get("text"))
        if text:
            snippet = text[:max_chars].rstrip() + ("..." if len(text) > max_chars else "")
            if current_page_number > 0:
                snippets.append(f"Page {current_page_number}: {snippet}")
            else:
                snippets.append(snippet)
    return " ".join(snippets) if snippets else None


def _rag_source_to_chat_source(raw_source: dict[str, Any]) -> SourceDocument | None:
    document_id = str(raw_source.get("document_id") or "").strip()
    if not document_id:
        return None

    page_numbers = _page_numbers_from_source(raw_source)

    return SourceDocument(
        document_id=document_id,
        filename=f"{document_id}.pdf",
        document_type=raw_source.get("document_type") or "unknown",
        page_numbers=page_numbers,
        excerpt=_source_excerpt(document_id, page_numbers),
        citation_label=raw_source.get("citation_label"),
    )


def _page_numbers_from_source(raw_source: dict[str, Any]) -> list[int]:
    raw_page_numbers = raw_source.get("page_numbers")
    if not isinstance(raw_page_numbers, list):
        raw_page_numbers = [raw_source.get("page_number")]

    page_numbers: list[int] = []
    for raw_page_number in raw_page_numbers:
        try:
            page_number = int(raw_page_number)
        except (TypeError, ValueError):
            continue
        if page_number > 0 and page_number not in page_numbers:
            page_numbers.append(page_number)
    return sorted(page_numbers)


def _rag_graph_context_to_chat_context(rag_result: dict[str, Any]) -> GraphContext:
    return GraphContext(
        neo4j=_graph_context_items_from_raw(rag_result.get("neo4j_graph_context")),
        prompt=_graph_context_items_from_raw(rag_result.get("prompt_graph_context")),
        error=str(rag_result.get("graph_error") or "").strip() or None,
    )


def _graph_context_items_from_raw(raw_items: Any) -> list[GraphContextItem]:
    if not isinstance(raw_items, list):
        return []

    items: list[GraphContextItem] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        items.append(
            GraphContextItem(
                rank=_positive_int_or_none(raw_item.get("rank")),
                triplet=_graph_triplet(raw_item.get("triplet")),
                text=str(raw_item.get("text") or "").strip(),
                cypher_query=str(raw_item.get("cypher_query") or "").strip() or None,
                cypher_row=(
                    raw_item.get("cypher_row")
                    if isinstance(raw_item.get("cypher_row"), dict)
                    else {}
                ),
                rerank_score=_float_or_none(raw_item.get("rerank_score")),
            )
        )
    return items


def _graph_triplet(value: Any) -> list[str]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        return []
    parts = [str(part).strip() for part in value]
    return parts if all(parts) else []


def _positive_int_or_none(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_uncertain_answer(
    answer: str,
    graph_error: str | None,
    confidence_warning: str | None = None,
) -> tuple[bool, str | None]:
    uncertainty_markers = (
        "insufficient",
        "missing",
        "not enough context",
        "not provided",
        "cannot determine",
        "can't determine",
    )
    answer_lower = answer.casefold()
    is_uncertain = bool(graph_error) or any(
        marker in answer_lower for marker in uncertainty_markers
    )
    if graph_error:
        return (
            True,
            f"Answered from document retrieval. Graph retrieval was unavailable: {graph_error}",
        )
    if confidence_warning:
        return True, confidence_warning
    if is_uncertain:
        return True, "The answer indicates that some supporting context may be missing."
    return False, None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/api/documents")
def list_documents() -> list[dict[str, Any]]:
    return [_document_summary_from_payload(document) for document in _load_extracted_documents()]


@app.get("/api/documents/{document_id}")
def get_document(document_id: str) -> dict[str, Any]:
    document = _documents_by_id().get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

    return {
        **_document_summary_from_payload(document),
        "full_text": str(document.get("full_text") or ""),
        "pages": document.get("pages") or [],
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        rag_result = await run_in_threadpool(
            rag,
            query=request.question,
            model=DEFAULT_LLM_MODEL,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG failed: {exc}") from exc

    sources = [
        source
        for source in (
            _rag_source_to_chat_source(raw_source) for raw_source in rag_result.get("sources", [])
        )
        if source is not None
    ]
    answer = str(rag_result.get("answer") or "").strip() or "No answer returned."
    graph_error = str(rag_result.get("graph_error") or "").strip() or None
    confidence_warning = str(rag_result.get("confidence_warning") or "").strip() or None
    is_uncertain, uncertainty_message = _is_uncertain_answer(
        answer,
        graph_error,
        confidence_warning,
    )

    return ChatResponse(
        answer=answer,
        sources=sources,
        graph_context=_rag_graph_context_to_chat_context(rag_result),
        structured_answer=None,
        is_uncertain=is_uncertain,
        uncertainty_message=uncertainty_message,
    )


@app.post("/api/graph/subgraph")
def query_subgraph(_request: GraphSubgraphRequest) -> dict[str, Any]:
    return {
        "nodes": [],
        "edges": [],
        "meta": {
            "strategy": "not_configured",
            "matched_people": [],
            "matched_properties": [],
            "matched_document_ids": [],
            "relationship_type": None,
        },
    }
