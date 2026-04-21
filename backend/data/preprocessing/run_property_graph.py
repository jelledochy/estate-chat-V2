"""Build the estate-document PropertyGraphIndex with the basic Neo4j config.

Run from the project root:
    python backend/data/preprocessing/run_property_graph.py

The script reads extracted JSON from backend/data/extracted, converts each
extracted page into a LlamaIndex Document, extracts schema-guided estate facts,
and writes the resulting property graph to Neo4j.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from time import time
from typing import Literal

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)

from backend.app.models.documents import ExtractedDocument  # noqa: E402

INPUT_DIR = BACKEND_DIR / "data" / "extracted"

DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_MAX_TRIPLETS_PER_CHUNK = 12
DEFAULT_NUM_WORKERS = 4

NEO4J_URL = os.getenv("NEO4J_URL") or os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4jpassword")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
LLM_MODEL = os.getenv("OPENAI_GRAPH_MODEL") or os.getenv("OPENAI_CHAT_MODEL", DEFAULT_LLM_MODEL)
EMBEDDING_MODEL = os.getenv("OPENAI_GRAPH_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
MAX_TRIPLETS_PER_CHUNK = DEFAULT_MAX_TRIPLETS_PER_CHUNK
NUM_WORKERS = DEFAULT_NUM_WORKERS
STRICT_SCHEMA = False
SHOW_PROGRESS = True

# The graph schema is intentionally estate-document specific. It guides the LLM
# toward facts that will later be useful as normalized graph-context text.
EstateEntity = Literal[
    "PERSON",
    "ORGANIZATION",
    "PROPERTY",
    "DOCUMENT",
    "LOCATION",
    "DATE",
    "MONEY",
    "LEGAL_ROLE",
    "LEGAL_EVENT",
]

EstateRelation = Literal[
    "APPEARED_BEFORE",
    "BENEFICIARY_OF",
    "BORROWER_OF",
    "BUYER_OF",
    "CHILD_OF",
    "CONTAINS_PROPERTY",
    "DATED",
    "DONEE_OF",
    "DONOR_OF",
    "EXECUTED_AT",
    "GRANTS_POWER_OF_ATTORNEY_TO",
    "HAS_AMOUNT",
    "HAS_CADASTRAL_REFERENCE",
    "HAS_PARTY",
    "HAS_ROLE",
    "HEIR_OF",
    "LENDER_OF",
    "LOCATED_AT",
    "MENTIONS",
    "NOTARY_OF",
    "OWNS",
    "PARENT_OF",
    "PAYS",
    "RECEIVES",
    "REFERENCES",
    "REPRESENTS",
    "RESIDES_AT",
    "SECURED_BY",
    "SELLER_OF",
    "SIGNED",
    "SPOUSE_OF",
    "TESTATOR_OF",
    "TRANSFERS_TO",
    "WITNESS_OF",
]

KG_VALIDATION_SCHEMA: dict[str, list[str]] = {
    "PERSON": [
        "APPEARED_BEFORE",
        "BENEFICIARY_OF",
        "BORROWER_OF",
        "BUYER_OF",
        "CHILD_OF",
        "DONEE_OF",
        "DONOR_OF",
        "GRANTS_POWER_OF_ATTORNEY_TO",
        "HAS_ROLE",
        "HEIR_OF",
        "NOTARY_OF",
        "OWNS",
        "PARENT_OF",
        "PAYS",
        "RECEIVES",
        "REPRESENTS",
        "RESIDES_AT",
        "SELLER_OF",
        "SIGNED",
        "SPOUSE_OF",
        "TESTATOR_OF",
        "TRANSFERS_TO",
        "WITNESS_OF",
    ],
    "ORGANIZATION": [
        "BORROWER_OF",
        "BUYER_OF",
        "HAS_ROLE",
        "LENDER_OF",
        "OWNS",
        "PAYS",
        "RECEIVES",
        "REPRESENTS",
        "SELLER_OF",
        "SIGNED",
        "TRANSFERS_TO",
    ],
    "PROPERTY": [
        "HAS_CADASTRAL_REFERENCE",
        "LOCATED_AT",
        "SECURED_BY",
    ],
    "DOCUMENT": [
        "CONTAINS_PROPERTY",
        "DATED",
        "EXECUTED_AT",
        "HAS_AMOUNT",
        "HAS_PARTY",
        "MENTIONS",
        "REFERENCES",
    ],
    "LOCATION": ["MENTIONS"],
    "DATE": ["MENTIONS"],
    "MONEY": ["MENTIONS"],
    "LEGAL_ROLE": ["MENTIONS"],
    "LEGAL_EVENT": ["DATED", "EXECUTED_AT", "HAS_AMOUNT", "MENTIONS"],
}


def collect_extracted_paths(input_dir: Path) -> list[Path]:
    """Return extracted document JSON files in deterministic order."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    return sorted(input_dir.glob("*.json"))


def load_extracted_document(path: Path) -> ExtractedDocument:
    """Load one OCR/PDF extraction artifact into the shared Pydantic model."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ExtractedDocument.model_validate(raw)


def _json_metadata(value: object) -> str:
    """Serialize nested metadata because graph/vector stores prefer scalar properties."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _display_path(path: Path) -> str:
    """Prefer project-relative provenance, but allow custom input directories."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _base_metadata(document: ExtractedDocument, source_path: Path) -> dict[str, object]:
    """Metadata copied onto every LlamaIndex source document/page."""
    return {
        "source_kind": "estate_extracted_json",
        "source_path": _display_path(source_path),
        "document_id": document.document_id,
        "filename": document.filename,
        "document_type": str(document.document_type),
        "document_date": document.date.isoformat() if document.date else "",
        "parties_json": _json_metadata(document.parties),
        "references_json": _json_metadata(document.references),
        "notes": document.notes or "",
    }


def build_llama_documents(
    extracted_documents: list[tuple[Path, ExtractedDocument]],
) -> list[object]:
    """
    Convert extracted estate documents into LlamaIndex documents.

    One LlamaIndex document is created per extracted page instead of per PDF.
    That keeps page provenance attached to graph source chunks, which matters
    when graph facts are later rendered back into prompt context.
    """
    try:
        from llama_index.core import Document
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LlamaIndex 0.14+ is required for PropertyGraphIndex. "
            "Run `poetry install` after updating pyproject.toml."
        ) from exc

    llama_documents: list[object] = []
    for source_path, document in extracted_documents:
        base_metadata = _base_metadata(document, source_path)

        if document.pages:
            for page in document.pages:
                text = page.text.strip()
                if not text:
                    continue
                llama_documents.append(
                    Document(
                        text=text,
                        id_=f"{document.document_id}-p{page.page_number:03d}",
                        metadata={
                            **base_metadata,
                            "source_id": f"{document.document_id}-p{page.page_number:03d}",
                            "page_number": page.page_number,
                        },
                    )
                )
            continue

        text = document.full_text.strip()
        if text:
            llama_documents.append(
                Document(
                    text=text,
                    id_=f"{document.document_id}-full",
                    metadata={**base_metadata, "source_id": f"{document.document_id}-full"},
                )
            )

    return llama_documents


def build_graph_store() -> object:
    """
    Create the Neo4j property graph store used by the basic graph build.

    Connection settings come from environment variables with repo defaults:
    NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD, and NEO4J_DATABASE.
    """
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

    return Neo4jPropertyGraphStore(
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        url=NEO4J_URL,
        database=NEO4J_DATABASE,
    )


def verify_neo4j_connection() -> None:
    """
    Fail fast if Neo4j is unavailable before LlamaIndex starts graph ingestion.

    The URL is intentionally still environment-driven:
    - host machine: bolt://localhost:7687
    - Compose service/container network: bolt://neo4j:7687
    """
    from neo4j import GraphDatabase

    try:
        driver = GraphDatabase.driver(
            NEO4J_URL,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            connection_timeout=3,
        )
        with driver:
            driver.verify_connectivity()
    except Exception as exc:
        raise RuntimeError(
            "Could not connect to Neo4j before building the property graph.\n"
            f"Configured URL: {NEO4J_URL}\n"
            "Start Neo4j first, or set NEO4J_URL/NEO4J_URI to the reachable Bolt URL.\n"
            "Use bolt://localhost:7687 when running from your host machine.\n"
            "Use bolt://neo4j:7687 when running inside the Compose network."
        ) from exc


def build_property_graph_index(
    *,
    llama_documents: list[object],
    graph_store: object,
) -> object:
    """
    Build the LlamaIndex PropertyGraphIndex from prepared page documents.

    The index does three things:
    1. chunks the input documents into LlamaIndex nodes,
    2. asks the LLM extractor for schema-guided estate-document triples,
    3. stores entities, relations, source chunks, and embeddings in the graph store.
    """
    try:
        from llama_index.core import PropertyGraphIndex
        from llama_index.core.indices.property_graph import (
            ImplicitPathExtractor,
            SchemaLLMPathExtractor,
        )
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing modern LlamaIndex property-graph packages. "
            "Run `poetry install` and confirm llama-index==0.14.20 plus "
            "llama-index-graph-stores-neo4j==0.7.0 are installed."
        ) from exc

    llm = OpenAI(model=LLM_MODEL, temperature=0.0)
    embed_model = OpenAIEmbedding(model_name=EMBEDDING_MODEL)
    kg_extractors = [
        ImplicitPathExtractor(),
        SchemaLLMPathExtractor(
            llm=llm,
            possible_entities=EstateEntity,
            possible_relations=EstateRelation,
            kg_validation_schema=KG_VALIDATION_SCHEMA,
            strict=STRICT_SCHEMA,
            num_workers=NUM_WORKERS,
            max_triplets_per_chunk=MAX_TRIPLETS_PER_CHUNK,
        ),
    ]

    return PropertyGraphIndex.from_documents(
        llama_documents,
        llm=llm,
        embed_model=embed_model,
        kg_extractors=kg_extractors,
        property_graph_store=graph_store,
        embed_kg_nodes=True,
        show_progress=SHOW_PROGRESS,
    )


def main() -> int:

    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        print(
            "OPENAI_API_KEY is not set. Add it to your environment or /workspace/.env and retry.",
            file=sys.stderr,
        )
        return 1
    if MAX_TRIPLETS_PER_CHUNK < 1:
        print("MAX_TRIPLETS_PER_CHUNK must be >= 1", file=sys.stderr)
        return 1
    if NUM_WORKERS < 1:
        print("NUM_WORKERS must be >= 1", file=sys.stderr)
        return 1

    t0 = time()

    try:
        extracted_paths = collect_extracted_paths(INPUT_DIR)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not extracted_paths:
        print(f"No extracted JSON files found in {INPUT_DIR}", file=sys.stderr)
        return 1

    extracted_documents: list[tuple[Path, ExtractedDocument]] = []
    failures: list[tuple[Path, str]] = []
    for path in extracted_paths:
        try:
            extracted_documents.append((path, load_extracted_document(path)))
        except Exception as exc:
            failures.append((path, str(exc)))
            print(f"Skipping {path.name}: {exc}", file=sys.stderr)

    if not extracted_documents:
        print("No valid extracted documents to process.", file=sys.stderr)
        return 1

    try:
        llama_documents = build_llama_documents(extracted_documents)
        verify_neo4j_connection()
        graph_store = build_graph_store()
        build_property_graph_index(
            llama_documents=llama_documents,
            graph_store=graph_store,
        )
    except Exception as exc:
        print(f"Error while building property graph: {exc}", file=sys.stderr)
        return 1

    took = time() - t0
    print(
        "Finished property graph build. "
        f"store=neo4j documents={len(extracted_documents)} "
        f"llama_documents={len(llama_documents)} failed_documents={len(failures)} "
        f"llm_model={LLM_MODEL} embedding_model={EMBEDDING_MODEL} "
        f"strict_schema={STRICT_SCHEMA} took_seconds={took:.1f}"
    )
    print(
        "Graph facts were written to Neo4j. "
        "Use Neo4j Browser to inspect the property graph."
    )

    if failures:
        print("Some files were skipped:", file=sys.stderr)
        for path, error in failures:
            print(f"  - {path.name}: {error}", file=sys.stderr)

    close = getattr(graph_store, "close", None)
    if callable(close):
        close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
