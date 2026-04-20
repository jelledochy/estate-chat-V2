from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.models.documents import ChunkMetadata, DocumentChunk, ExtractedDocument

# One-time preprocessing configuration (fixed values)
INPUT_DIR = BACKEND_DIR / "data" / "extracted"
CHROMA_PATH = BACKEND_DIR / "data" / "chroma_db"
COLLECTION_NAME = "estate_documents"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 64

NAME_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:-[A-Z][a-z]+)?(?:\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)?)+)\b")
DISALLOWED_NAME_TOKENS = {
    "and",
    "article",
    "authority",
    "bank",
    "belgium",
    "bruges",
    "certificate",
    "date",
    "document",
    "family",
    "id",
    "issue",
    "mortgage",
    "notarial",
    "notary",
    "place",
    "public",
    "relationship",
    "seal",
    "signature",
    "type",
}


def collect_extracted_paths(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    return sorted(input_dir.glob("*.json"))


def load_extracted_document(path: Path) -> ExtractedDocument:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ExtractedDocument.model_validate(raw)


def extract_person_names(text: str) -> list[str]:
    names: set[str] = set()

    for match in NAME_PATTERN.finditer(text):
        candidate = " ".join(match.group(1).split())
        lowered_tokens = [token.strip(".,:;()[]{}").lower() for token in candidate.split()]
        if any(token in DISALLOWED_NAME_TOKENS for token in lowered_tokens):
            continue
        names.add(candidate)

    return sorted(names)


def build_chunks(
    documents: list[ExtractedDocument],
    splitter: RecursiveCharacterTextSplitter,
) -> tuple[list[DocumentChunk], dict[str, int]]:
    chunks: list[DocumentChunk] = []
    per_document_counts: dict[str, int] = {}

    for document in documents:
        chunk_index = 0

        segments = (
            [(page.page_number, page.text) for page in document.pages]
            if document.pages
            else [(0, document.full_text)]
        )

        for page_number, raw_text in segments:
            text = raw_text.strip()
            if not text:
                continue

            for chunk_text in splitter.split_text(text):
                cleaned = chunk_text.strip()
                if not cleaned:
                    continue

                chunk_id = f"{document.document_id}-p{page_number:03d}-c{chunk_index:04d}"
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    embedding_id=chunk_id,
                    text=cleaned,
                    metadata=ChunkMetadata(
                        document_id=document.document_id,
                        document_type=document.document_type,
                        chunk_index=chunk_index,
                        page_numbers=[page_number] if page_number else [],
                        person_names_mentioned=extract_person_names(cleaned),
                    ),
                )
                chunks.append(chunk)
                chunk_index += 1

        per_document_counts[document.document_id] = chunk_index

    return chunks, per_document_counts


def chunk_to_chroma_metadata(chunk: DocumentChunk) -> dict[str, str | int]:
    page_number = chunk.metadata.page_numbers[0] if chunk.metadata.page_numbers else 0
    return {
        "document_id": chunk.metadata.document_id,
        "document_type": str(chunk.metadata.document_type),
        "page_number": page_number,
        "person_names_mentioned": json.dumps(
            chunk.metadata.person_names_mentioned, ensure_ascii=False
        ),
    }


def upsert_chunks(
    *,
    chunks: list[DocumentChunk],
    openai_client: OpenAI,
    embedding_model_name: str,
    collection: chromadb.Collection,
    batch_size: int,
) -> None:
    total = len(chunks)

    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        ids = [chunk.chunk_id for chunk in batch]
        documents = [chunk.text for chunk in batch]
        metadatas = [chunk_to_chroma_metadata(chunk) for chunk in batch]
        response = openai_client.embeddings.create(
            model=embedding_model_name,
            input=documents,
        )
        embeddings = [item.embedding for item in response.data]

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"Upserted chunks {start + 1}-{start + len(batch)} / {total}")


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        print(
            "OPENAI_API_KEY is not set. Add it to your environment or /workspace/.env and retry.",
            file=sys.stderr,
        )
        return 1

    try:
        extracted_paths = collect_extracted_paths(INPUT_DIR)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not extracted_paths:
        print(f"No extracted JSON files found in {INPUT_DIR}", file=sys.stderr)
        return 1

    documents: list[ExtractedDocument] = []
    failures: list[tuple[Path, str]] = []

    for path in extracted_paths:
        try:
            documents.append(load_extracted_document(path))
        except Exception as exc:
            failures.append((path, str(exc)))
            print(f"Skipping {path.name}: {exc}", file=sys.stderr)

    if not documents:
        print("No valid extracted documents to process.", file=sys.stderr)
        return 1

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks, counts = build_chunks(documents, splitter)

    if not chunks:
        print("No non-empty chunks were produced.", file=sys.stderr)
        return 1

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    openai_client = OpenAI()

    upsert_chunks(
        chunks=chunks,
        openai_client=openai_client,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        collection=collection,
        batch_size=BATCH_SIZE,
    )

    print(
        "Finished embedding run. "
        f"documents={len(documents)} chunks={len(chunks)} "
        f"collection='{COLLECTION_NAME}' total_vectors={collection.count()} "
        f"failed_documents={len(failures)}"
    )
    for document_id, count in sorted(counts.items()):
        print(f"  - {document_id}: {count} chunk(s)")

    if failures:
        print("Some files were skipped:", file=sys.stderr)
        for path, error in failures:
            print(f"  - {path.name}: {error}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
