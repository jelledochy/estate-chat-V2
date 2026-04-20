from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import fitz

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.models.documents import ExtractedDocument, ExtractedPage

DEFAULT_INPUT_DIR = BACKEND_DIR / "data" / "pdfs"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "data" / "extracted"
DEFAULT_MANIFEST_PATH = BACKEND_DIR / "data" / "artifacts" / "documents_manifest.json"


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_manifest_types(manifest_path: Path) -> dict[str, str]:
    if not manifest_path.exists():
        return {}

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    result: dict[str, str] = {}

    for item in raw:
        document_id = item.get("document_id")
        document_type = item.get("type") or item.get("document_type")
        if document_id and document_type:
            result[document_id] = document_type

    return result


def infer_document_type(document_id: str, manifest_types: dict[str, str]) -> str:
    if document_id in manifest_types:
        return manifest_types[document_id]

    exact_map = {
        "poa_010": "power_of_attorney",
        "poa_011": "power_of_attorney",
        "will_012": "notarial_will",
        "will_013": "notarial_will",
        "mortgage_014": "mortgage_deed",
        "certificate_016": "family_composition_certificate",
    }
    if document_id in exact_map:
        return exact_map[document_id]

    prefix_map = {
        "purchase": "purchase_contract",
        "sale": "sale_deed",
        "donation_property": "property_donation",
        "donation_money": "monetary_donation",
        "poa": "power_of_attorney",
        "will": "notarial_will",
        "mortgage": "mortgage_deed",
        "certificate": "family_composition_certificate",
    }

    for prefix, document_type in prefix_map.items():
        if document_id.startswith(prefix):
            return document_type

    return "unknown"


def extract_document(pdf_path: Path, *, document_type: str) -> ExtractedDocument:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: list[ExtractedPage] = []

    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            text = normalize_text(page.get_text("text", sort=True))
            pages.append(ExtractedPage(page_number=page_index + 1, text=text))

    full_text = "\n\n".join(page.text for page in pages if page.text).strip()

    return ExtractedDocument(
        document_id=pdf_path.stem,
        filename=pdf_path.name,
        document_type=document_type,
        pages=pages,
        full_text=full_text,
        extraction_method="pymupdf",
        extracted_at=datetime.now(UTC),
    )


def collect_pdf_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {input_path}")
        return [input_path]

    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("*.pdf"))

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def output_is_current(pdf_path: Path, output_path: Path) -> bool:
    if not output_path.exists():
        return False
    return output_path.stat().st_mtime >= pdf_path.stat().st_mtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract embedded PDF text and save structured JSON output."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=str(DEFAULT_INPUT_DIR),
        help="A PDF file or a directory containing PDFs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where extracted .json files will be written.",
    )
    parser.add_argument(
        "--manifest-path",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Optional documents manifest used to populate document_type.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing JSON files even if they are already current.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = Path(args.manifest_path).expanduser().resolve()

    try:
        pdf_paths = collect_pdf_paths(input_path)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not pdf_paths:
        print(f"No PDF files found in {input_path}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_types = load_manifest_types(manifest_path)

    processed = 0
    skipped = 0
    failures: list[tuple[Path, str]] = []

    for pdf_path in pdf_paths:
        output_path = output_dir / f"{pdf_path.stem}.json"

        if output_is_current(pdf_path, output_path) and not args.force:
            print(f"Skipping {pdf_path.name}: {output_path.name} is up to date")
            skipped += 1
            continue

        document_type = infer_document_type(pdf_path.stem, manifest_types)

        try:
            document = extract_document(pdf_path, document_type=document_type)
        except Exception as exc:
            failures.append((pdf_path, str(exc)))
            print(f"Failed {pdf_path.name}: {exc}", file=sys.stderr)
            continue

        output_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        processed += 1
        print(f"Wrote {output_path.name} (method=pymupdf, pages={len(document.pages)})")

    print(
        f"Finished OCR run. processed={processed} skipped={skipped} failed={len(failures)}",
        file=sys.stderr if failures else sys.stdout,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
