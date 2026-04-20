from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Any, Literal

from pydantic import Field

from backend.app.models import AppBaseModel


class DocumentType(str, Enum):
    purchase_contract = "purchase_contract"
    sale_deed = "sale_deed"
    property_donation = "property_donation"
    monetary_donation = "monetary_donation"
    power_of_attorney = "power_of_attorney"
    notarial_will = "notarial_will"
    mortgage_deed = "mortgage_deed"
    family_composition_certificate = "family_composition_certificate"
    unknown = "unknown"


class ExtractedPage(AppBaseModel):
    page_number: int = Field(..., ge=1)
    text: str = ""


class DocumentMetadata(AppBaseModel):
    document_id: str = Field(..., min_length=1)
    filename: str = Field(..., min_length=1)
    document_type: DocumentType | str = DocumentType.unknown
    date: dt.date | None = None
    parties: dict[str, list[str]] = Field(default_factory=dict)
    references: list[str] = Field(default_factory=list)
    notes: str | None = None


class ExtractedDocument(DocumentMetadata):
    pages: list[ExtractedPage] = Field(default_factory=list)
    full_text: str = ""
    extraction_method: Literal["pymupdf", "tesseract", "pymupdf+tesseract"] = "pymupdf"
    extracted_at: dt.datetime


class ChunkMetadata(AppBaseModel):
    document_id: str = Field(..., min_length=1)
    document_type: DocumentType | str = DocumentType.unknown
    chunk_index: int = Field(..., ge=0)
    page_numbers: list[int] = Field(default_factory=list)
    person_names_mentioned: list[str] = Field(default_factory=list)


class DocumentChunk(AppBaseModel):
    chunk_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    embedding_id: str | None = None
    metadata: ChunkMetadata


class DocumentSummary(AppBaseModel):
    document_id: str = Field(..., min_length=1)
    filename: str = Field(..., min_length=1)
    document_type: DocumentType | str = DocumentType.unknown
    date: dt.date | None = None
    parties: dict[str, list[str]] = Field(default_factory=dict)
    property_details: dict[str, Any] | None = Field(default=None, alias="property")
    amount: float | None = Field(default=None, ge=0)
    references: list[str] = Field(default_factory=list)
    notes: str | None = None


class DocumentDetail(DocumentSummary):
    full_text: str = ""
