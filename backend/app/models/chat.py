from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from backend.app.models import AppBaseModel
from backend.app.models.documents import DocumentType


class ChatFilters(AppBaseModel):
    document_types: list[DocumentType | str] = Field(default_factory=list)
    person_names: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    date_from: dt.date | None = None
    date_to: dt.date | None = None


class ChatRequest(AppBaseModel):
    question: str = Field(..., min_length=1)
    filters: ChatFilters | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class SourceDocument(AppBaseModel):
    document_id: str = Field(..., min_length=1)
    filename: str | None = None
    document_type: DocumentType | str | None = None
    page_numbers: list[int] = Field(default_factory=list)
    excerpt: str | None = None
    citation_label: str | None = None


class GraphContextItem(AppBaseModel):
    rank: int | None = Field(default=None, ge=1)
    triplet: list[str] = Field(default_factory=list, max_length=3)
    text: str = ""
    cypher_query: str | None = None
    cypher_row: dict[str, Any] = Field(default_factory=dict)
    rerank_score: float | None = None


class GraphContext(AppBaseModel):
    neo4j: list[GraphContextItem] = Field(default_factory=list)
    prompt: list[GraphContextItem] = Field(default_factory=list)
    error: str | None = None


class TransactionType(str, Enum):
    purchase = "purchase"
    sale = "sale"
    property_donation = "property_donation"
    monetary_donation = "monetary_donation"
    mortgage = "mortgage"
    inheritance = "inheritance"


class StructuredTransaction(AppBaseModel):
    transaction_type: TransactionType | None = None
    from_parties: list[str] = Field(default_factory=list)
    to_parties: list[str] = Field(default_factory=list)
    property_address: str | None = None
    amount: int | None = Field(default=None, ge=0)
    date: dt.date | None = None
    source_document_id: str | None = None

    @model_validator(mode="after")
    def validate_business_rules(self) -> StructuredTransaction:
        if self.date and self.date > dt.date.today():
            raise ValueError("future dates are not allowed")

        has_meaningful_content = any(
            [
                self.transaction_type,
                self.from_parties,
                self.to_parties,
                self.property_address,
                self.amount is not None,
                self.date,
            ]
        )
        if not has_meaningful_content:
            raise ValueError("transaction is empty")

        return self


class StructuredAnswer(AppBaseModel):
    summary: str | None = None
    transactions: list[StructuredTransaction] = Field(default_factory=list)
    people: list[str] = Field(default_factory=list)
    properties: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_not_empty(self) -> StructuredAnswer:
        if not any([self.summary, self.transactions, self.people, self.properties]):
            raise ValueError("structured answer is empty")
        return self


class ChatResponse(AppBaseModel):
    answer: str = Field(..., min_length=1)
    sources: list[SourceDocument] = Field(default_factory=list)
    graph_context: GraphContext = Field(default_factory=GraphContext)
    structured_answer: StructuredAnswer | None = None
    is_uncertain: bool = False
    uncertainty_message: str | None = None
