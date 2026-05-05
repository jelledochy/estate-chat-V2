from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=True)

CHROMA_PATH = PROJECT_ROOT / "backend" / "data" / "chroma_db"
COLLECTION_NAME = "estate_documents"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
GRAPH_EMBEDDING_MODEL_NAME = os.getenv(
    "OPENAI_GRAPH_EMBEDDING_MODEL",
    EMBEDDING_MODEL_NAME,
)
DEFAULT_TOP_K = 5
DEFAULT_LLM_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
CROSS_ENCODER_MODEL_NAME = os.getenv(
    "RAG_CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
RETRIEVAL_CANDIDATE_MULTIPLIER = int(os.getenv("RAG_RETRIEVAL_CANDIDATE_MULTIPLIER", "4"))
GRAPH_RESULT_LIMIT = int(os.getenv("RAG_GRAPH_RESULT_LIMIT", "50"))
GRAPH_VECTOR_TOP_K = int(os.getenv("RAG_GRAPH_VECTOR_TOP_K", "5"))
GRAPH_EXPANSION_DEPTH = int(os.getenv("RAG_GRAPH_EXPANSION_DEPTH", "1"))
GRAPH_MODEL = os.getenv("OPENAI_GRAPH_MODEL") or DEFAULT_LLM_MODEL
NEO4J_URL = os.getenv("NEO4J_URL") or os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4jpassword")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


PROMPT_TEMPLATE = """
You are an estate-document assistant.
Answer the QUESTION based only on the CONTEXT from retrieved notarial documents and
supplemental graph context.
If the context is insufficient, clearly say what is missing and do not invent details.
Use graph context only as supplemental context. Do not mention it as a source and
do not cite graph IDs or triplets in the answer.
When you cite facts, cite only the document citation_label exactly in square brackets,
like [certificate_016.pdf p.1].
Do not cite generic labels like [Document 1], [Document source], [Graph 1],
[Graph context], or [graph-1].
If document context and graph context disagree, prefer the document context and say that
the retrieved context conflicts instead of presenting the conflicting fact as confirmed.

QUESTION:
{question}

CONTEXT:
{context}
""".strip()


DOCUMENT_ENTRY_TEMPLATE = """
Document source:
citation_label: {citation_label}
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


GRAPH_ENTRY_TEMPLATE = """
Graph context:
triplet: {triplet}
text:
{text}
""".strip()


GRAPH_TEXT_TO_CYPHER_TEMPLATE = """
Task: Generate a Cypher statement to query a Neo4j graph database.
Instructions:
Use only the relationship types and properties in the schema.
Return only the Cypher statement.
Do not include explanations, apologies, or markdown fences.
Respect relationship direction exactly as shown in the schema.
Use explicit aliases in RETURN clauses, such as person_name, property_name,
document_id, or relation_type.
When matching names from the question, use case-insensitive WHERE clauses
instead of inline property maps. For example, write
`MATCH (person:PERSON) WHERE toLower(person.name) = toLower('thomas janssen')`
rather than `MATCH (person:PERSON {name: 'thomas janssen'})`.
Apply this to PERSON, ORGANIZATION, PROPERTY, DOCUMENT, and any other named
entity with a name-like property.
Use a broad LIMIT when returning rows. Do not use LIMIT 1 for relationship
lookups because people can have multiple parents, spouses, children, donors, or
beneficiaries.
For questions about how multiple parties relate through the same act or deed,
join them through the shared DOCUMENT or PROPERTY node when appropriate.
Prefer the most specific relationship names present in the schema over
semantically similar alternatives.
When a question asks about a person receiving, donating, buying, selling, or
parenting, infer the correct relationship from the schema instead of inventing
a new pattern.
If the question asks who, what, which, or where, return the fields needed to
answer that question directly.

Schema:
{schema}

Question:
{question}
""".strip()
