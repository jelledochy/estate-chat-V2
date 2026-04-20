from __future__ import annotations

from typing import Any

from pydantic import Field

from backend.app.models import AppBaseModel
from app.services.graph_service import (
    DEFAULT_MAX_NODES,
    DEFAULT_MAX_SEED_PEOPLE,
    DEFAULT_MAX_SEED_PROPERTIES,
    DEFAULT_SUBGRAPH_DEPTH,
    MAX_ALLOWED_DEPTH,
    MAX_ALLOWED_NODES,
)


class GraphNode(AppBaseModel):
    id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1)
    color: str = Field(..., min_length=1)
    size: int = Field(..., ge=1)
    title: str = ""
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(AppBaseModel):
    id: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1)
    title: str = ""
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphMeta(AppBaseModel):
    strategy: str
    matched_people: list[str] = Field(default_factory=list)
    matched_properties: list[str] = Field(default_factory=list)
    matched_document_ids: list[str] = Field(default_factory=list)
    relationship_type: str | None = None


class GraphResponse(AppBaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    meta: GraphMeta | None = None


class QuerySubgraphRequest(AppBaseModel):
    query: str = ""
    depth: int = Field(default=DEFAULT_SUBGRAPH_DEPTH, ge=1, le=MAX_ALLOWED_DEPTH)
    max_nodes: int = Field(default=DEFAULT_MAX_NODES, ge=1, le=MAX_ALLOWED_NODES)
    max_seed_people: int = Field(default=DEFAULT_MAX_SEED_PEOPLE, ge=1, le=MAX_ALLOWED_NODES)
    max_seed_properties: int = Field(
        default=DEFAULT_MAX_SEED_PROPERTIES,
        ge=1,
        le=MAX_ALLOWED_NODES,
    )
