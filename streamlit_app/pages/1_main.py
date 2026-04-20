from __future__ import annotations

import datetime as dt
import html
import os
import re
from collections import Counter
from typing import Any

import httpx
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

DEFAULT_API_BASE_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

DATE_ISO_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
MONTH_PATTERN = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)

TIMELINE_QUERY_HINTS = (
    "timeline",
    "chronology",
    "history",
    "between",
    "before",
    "after",
    "first",
    "then",
    "later",
    "earlier",
    "event",
    "events",
)
RELATIONSHIP_QUERY_HINTS = (
    "who is related",
    "related to whom",
    "relationship",
    "related",
    "family tree",
    "family relationship",
    "spouse",
    "married",
    "parent",
    "child",
    "father",
    "mother",
    "son",
    "daughter",
)

DOCUMENT_TYPE_STYLES: dict[str, dict[str, str]] = {
    "purchase_contract": {
        "icon": "🏠",
        "label": "Purchase",
        "bg": "#E7F4FF",
        "border": "#B9DCFF",
        "text": "#0D4A7A",
    },
    "sale_deed": {
        "icon": "🏠",
        "label": "Sale",
        "bg": "#FFECD9",
        "border": "#FFCFA2",
        "text": "#8A4600",
    },
    "property_donation": {
        "icon": "🏠",
        "label": "Property Donation",
        "bg": "#E9FAEF",
        "border": "#BAEAC9",
        "text": "#166534",
    },
    "monetary_donation": {
        "icon": "💰",
        "label": "Money Donation",
        "bg": "#FFF8DE",
        "border": "#F4E2A2",
        "text": "#7A5A00",
    },
    "notarial_will": {
        "icon": "✍️",
        "label": "Will",
        "bg": "#FBEAFF",
        "border": "#E7C2F5",
        "text": "#6A2A8A",
    },
    "power_of_attorney": {
        "icon": "🖋️",
        "label": "Power of Attorney",
        "bg": "#EAF2FF",
        "border": "#C4D6FF",
        "text": "#1E429F",
    },
    "mortgage_deed": {
        "icon": "🏦",
        "label": "Mortgage",
        "bg": "#FFEFF0",
        "border": "#FFCDD2",
        "text": "#9F1239",
    },
    "family_composition_certificate": {
        "icon": "👪",
        "label": "Family Certificate",
        "bg": "#ECFEFF",
        "border": "#B5EEF5",
        "text": "#0B5663",
    },
    "unknown": {
        "icon": "📄",
        "label": "Document",
        "bg": "#F3F4F6",
        "border": "#D1D5DB",
        "text": "#374151",
    },
}

TIMELINE_EVENT_LABELS = {
    "purchase_contract": "Property purchased",
    "sale_deed": "Property sold",
    "property_donation": "Property donated",
    "monetary_donation": "Monetary donation",
    "notarial_will": "Will signed",
    "power_of_attorney": "Power of attorney granted",
    "mortgage_deed": "Mortgage deed signed",
    "family_composition_certificate": "Family composition updated",
    "unknown": "Document recorded",
}

GRAPH_NODE_ICON_BY_TYPE = {
    "Person": "👤",
    "Property": "🏠",
    "Document": "📄",
    "Node": "🔹",
}

GRAPH_NODE_COLOR_BY_TYPE = {
    "Person": "#4A90E2",
    "Property": "#2EAD67",
    "Document": "#F39C12",
    "Node": "#7F8C8D",
}

st.set_page_config(page_title="Estate Planner Chat", page_icon=":scroll:", layout="wide")


# ============================================================================
# API + Data Helpers
# ============================================================================


def post_chat_request(api_base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{api_base_url.rstrip('/')}/api/chat"
    with httpx.Client(timeout=45.0) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


def fetch_query_subgraph(api_base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{api_base_url.rstrip('/')}/api/graph/subgraph"
    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


@st.cache_data(show_spinner=False, ttl=180)
def fetch_documents_index(api_base_url: str) -> list[dict[str, Any]]:
    url = f"{api_base_url.rstrip('/')}/api/documents"
    with httpx.Client(timeout=30.0) as client:
        response = client.get(url)
        response.raise_for_status()
        payload = response.json()

    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def normalize_sources(raw_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for raw_source in raw_sources:
        if not isinstance(raw_source, dict):
            continue

        document_id = str(raw_source.get("document_id") or "").strip()
        if not document_id:
            continue

        page_numbers: list[int] = []
        for raw_page in raw_source.get("page_numbers") or []:
            try:
                page_number = int(raw_page)
            except (TypeError, ValueError):
                continue
            if page_number > 0 and page_number not in page_numbers:
                page_numbers.append(page_number)

        sources.append(
            {
                "document_id": document_id,
                "filename": str(raw_source.get("filename") or "").strip() or None,
                "document_type": str(raw_source.get("document_type") or "").strip() or None,
                "page_numbers": sorted(page_numbers),
                "excerpt": str(raw_source.get("excerpt") or "").strip() or None,
            }
        )

    return sources


def to_agraph_nodes(raw_nodes: list[dict[str, Any]]) -> list[Node]:
    nodes: list[Node] = []
    for raw in raw_nodes:
        node_id = str(raw.get("id") or "").strip()
        if not node_id:
            continue

        node_type = str(raw.get("type") or "Node").strip() or "Node"

        try:
            size = int(raw.get("size", 20))
        except (TypeError, ValueError):
            size = 20

        icon = GRAPH_NODE_ICON_BY_TYPE.get(node_type, GRAPH_NODE_ICON_BY_TYPE["Node"])
        raw_label = str(raw.get("label") or node_id).strip() or node_id

        node_color = str(
            raw.get("color")
            or GRAPH_NODE_COLOR_BY_TYPE.get(node_type)
            or GRAPH_NODE_COLOR_BY_TYPE["Node"]
        )

        nodes.append(
            Node(
                id=node_id,
                label=icon,
                title=str(raw.get("title") or raw_label),
                size=max(size, 20),
                color=node_color,
                shape="circle",
                borderWidth=2,
                font={"size": 18, "strokeWidth": 0},
            )
        )
    return nodes


def to_agraph_edges(raw_edges: list[dict[str, Any]]) -> list[Edge]:
    edges: list[Edge] = []
    for raw in raw_edges:
        source = str(raw.get("source") or "").strip()
        target = str(raw.get("target") or "").strip()
        if not source or not target:
            continue

        edges.append(
            Edge(
                source=source,
                target=target,
                label=str(raw.get("label") or raw.get("type") or "RELATED_TO"),
                title=str(raw.get("title") or ""),
            )
        )
    return edges


# ============================================================================
# Formatting + Enrichment Helpers
# ============================================================================


def unique_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        marker = normalized.casefold()
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(normalized)
    return unique


def parse_date(value: Any) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


def format_date_label(value: Any) -> str:
    parsed = parse_date(value)
    return parsed.isoformat() if parsed else "-"


def format_amount_label(value: Any) -> str | None:
    if value is None:
        return None
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    if amount < 0:
        return None
    return f"{amount:,.0f}"


def shorten_text(value: str, max_length: int) -> str:
    clean_value = " ".join(value.split())
    if len(clean_value) <= max_length:
        return clean_value
    return clean_value[: max_length - 1].rstrip() + "…"


def normalize_parties(raw_parties: Any) -> dict[str, list[str]]:
    if not isinstance(raw_parties, dict):
        return {}

    parties: dict[str, list[str]] = {}
    for raw_role, raw_names in raw_parties.items():
        role = str(raw_role or "").strip()
        if not role:
            continue

        if isinstance(raw_names, list):
            names = [str(name).strip() for name in raw_names if str(name).strip()]
        else:
            one_name = str(raw_names or "").strip()
            names = [one_name] if one_name else []

        parties[role] = unique_values(names)

    return parties


def extract_property_address(document: dict[str, Any]) -> str | None:
    raw_property = document.get("property")
    if not isinstance(raw_property, dict):
        return None
    address = str(raw_property.get("address") or "").strip()
    return address or None


def first_names_for_role_keywords(
    parties: dict[str, list[str]],
    role_keywords: tuple[str, ...],
) -> list[str]:
    names: list[str] = []
    for role, role_names in parties.items():
        role_text = role.casefold()
        if any(keyword in role_text for keyword in role_keywords):
            names.extend(name for name in role_names if name.casefold() != "purchasing together")
    return unique_values(names)


def summarize_event_participants(raw_parties: Any) -> str | None:
    parties = normalize_parties(raw_parties)
    if not parties:
        return None

    from_names = first_names_for_role_keywords(
        parties,
        ("seller", "donor", "grantor", "testator", "borrower", "owner"),
    )
    to_names = first_names_for_role_keywords(
        parties,
        ("buyer", "donee", "attorney", "heir", "beneficiar", "lender"),
    )

    if from_names and to_names:
        left = ", ".join(from_names[:2])
        right = ", ".join(to_names[:2])
        return f"{left} → {right}"

    if from_names:
        return ", ".join(from_names[:3])
    if to_names:
        return ", ".join(to_names[:3])

    first_role = next(iter(parties))
    first_names = ", ".join(parties[first_role][:3])
    return f"{first_role}: {first_names}" if first_names else None


def canonical_document_type(raw_value: Any) -> str:
    value = str(raw_value or "").strip().casefold()
    if not value:
        return "unknown"

    if "power_of_attorney" in value or "volmacht" in value or "procuration" in value:
        return "power_of_attorney"
    if "property_donation" in value:
        return "property_donation"
    if "monetary_donation" in value or "donation_money" in value:
        return "monetary_donation"
    if "purchase" in value:
        return "purchase_contract"
    if "sale" in value:
        return "sale_deed"
    if "notarial_will" in value or value.startswith("will"):
        return "notarial_will"
    if "mortgage" in value:
        return "mortgage_deed"
    if "family_composition_certificate" in value:
        return "family_composition_certificate"
    return value


def format_document_type_label(raw_value: Any) -> str:
    canonical = canonical_document_type(raw_value)
    styled = DOCUMENT_TYPE_STYLES.get(canonical)
    if styled:
        return styled["label"]

    text = str(raw_value or "").strip() or canonical
    return text.replace("_", " ").strip().title() or "Document"


def render_document_type_badge_html(raw_value: Any) -> str:
    canonical = canonical_document_type(raw_value)
    style = DOCUMENT_TYPE_STYLES.get(canonical, DOCUMENT_TYPE_STYLES["unknown"])

    icon = html.escape(style["icon"])
    label = html.escape(format_document_type_label(raw_value))
    return (
        "<span class='doc-badge' "
        f"style='background:{style['bg']}; border-color:{style['border']}; color:{style['text']};'>"
        f"{icon} {label}</span>"
    )


def build_documents_by_id(documents: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    document_index: dict[str, dict[str, Any]] = {}
    for item in documents:
        document_id = str(item.get("document_id") or "").strip()
        if not document_id:
            continue
        document_index[document_id] = item
    return document_index


def timeline_query_detected(query: str) -> bool:
    text = query.casefold().strip()
    if not text:
        return False

    if text.count("when") >= 2:
        return True

    iso_dates = set(DATE_ISO_PATTERN.findall(text))
    years = set(YEAR_PATTERN.findall(text))
    month_mentions = MONTH_PATTERN.findall(text)
    if len(iso_dates) + len(years) >= 2:
        return True
    if len(month_mentions) >= 2:
        return True

    event_hint_count = sum(1 for hint in TIMELINE_QUERY_HINTS if hint in text)
    return event_hint_count >= 2


def build_timeline_events(
    *,
    sources: list[dict[str, Any]],
    documents_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seen_document_ids: set[str] = set()

    for source in sources:
        document_id = str(source.get("document_id") or "").strip()
        if not document_id or document_id in seen_document_ids:
            continue
        seen_document_ids.add(document_id)

        document = documents_by_id.get(document_id) or {}
        date_value = document.get("date")
        parsed_date = parse_date(date_value)
        if parsed_date is None:
            continue

        raw_type = document.get("document_type") or source.get("document_type")
        canonical_type = canonical_document_type(raw_type)
        event_title = TIMELINE_EVENT_LABELS.get(canonical_type, TIMELINE_EVENT_LABELS["unknown"])

        detail_parts: list[str] = []
        participants = summarize_event_participants(document.get("parties"))
        if participants:
            detail_parts.append(participants)

        property_address = extract_property_address(document)
        if property_address:
            detail_parts.append(shorten_text(property_address, max_length=66))

        amount = format_amount_label(document.get("amount"))
        if amount:
            detail_parts.append(f"amount {amount}")

        events.append(
            {
                "date_sort": parsed_date.isoformat(),
                "date_label": parsed_date.isoformat(),
                "title": event_title,
                "details": " | ".join(detail_parts),
                "document_id": document_id,
                "document_type": raw_type,
            }
        )

    events.sort(key=lambda item: item["date_sort"])
    return events


def should_show_timeline(query: str, timeline_events: list[dict[str, Any]]) -> bool:
    if len(timeline_events) < 2:
        return False
    return timeline_query_detected(query)


def prettify_relationship_label(raw_label: Any) -> str:
    label = str(raw_label or "RELATED_TO").strip()
    if not label:
        return "Related to"
    return label.replace("_", " ").casefold()


def is_relationship_query(query: str) -> bool:
    normalized = query.casefold()
    if "who is" in normalized and "related" in normalized:
        return True
    return any(hint in normalized for hint in RELATIONSHIP_QUERY_HINTS)


def build_relationship_cards(
    graph_payload: dict[str, Any],
    *,
    max_cards: int = 8,
    max_relations_per_card: int = 6,
) -> list[dict[str, Any]]:
    raw_nodes = graph_payload.get("nodes") or []
    raw_edges = graph_payload.get("edges") or []

    nodes_by_id: dict[str, dict[str, str]] = {}
    for raw_node in raw_nodes:
        node_id = str(raw_node.get("id") or "").strip()
        if not node_id:
            continue
        nodes_by_id[node_id] = {
            "label": str(raw_node.get("label") or node_id).strip() or node_id,
            "type": str(raw_node.get("type") or "").strip() or "Node",
        }

    cards: dict[str, dict[str, Any]] = {}
    for node_id, node_data in nodes_by_id.items():
        if node_data["type"] == "Person":
            cards[node_id] = {"name": node_data["label"], "relationships": []}

    if not cards:
        return []

    for raw_edge in raw_edges:
        source = str(raw_edge.get("source") or "").strip()
        target = str(raw_edge.get("target") or "").strip()
        if source not in nodes_by_id or target not in nodes_by_id:
            continue

        relationship = prettify_relationship_label(raw_edge.get("label") or raw_edge.get("type"))
        target_label = nodes_by_id[target]["label"]
        source_label = nodes_by_id[source]["label"]

        if source in cards:
            cards[source]["relationships"].append(f"{relationship} → {target_label}")
        if target in cards:
            cards[target]["relationships"].append(f"{relationship} ← {source_label}")

    result_cards: list[dict[str, Any]] = []
    for card in cards.values():
        deduped_relationships = unique_values(card["relationships"])
        if not deduped_relationships:
            continue
        result_cards.append(
            {
                "name": card["name"],
                "relationships": deduped_relationships[:max_relations_per_card],
            }
        )

    result_cards.sort(key=lambda item: (-len(item["relationships"]), str(item["name"]).casefold()))
    return result_cards[:max_cards]


def collect_people_frequency(documents: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for document in documents:
        parties = normalize_parties(document.get("parties"))
        for names in parties.values():
            for name in names:
                clean_name = name.strip()
                if not clean_name or clean_name.casefold() == "purchasing together":
                    continue
                counter[clean_name] += 1
    return counter


def first_party_name(
    document: dict[str, Any],
    role_keywords: tuple[str, ...],
) -> str | None:
    parties = normalize_parties(document.get("parties"))
    names = first_names_for_role_keywords(parties, role_keywords)
    return names[0] if names else None


def compact_address(address: str) -> str:
    first_part = address.split(",")[0].strip()
    if first_part:
        return first_part
    return shorten_text(address, max_length=48)


def unique_question_list(questions: list[str], *, limit: int) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for question in questions:
        clean = " ".join(question.split()).strip()
        if not clean:
            continue
        marker = clean.casefold()
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(clean)
        if len(deduped) >= limit:
            break
    return deduped


def detect_query_topics(query: str | None) -> set[str]:
    normalized = str(query or "").casefold()
    if not normalized:
        return set()

    topics: set[str] = set()
    if any(token in normalized for token in ("own", "owner", "ownership")):
        topics.add("ownership")
    if any(
        token in normalized
        for token in ("timeline", "chronolog", "when", "before", "after", "event")
    ):
        topics.add("timeline")
    if any(
        token in normalized
        for token in ("related", "relationship", "family", "parent", "child", "spouse", "married")
    ):
        topics.add("relationship")
    if any(token in normalized for token in ("donat", "gift")):
        topics.add("donation")
    if any(token in normalized for token in ("will", "beneficiar", "inherit")):
        topics.add("will")
    if any(token in normalized for token in ("power of attorney", "poa", "attorney")):
        topics.add("poa")
    if any(token in normalized for token in ("mortgage", "loan", "borrower", "lender")):
        topics.add("mortgage")

    return topics


def first_known_party_name(document: dict[str, Any]) -> str | None:
    parties = normalize_parties(document.get("parties"))
    for names in parties.values():
        for name in names:
            candidate = str(name).strip()
            if candidate and candidate.casefold() != "purchasing together":
                return candidate
    return None


def get_recent_chat_context(
    chat_messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    last_user_query: str | None = None
    last_assistant_sources: list[dict[str, Any]] = []

    for message in reversed(chat_messages):
        role = str(message.get("role") or "").strip()
        if role == "assistant" and not last_assistant_sources:
            raw_sources = message.get("sources")
            if isinstance(raw_sources, list):
                last_assistant_sources = [item for item in raw_sources if isinstance(item, dict)]
        if role == "user" and last_user_query is None:
            content = str(message.get("content") or "").strip()
            if content:
                last_user_query = content

        if last_user_query is not None and last_assistant_sources:
            break

    return last_user_query, last_assistant_sources


def build_contextual_followup_questions(
    *,
    last_user_query: str | None,
    last_assistant_sources: list[dict[str, Any]] | None,
    documents_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    if not last_user_query:
        return []

    topics = detect_query_topics(last_user_query)
    sources = last_assistant_sources or []

    context_document: dict[str, Any] | None = None
    context_document_id: str | None = None
    for source in sources:
        source_document_id = str(source.get("document_id") or "").strip()
        if not source_document_id:
            continue
        document = documents_by_id.get(source_document_id)
        if document:
            context_document = document
            context_document_id = source_document_id
            break

    context_address = extract_property_address(context_document or {})
    context_person = first_known_party_name(context_document or {})

    followups: list[str] = []
    if "ownership" in topics:
        if context_address:
            address = compact_address(context_address)
            followups.append(f"Which documents prove current ownership of {address}?")
            followups.append(f"Can you list ownership changes for {address} in order?")
        else:
            followups.append("Which documents are strongest proof of ownership in this answer?")

    if "timeline" in topics:
        if context_person:
            followups.append(
                f"What happened immediately before and after events involving {context_person}?"
            )
        followups.append("Which event in the timeline has the strongest supporting evidence?")

    if "relationship" in topics:
        if context_person:
            followups.append(f"Which documents directly support {context_person}'s relationships?")
        followups.append("Can you show only direct family relationships from the evidence?")

    if "donation" in topics:
        followups.append("Can you break down donations by year and recipient?")

    if "will" in topics:
        followups.append("How do the will provisions compare with prior donations or transfers?")

    if "poa" in topics:
        followups.append("Are there limits or conditions in the power-of-attorney documents?")

    if "mortgage" in topics:
        if context_address:
            followups.append(
                f"What obligations are tied to the mortgage on {compact_address(context_address)}?"
            )
        else:
            followups.append("Which mortgage terms are most relevant to this answer?")

    if context_document_id:
        followups.append(
            f"Can you summarize what {context_document_id} contributes to this answer?"
        )

    followups.append("What details are still uncertain or missing in the current evidence?")
    return followups


def build_suggested_questions(
    documents: list[dict[str, Any]],
    *,
    last_user_query: str | None = None,
    last_assistant_sources: list[dict[str, Any]] | None = None,
    documents_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    fallback_questions = [
        "Who owns property X?",
        "When did key estate events happen?",
        "Who is related to whom in this family?",
        "What donations were made and to whom?",
    ]

    document_lookup = documents_by_id or build_documents_by_id(documents)
    suggestions: list[str] = build_contextual_followup_questions(
        last_user_query=last_user_query,
        last_assistant_sources=last_assistant_sources,
        documents_by_id=document_lookup,
    )

    if not documents:
        return (
            unique_question_list([*suggestions, *fallback_questions], limit=6) or fallback_questions
        )

    dated_documents = [(parse_date(document.get("date")), document) for document in documents]
    dated_documents = [
        (date_value, document) for date_value, document in dated_documents if date_value
    ]
    dated_documents.sort(key=lambda item: item[0])

    if len(dated_documents) >= 2:
        start_year = dated_documents[0][0].year
        end_year = dated_documents[-1][0].year
        suggestions.append(
            f"Can you show a timeline of major events between {start_year} and {end_year}?"
        )

    property_docs = [document for document in documents if extract_property_address(document)]
    if property_docs:
        address = extract_property_address(property_docs[0]) or ""
        suggestions.append(f"Who owns the property at {compact_address(address)}?")
        suggestions.append(f"When did events happen for {compact_address(address)}?")

    will_doc = next(
        (
            document
            for document in documents
            if canonical_document_type(document.get("document_type")) == "notarial_will"
        ),
        None,
    )
    if will_doc:
        testator = first_party_name(will_doc, ("testator",))
        if testator:
            suggestions.append(f"Who are the beneficiaries in {testator}'s will?")
        else:
            suggestions.append("Who are the beneficiaries listed in the wills?")

    donation_doc = next(
        (
            document
            for document in documents
            if canonical_document_type(document.get("document_type"))
            in {"monetary_donation", "property_donation"}
        ),
        None,
    )
    if donation_doc:
        donee = first_party_name(donation_doc, ("donee", "beneficiar"))
        if donee:
            suggestions.append(f"How much has {donee} received in donations?")
        else:
            suggestions.append("Which donations were made, and to whom?")

    poa_doc = next(
        (
            document
            for document in documents
            if canonical_document_type(document.get("document_type")) == "power_of_attorney"
        ),
        None,
    )
    if poa_doc:
        grantor = first_party_name(poa_doc, ("grantor",))
        if grantor:
            suggestions.append(f"Who has power of attorney for {grantor}?")
        else:
            suggestions.append("Who holds power of attorney, and over whom?")

    people_frequency = collect_people_frequency(documents)
    if people_frequency:
        person, _count = people_frequency.most_common(1)[0]
        suggestions.append(f"What are the most important transactions involving {person}?")

    suggestions.append("Who is related to whom in this family?")
    deduped = unique_question_list(suggestions, limit=6)
    return deduped or fallback_questions


# ============================================================================
# UI Rendering Helpers
# ============================================================================


def apply_app_styles() -> None:
    palette = {
        "bg_base": "#F8FAFC",
        "bg_glow": "#E7F1FF",
        "text": "#0F172A",
        "panel": "rgba(255, 255, 255, 0.86)",
        "border": "#D6E2F0",
        "muted": "#475569",
        "line": "#8DA2B8",
        "dot": "#2563EB",
    }

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: radial-gradient(circle at top right, {palette["bg_glow"]} 0%, {palette["bg_base"]} 48%);
            color: {palette["text"]};
        }}
        div[data-testid="stChatMessage"] {{
            background: {palette["panel"]};
            border: 1px solid {palette["border"]};
            border-radius: 14px;
            padding: 0.55rem 0.9rem;
            margin-bottom: 0.65rem;
        }}
        .doc-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border: 1px solid;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            line-height: 1.1;
            padding: 0.18rem 0.58rem;
            white-space: nowrap;
        }}
        .source-row {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.45rem;
            margin-bottom: 0.25rem;
        }}
        .source-meta {{
            color: {palette["muted"]};
            font-size: 0.88rem;
        }}
        .timeline-wrap {{
            border-left: 3px solid {palette["line"]};
            margin: 0.2rem 0 0.3rem 0.45rem;
            padding-left: 0.8rem;
        }}
        .timeline-item {{
            position: relative;
            margin: 0 0 0.85rem 0;
        }}
        .timeline-item::before {{
            content: "";
            width: 0.62rem;
            height: 0.62rem;
            border-radius: 50%;
            background: {palette["dot"]};
            position: absolute;
            left: -1.08rem;
            top: 0.35rem;
        }}
        .timeline-date {{
            color: {palette["muted"]};
            font-size: 0.8rem;
            font-weight: 700;
        }}
        .timeline-title {{
            font-size: 0.95rem;
            font-weight: 700;
            margin-top: 0.1rem;
        }}
        .timeline-details {{
            color: {palette["muted"]};
            font-size: 0.88rem;
            margin-top: 0.1rem;
        }}
        .timeline-doc {{
            font-size: 0.78rem;
            color: {palette["muted"]};
            margin-top: 0.15rem;
        }}
        .relationship-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.65rem;
            margin-top: 0.2rem;
        }}
        .relationship-card {{
            border: 1px solid {palette["border"]};
            border-radius: 12px;
            padding: 0.58rem 0.72rem;
            background: {palette["panel"]};
        }}
        .relationship-name {{
            font-weight: 700;
            margin-bottom: 0.35rem;
        }}
        .relationship-list {{
            margin: 0;
            padding-left: 1rem;
            color: {palette["muted"]};
            font-size: 0.88rem;
        }}
        .suggestion-note {{
            color: {palette["muted"]};
            font-size: 0.88rem;
            margin-bottom: 0.35rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_chat_error(exc: Exception) -> None:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        detail = ""
        try:
            response_json = exc.response.json()
            detail = str(response_json.get("detail") or "")
        except Exception:
            detail = exc.response.text

        st.error(f"Backend returned HTTP {status_code}. {detail}".strip())
        return

    if isinstance(exc, httpx.RequestError):
        st.error(
            "Could not reach backend API. "
            "Check that FastAPI is running and the API URL is correct."
        )
        return

    st.error(f"Unexpected error while sending chat request: {exc}")


def show_graph_error(exc: Exception) -> None:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        detail = ""
        try:
            response_json = exc.response.json()
            detail = str(response_json.get("detail") or "")
        except Exception:
            detail = exc.response.text

        st.error(f"Backend returned HTTP {status_code}. {detail}".strip())
        return

    if isinstance(exc, httpx.RequestError):
        st.error(
            "Could not reach backend API. " "Check that FastAPI is running and API URL is correct."
        )
        return

    st.error(f"Unexpected error while loading graph: {exc}")


def render_sources(
    sources: list[dict[str, Any]],
    *,
    documents_by_id: dict[str, dict[str, Any]],
) -> None:
    if not sources:
        return

    with st.expander("📄 Sources"):
        for source in sources:
            document_id = str(source.get("document_id") or "").strip()
            if not document_id:
                continue

            document = documents_by_id.get(document_id) or {}
            raw_type = source.get("document_type") or document.get("document_type")
            badge_html = render_document_type_badge_html(raw_type)

            page_numbers = source.get("page_numbers") or []
            page_label = ", ".join(str(page) for page in page_numbers) if page_numbers else "-"
            date_label = format_date_label(document.get("date"))

            meta_parts = [f"<code>{html.escape(document_id)}</code>"]
            if date_label != "-":
                meta_parts.append(f"date: {html.escape(date_label)}")
            if page_label != "-":
                meta_parts.append(f"pages: {html.escape(page_label)}")

            st.markdown(
                (
                    "<div class='source-row'>"
                    f"{badge_html}"
                    f"<span class='source-meta'>{' | '.join(meta_parts)}</span>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

            excerpt = str(source.get("excerpt") or "").strip()
            if excerpt:
                st.caption(excerpt)


def render_timeline(timeline_events: list[dict[str, Any]]) -> None:
    if len(timeline_events) < 2:
        return

    timeline_rows: list[str] = ["<div class='timeline-wrap'>"]
    for event in timeline_events:
        date_label = html.escape(str(event.get("date_label") or "-"))
        title = html.escape(str(event.get("title") or "Event"))
        details_text = str(event.get("details") or "").strip()
        details_html = (
            f"<div class='timeline-details'>{html.escape(details_text)}</div>"
            if details_text
            else ""
        )

        document_id = str(event.get("document_id") or "").strip()
        doc_html = (
            f"<div class='timeline-doc'>source: {html.escape(document_id)}</div>"
            if document_id
            else ""
        )

        timeline_rows.append(
            (
                "<div class='timeline-item'>"
                f"<div class='timeline-date'>{date_label}</div>"
                f"<div class='timeline-title'>{title}</div>"
                f"{details_html}"
                f"{doc_html}"
                "</div>"
            )
        )

    timeline_rows.append("</div>")

    st.markdown("**🕒 Timeline**")
    st.markdown("".join(timeline_rows), unsafe_allow_html=True)


def render_relationship_cards(relationship_cards: list[dict[str, Any]]) -> None:
    if not relationship_cards:
        return

    card_rows: list[str] = ["<div class='relationship-grid'>"]
    for card in relationship_cards:
        name = html.escape(str(card.get("name") or "Unknown"))
        relationships = card.get("relationships") or []

        relationship_items = "".join(f"<li>{html.escape(str(item))}</li>" for item in relationships)

        card_rows.append(
            (
                "<div class='relationship-card'>"
                f"<div class='relationship-name'>👤 {name}</div>"
                f"<ul class='relationship-list'>{relationship_items}</ul>"
                "</div>"
            )
        )

    card_rows.append("</div>")

    st.markdown("**👥 Relationship Diagram**")
    st.markdown("".join(card_rows), unsafe_allow_html=True)


def render_suggested_questions(questions: list[str]) -> str | None:
    if not questions:
        return None

    st.markdown("**Suggested questions**")
    st.markdown(
        "<div class='suggestion-note'>Click one to ask it instantly.</div>",
        unsafe_allow_html=True,
    )

    selected_question: str | None = None
    column_count = min(3, max(1, len(questions)))
    columns = st.columns(column_count)

    for index, question in enumerate(questions):
        with columns[index % column_count]:
            if st.button(question, key=f"suggested-question-{index}", use_container_width=True):
                selected_question = question

    return selected_question


def render_knowledge_graph(
    api_base_url: str,
    query: str,
    depth: int = 2,
    max_nodes: int = 80,
    max_seed_people: int = 3,
    max_seed_properties: int = 3,
) -> None:
    payload = {
        "query": query,
        "depth": depth,
        "max_nodes": max_nodes,
        "max_seed_people": max_seed_people,
        "max_seed_properties": max_seed_properties,
    }

    with st.spinner("Loading knowledge graph..."):
        try:
            graph_payload = fetch_query_subgraph(api_base_url, payload)
        except Exception as exc:
            show_graph_error(exc)
            return

    raw_nodes = graph_payload.get("nodes") or []
    raw_edges = graph_payload.get("edges") or []

    agraph_nodes = to_agraph_nodes(raw_nodes)
    agraph_edges = to_agraph_edges(raw_edges)

    st.write(f"**Nodes:** {len(agraph_nodes)} | **Edges:** {len(agraph_edges)}")

    if not agraph_nodes:
        st.warning("No nodes returned for this query.")
    else:
        config = Config(
            width=900,
            height=600,
            directed=True,
            physics=True,
            hierarchical=False,
        )
        selected_node_id = agraph(nodes=agraph_nodes, edges=agraph_edges, config=config)

        if selected_node_id:
            node_by_id = {
                str(node.get("id")): node for node in raw_nodes if node.get("id") is not None
            }
            selected_node = node_by_id.get(str(selected_node_id))
            if selected_node:
                st.subheader("Selected node details")
                st.json(selected_node)


# ============================================================================
# Chat Session Helpers
# ============================================================================


def build_chat_title_from_messages(messages: list[dict[str, Any]]) -> str | None:
    for message in messages:
        if str(message.get("role") or "").strip() != "user":
            continue
        content = str(message.get("content") or "").strip()
        if content:
            return shorten_text(content, max_length=32)
    return None


def create_chat_session(chat_index: int) -> dict[str, Any]:
    return {
        "id": f"chat-{chat_index}",
        "title": f"Chat {chat_index}",
        "messages": [],
        "last_query": None,
    }


def initialize_chat_sessions_state() -> None:
    if "chat_sessions" not in st.session_state:
        initial_chat = create_chat_session(1)

        legacy_messages = st.session_state.get("chat_messages")
        if isinstance(legacy_messages, list):
            initial_chat["messages"] = legacy_messages

        legacy_last_query = st.session_state.get("last_query")
        if isinstance(legacy_last_query, str) and legacy_last_query.strip():
            initial_chat["last_query"] = legacy_last_query.strip()

        auto_title = build_chat_title_from_messages(initial_chat["messages"])
        if auto_title:
            initial_chat["title"] = auto_title

        st.session_state["chat_sessions"] = [initial_chat]

    if "chat_counter" not in st.session_state:
        st.session_state["chat_counter"] = max(1, len(st.session_state["chat_sessions"]))

    if not st.session_state["chat_sessions"]:
        fallback_chat = create_chat_session(1)
        st.session_state["chat_sessions"] = [fallback_chat]
        st.session_state["chat_counter"] = 1

    if "active_chat_id" not in st.session_state:
        st.session_state["active_chat_id"] = str(st.session_state["chat_sessions"][0]["id"])

    if "pending_prompt" not in st.session_state:
        st.session_state["pending_prompt"] = None


def get_active_chat_session() -> dict[str, Any]:
    chat_sessions = st.session_state.get("chat_sessions") or []
    active_chat_id = str(st.session_state.get("active_chat_id") or "")

    for chat in chat_sessions:
        if str(chat.get("id") or "") == active_chat_id:
            return chat

    fallback_chat = chat_sessions[0]
    st.session_state["active_chat_id"] = str(fallback_chat["id"])
    return fallback_chat


def create_new_chat_session() -> None:
    st.session_state["chat_counter"] += 1
    new_chat = create_chat_session(st.session_state["chat_counter"])
    st.session_state["chat_sessions"].insert(0, new_chat)
    st.session_state["active_chat_id"] = str(new_chat["id"])
    st.session_state["pending_prompt"] = None


def maybe_update_chat_title(chat: dict[str, Any], prompt: str) -> None:
    title = str(chat.get("title") or "").strip()
    if not title.startswith("Chat "):
        return
    next_title = shorten_text(prompt, max_length=32)
    if next_title:
        chat["title"] = next_title


# ============================================================================
# Initialize Session State
# ============================================================================


initialize_chat_sessions_state()

if "api_base_url" not in st.session_state:
    st.session_state["api_base_url"] = DEFAULT_API_BASE_URL


# ============================================================================
# Main UI
# ============================================================================


st.title("💬 Estate Planner Chat")

api_base_url = st.session_state["api_base_url"]
active_chat = get_active_chat_session()
active_chat_messages = active_chat["messages"]

try:
    documents_index = fetch_documents_index(api_base_url)
except Exception:
    documents_index = []
documents_by_id = build_documents_by_id(documents_index)

with st.sidebar:

    if st.button("➕ New Chat", use_container_width=True):
        create_new_chat_session()
        st.rerun()

    if st.button("🗑️ Clear Current Chat", use_container_width=True):
        active_chat["messages"] = []
        active_chat["last_query"] = None
        st.session_state["pending_prompt"] = None
        st.rerun()

    st.markdown("---")
    st.markdown("### Chat Log")

    selected_chat_id: str | None = None
    for chat in st.session_state["chat_sessions"]:
        chat_id = str(chat.get("id") or "")
        title = str(chat.get("title") or "").strip() or "Untitled chat"
        label = shorten_text(title, max_length=26)
        is_active = chat_id == str(st.session_state.get("active_chat_id") or "")

        if st.button(
            label,
            key=f"chat-log-{chat_id}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            selected_chat_id = chat_id

    if selected_chat_id and selected_chat_id != str(st.session_state.get("active_chat_id") or ""):
        st.session_state["active_chat_id"] = selected_chat_id
        st.session_state["pending_prompt"] = None
        st.rerun()

apply_app_styles()

tab_chat, tab_graph = st.tabs(["💬 Chat", "🔗 Knowledge Graph"])


# ============================================================================
# CHAT TAB
# ============================================================================


with tab_chat:
    for message in active_chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            if message["role"] == "assistant":

                if message.get("is_uncertain") and message.get("uncertainty_message"):
                    st.warning(message["uncertainty_message"])

                render_timeline(message.get("timeline_events") or [])
                render_relationship_cards(message.get("relationship_cards") or [])
                render_sources(
                    message.get("sources") or [],
                    documents_by_id=documents_by_id,
                )

    recent_user_query, recent_assistant_sources = get_recent_chat_context(active_chat_messages)
    suggested_questions = build_suggested_questions(
        documents_index,
        last_user_query=recent_user_query,
        last_assistant_sources=recent_assistant_sources,
        documents_by_id=documents_by_id,
    )
    selected_template = render_suggested_questions(suggested_questions)
    if selected_template:
        st.session_state["pending_prompt"] = selected_template
        st.rerun()

    live_message_container = st.container()

    prompt = st.chat_input("Ask a question about the estate documents...")
    if prompt is None and st.session_state.get("pending_prompt"):
        prompt = str(st.session_state.get("pending_prompt") or "").strip()
        st.session_state["pending_prompt"] = None

    if prompt:
        user_message = {"role": "user", "content": prompt}
        active_chat_messages.append(user_message)
        maybe_update_chat_title(active_chat, prompt)

        payload: dict[str, Any] = {
            "question": prompt,
            "top_k": 5,
        }

        with live_message_container:
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = post_chat_request(api_base_url, payload)
                    except Exception as exc:
                        show_chat_error(exc)
                        st.stop()

        assistant_content = str(response.get("answer") or "").strip() or "No answer returned."

        sources = normalize_sources(response.get("sources") or [])
        is_uncertain = bool(response.get("is_uncertain"))
        uncertainty_message = str(response.get("uncertainty_message") or "").strip() or None

        timeline_candidates = build_timeline_events(
            sources=sources,
            documents_by_id=documents_by_id,
        )
        timeline_events = (
            timeline_candidates if should_show_timeline(prompt, timeline_candidates) else []
        )

        relationship_cards: list[dict[str, Any]] = []
        if is_relationship_query(prompt):
            relationship_payload = {
                "query": prompt,
                "depth": 2,
                "max_nodes": 60,
                "max_seed_people": 6,
                "max_seed_properties": 3,
            }
            try:
                relationship_graph = fetch_query_subgraph(api_base_url, relationship_payload)
                relationship_cards = build_relationship_cards(relationship_graph)
            except Exception:
                relationship_cards = []

        assistant_message = {
            "role": "assistant",
            "content": assistant_content,
            "sources": sources,
            "is_uncertain": is_uncertain,
            "uncertainty_message": uncertainty_message,
            "timeline_events": timeline_events,
            "relationship_cards": relationship_cards,
            "original_query": prompt,
        }
        active_chat_messages.append(assistant_message)
        active_chat["last_query"] = prompt

        st.rerun()


# ============================================================================
# KNOWLEDGE GRAPH TAB
# ============================================================================


with tab_graph:
    st.subheader("Query Visualization")

    last_query = str(active_chat.get("last_query") or "").strip()
    if last_query:
        preview = shorten_text(last_query, max_length=100)
        st.info(f"📍 Auto-loaded from last chat query: **{preview}**")

        if st.button("🔄 Refresh Graph", use_container_width=True):
            render_knowledge_graph(
                api_base_url,
                last_query,
                depth=2,
                max_nodes=80,
                max_seed_people=3,
                max_seed_properties=3,
            )
    else:
        st.info("💡 Ask a question in the Chat tab first to visualize its knowledge graph.")
