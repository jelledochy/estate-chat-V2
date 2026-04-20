from __future__ import annotations

import datetime as dt
import os
import re
from collections import defaultdict
from typing import Any

import httpx
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

DEFAULT_API_BASE_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

PERSON_COLOR_SELECTED = "#1D4ED8"
PERSON_COLOR_DEFAULT = "#93C5FD"
EDGE_COLOR_PARENT = "#2563EB"
EDGE_COLOR_PARTNER = "#7C3AED"
EDGE_COLOR_GRANDPARENT = "#0F766E"

ORG_HINTS = (
    " nv",
    " bv",
    " bvba",
    " bank",
    " foundation",
    " purchasers",
    "projects",
    "residence",
    "development",
    "vastgoed",
)

SKIP_NAMES = {
    "purchasing together",
    "third-party purchasers",
}


st.title("Persons")
st.caption(
    "Person-level overview of possessions, static family-tree links, and supporting documents."
)


# ============================================================================
# API Helpers
# ============================================================================


@st.cache_data(show_spinner=False, ttl=120)
def fetch_documents(api_base_url: str) -> list[dict[str, Any]]:
    url = f"{api_base_url.rstrip('/')}" + "/api/documents"
    with httpx.Client(timeout=30.0) as client:
        response = client.get(url)
        response.raise_for_status()
        payload = response.json()

    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


# ============================================================================
# Formatting Helpers
# ============================================================================


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def normalize_key(value: Any) -> str:
    return normalize_text(value).casefold()


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


def format_date(value: Any) -> str:
    parsed = parse_date(value)
    return parsed.isoformat() if parsed else "-"


def format_relation_label(value: str) -> str:
    mapping = {
        "partner_of": "Partner",
        "parent_of": "Parent",
        "grandparent_of": "Grandparent",
    }
    return mapping.get(value, value.replace("_", " ").strip().title())


def show_request_error(exc: Exception) -> None:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        detail = ""
        try:
            payload = exc.response.json()
            detail = str(payload.get("detail") or "")
        except Exception:
            detail = exc.response.text
        st.error(f"Backend returned HTTP {status_code}. {detail}".strip())
        return

    if isinstance(exc, httpx.RequestError):
        st.error(
            "Could not reach backend API. Check that FastAPI is running and the API URL is correct."
        )
        return

    st.error(f"Unexpected backend error: {exc}")


# ============================================================================
# Person + Relationship Builders
# ============================================================================


def is_probably_person(raw_name: str) -> bool:
    name = normalize_text(raw_name)
    if not name:
        return False

    lowered = f" {name.casefold()} "
    if name.casefold() in SKIP_NAMES:
        return False
    if any(token in lowered for token in ORG_HINTS):
        return False
    return True


def normalize_person_name(raw_name: str) -> str | None:
    name = normalize_text(raw_name)
    if not is_probably_person(name):
        return None
    return name


def names_match(left: str, right: str) -> bool:
    left_key = normalize_key(left)
    right_key = normalize_key(right)
    if not left_key or not right_key:
        return False
    return left_key == right_key or left_key in right_key or right_key in left_key


def canonical_person(name: str, canonical_people: list[str]) -> str:
    for candidate in canonical_people:
        if names_match(name, candidate):
            return candidate
    return normalize_text(name)


def unique_people(names: list[str]) -> list[str]:
    deduped: dict[str, str] = {}
    for name in names:
        normalized = normalize_person_name(name)
        if not normalized:
            continue
        key = normalized.casefold()
        if key not in deduped:
            deduped[key] = normalized
    return sorted(deduped.values(), key=lambda item: item.casefold())


def extract_people(documents: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    ordered_documents = sorted(
        documents,
        key=lambda item: parse_date(item.get("date")) or dt.date.min,
    )

    for document in ordered_documents:
        parties = document.get("parties")
        if not isinstance(parties, dict):
            continue

        for raw_names in parties.values():
            if isinstance(raw_names, list):
                names.extend(str(item) for item in raw_names)
            else:
                names.append(str(raw_names))

    return unique_people(names)


def people_for_role(parties: dict[str, Any], role_name: str, all_people: list[str]) -> list[str]:
    raw_names = parties.get(role_name)
    if raw_names is None:
        return []

    values = raw_names if isinstance(raw_names, list) else [raw_names]
    output: list[str] = []
    seen: set[str] = set()

    for raw in values:
        normalized = normalize_person_name(str(raw))
        if not normalized:
            continue
        canonical = canonical_person(normalized, all_people)
        marker = canonical.casefold()
        if marker in seen:
            continue
        seen.add(marker)
        output.append(canonical)

    return output


def note_mentions_parent(note_text: str, parent: str) -> bool:
    low = note_text.casefold()
    parent_low = parent.casefold()
    patterns = (
        f"daughter of {parent_low}",
        f"son of {parent_low}",
        f"child of {parent_low}",
    )
    return any(pattern in low for pattern in patterns)


def note_mentions_grandchild(note_text: str) -> bool:
    low = note_text.casefold()
    return "granddaughter" in low or "grandson" in low or "grandchild" in low

def surname_parts(name: str) -> set[str]:
    tokens = normalize_text(name).casefold().replace("-", " ").split()
    if len(tokens) <= 1:
        return set()

    parts: set[str] = set()
    for token in tokens[1:]:
        for fragment in re.findall(r"[a-z]+", token):
            if len(fragment) >= 3:
                parts.add(fragment)
    return parts


def shares_family_name(left: str, right: str) -> bool:
    return bool(surname_parts(left).intersection(surname_parts(right)))


def should_infer_parent_from_donation(
    donor: str,
    donee: str,
    donees: list[str],
) -> bool:
    if shares_family_name(donor, donee):
        return True

    if len(donees) <= 1:
        return True

    # If another donee shares the donor family name, a non-matching donee is
    # likely an in-law/partner rather than a direct child.
    for other_donee in donees:
        if names_match(other_donee, donee):
            continue
        if shares_family_name(donor, other_donee):
            return False

    return True


def add_relation(
    relation_map: dict[tuple[str, str, str], set[str]],
    source: str,
    target: str,
    relation: str,
    document_id: str,
) -> None:
    source_name = normalize_text(source)
    target_name = normalize_text(target)
    if not source_name or not target_name or source_name.casefold() == target_name.casefold():
        return

    key = (source_name, target_name, relation)
    relation_map[key].add(document_id)


def build_static_relationships(
    documents: list[dict[str, Any]],
    people: list[str],
) -> dict[tuple[str, str, str], set[str]]:
    relations: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    for document in documents:
        document_id = normalize_text(document.get("document_id")) or "unknown_doc"
        parties = document.get("parties")
        if not isinstance(parties, dict):
            continue

        buyers = people_for_role(parties, "buyers", people)
        borrowers = people_for_role(parties, "borrowers", people)
        donors = people_for_role(parties, "donors", people)
        donees = people_for_role(parties, "donees", people)
        testators = people_for_role(parties, "testator", people)
        residual_heirs = people_for_role(parties, "residual_heirs", people)
        specific_beneficiaries = people_for_role(parties, "specific_beneficiaries", people)

        note_text = normalize_text(document.get("notes"))

        raw_buyers = parties.get("buyers")
        has_together_marker = False
        if isinstance(raw_buyers, list):
            has_together_marker = any(
                normalize_key(item) == "purchasing together" for item in raw_buyers
            )

        if len(buyers) == 2 and has_together_marker:
            left, right = sorted(buyers, key=lambda item: item.casefold())
            add_relation(relations, left, right, "partner_of", document_id)

        if len(borrowers) == 2:
            left, right = sorted(borrowers, key=lambda item: item.casefold())
            add_relation(relations, left, right, "partner_of", document_id)

        if len(donors) == 2 and donees:
            left, right = sorted(donors, key=lambda item: item.casefold())
            add_relation(relations, left, right, "partner_of", document_id)

        for donor in donors:
            for donee in donees:
                relation_type = (
                    "grandparent_of"
                    if note_mentions_grandchild(note_text)
                    else "parent_of"
                )
                if relation_type == "parent_of" and not should_infer_parent_from_donation(
                    donor, donee, donees
                ):
                    continue

                add_relation(relations, donor, donee, relation_type, document_id)

        # Keep explicit parent statements from notes.
        for candidate_child in donees + residual_heirs + specific_beneficiaries:
            for candidate_parent in people:
                if note_mentions_parent(note_text, candidate_parent):
                    add_relation(relations, candidate_parent, candidate_child, "parent_of", document_id)

        # Residual heirs can default to children; specific beneficiaries do not.
        for testator in testators:
            for heir in residual_heirs:
                add_relation(relations, testator, heir, "parent_of", document_id)

    # If a pair appears as both parent and grandparent, keep grandparent only.
    for source, target, relation in list(relations):
        if relation != "grandparent_of":
            continue

        parent_key = (source, target, "parent_of")
        parent_evidence = relations.get(parent_key)
        if parent_evidence:
            relations[(source, target, "grandparent_of")].update(parent_evidence)
            relations.pop(parent_key, None)

    return relations


def build_person_mentions(
    documents: list[dict[str, Any]],
    selected_person: str,
    all_people: list[str],
) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []

    for document in documents:
        parties = document.get("parties")
        if not isinstance(parties, dict):
            continue

        matching_roles: list[str] = []
        for role, raw_names in parties.items():
            names = raw_names if isinstance(raw_names, list) else [raw_names]
            normalized_role = normalize_text(role)
            if not normalized_role:
                continue

            role_has_person = False
            for raw_name in names:
                normalized = normalize_person_name(str(raw_name))
                if not normalized:
                    continue
                canonical = canonical_person(normalized, all_people)
                if names_match(canonical, selected_person):
                    role_has_person = True
                    break

            if role_has_person:
                matching_roles.append(normalized_role)

        if not matching_roles:
            continue

        raw_property = document.get("property")
        property_address = None
        if isinstance(raw_property, dict):
            property_address = normalize_text(raw_property.get("address")) or None

        mentions.append(
            {
                "document_id": normalize_text(document.get("document_id")),
                "document_type": normalize_text(document.get("document_type")) or "unknown",
                "date": parse_date(document.get("date")),
                "date_text": format_date(document.get("date")),
                "roles": sorted(set(matching_roles), key=lambda item: item.casefold()),
                "property": property_address,
                "notes": normalize_text(document.get("notes")),
            }
        )

    mentions.sort(
        key=lambda item: (
            item.get("date") or dt.date.min,
            str(item.get("document_id") or ""),
        ),
        reverse=True,
    )
    return mentions


def build_property_rows(mentions: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows_by_address: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"roles": set(), "documents": set()}
    )

    for mention in mentions:
        address = normalize_text(mention.get("property"))
        if not address:
            continue

        for role in mention.get("roles") or []:
            rows_by_address[address]["roles"].add(normalize_text(role))

        document_id = normalize_text(mention.get("document_id"))
        if document_id:
            rows_by_address[address]["documents"].add(document_id)

    rows: list[dict[str, str]] = []
    for address in sorted(rows_by_address, key=lambda item: item.casefold()):
        roles = sorted(rows_by_address[address]["roles"], key=lambda item: item.casefold())
        docs = sorted(rows_by_address[address]["documents"], key=lambda item: item.casefold())

        rows.append(
            {
                "Property": address,
                "Person role(s)": ", ".join(roles) if roles else "-",
                "Evidence": ", ".join(docs) if docs else "-",
            }
        )

    return rows


def build_relation_rows_for_person(
    selected_person: str,
    relation_map: dict[tuple[str, str, str], set[str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for (source, target, relation), document_ids in relation_map.items():
        doc_text = ", ".join(sorted(document_ids, key=lambda item: item.casefold()))

        if relation == "partner_of":
            if names_match(source, selected_person):
                rows.append(
                    {
                        "Relation": "Partner",
                        "Person": target,
                        "Evidence": doc_text,
                    }
                )
            elif names_match(target, selected_person):
                rows.append(
                    {
                        "Relation": "Partner",
                        "Person": source,
                        "Evidence": doc_text,
                    }
                )
            continue

        if names_match(source, selected_person):
            rows.append(
                {
                    "Relation": f"{format_relation_label(relation)} of",
                    "Person": target,
                    "Evidence": doc_text,
                }
            )
        elif names_match(target, selected_person):
            reverse_label = "Child of" if relation == "parent_of" else "Grandchild of"
            rows.append(
                {
                    "Relation": reverse_label,
                    "Person": source,
                    "Evidence": doc_text,
                }
            )

    rows.sort(key=lambda item: (item["Relation"].casefold(), item["Person"].casefold()))

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        marker = (row["Relation"], row["Person"], row["Evidence"])
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(row)

    return deduped


def family_people_with_links(
    relation_map: dict[tuple[str, str, str], set[str]],
) -> set[str]:
    people: set[str] = set()
    for (source, target, _relation), evidence_docs in relation_map.items():
        if not evidence_docs:
            continue
        people.add(source)
        people.add(target)
    return people


def build_global_tree_levels(
    scope: set[str],
    relation_map: dict[tuple[str, str, str], set[str]],
) -> dict[str, int]:
    # Fixed 3-row family tree:
    # 0 = grandparents/top generation, 1 = parents + partners, 2 = children.
    parent_edges: list[tuple[str, str]] = []
    partner_pairs: list[tuple[str, str]] = []

    for (source, target, relation), evidence_docs in relation_map.items():
        if not evidence_docs or source not in scope or target not in scope:
            continue

        if relation == "parent_of":
            parent_edges.append((source, target))
        elif relation == "partner_of":
            partner_pairs.append((source, target))

    children_by_parent: dict[str, set[str]] = defaultdict(set)
    parents_by_child: dict[str, set[str]] = defaultdict(set)
    for parent, child in parent_edges:
        children_by_parent[parent].add(child)
        parents_by_child[child].add(parent)

    parent_nodes = set(children_by_parent.keys())
    child_nodes = set(parents_by_child.keys())

    top_generation = {name for name in parent_nodes if name not in child_nodes}
    if not top_generation:
        top_generation = set(parent_nodes)

    second_generation: set[str] = set()
    for grandparent in top_generation:
        second_generation.update(children_by_parent.get(grandparent) or set())

    third_generation: set[str] = set()
    for parent in second_generation:
        third_generation.update(children_by_parent.get(parent) or set())

    # Put partners on the same row as their linked family member.
    changed = True
    while changed:
        changed = False
        for left, right in partner_pairs:
            if left in top_generation and right not in top_generation:
                top_generation.add(right)
                changed = True
            if right in top_generation and left not in top_generation:
                top_generation.add(left)
                changed = True

            if left in second_generation and right not in second_generation:
                second_generation.add(right)
                changed = True
            if right in second_generation and left not in second_generation:
                second_generation.add(left)
                changed = True

            if left in third_generation and right not in third_generation:
                third_generation.add(right)
                changed = True
            if right in third_generation and left not in third_generation:
                third_generation.add(left)
                changed = True

    # Resolve overlaps with explicit row priority: top -> second -> third.
    second_generation -= top_generation
    third_generation -= top_generation
    third_generation -= second_generation

    # Any remaining linked people go in second row to keep the tree compact.
    remaining = scope - top_generation - second_generation - third_generation
    second_generation.update(remaining)

    levels: dict[str, int] = {}
    for name in top_generation:
        levels[name] = 0
    for name in second_generation:
        levels[name] = 1
    for name in third_generation:
        levels[name] = 2

    return levels


def render_static_family_tree(
    relation_map: dict[tuple[str, str, str], set[str]],
) -> None:
    scope = family_people_with_links(relation_map)
    if not scope:
        st.info("No family links inferred yet from the available documents.")
        return

    level_by_person = build_global_tree_levels(scope, relation_map)

    node_names = sorted(
        scope,
        key=lambda item: (level_by_person.get(item, 0), item.casefold()),
    )
    nodes: list[Node] = []
    for name in node_names:
        level = level_by_person.get(name, 0)
        nodes.append(
            Node(
                id=name,
                label=name,
                level=level,
                color=PERSON_COLOR_SELECTED if level == 0 else PERSON_COLOR_DEFAULT,
                size=30 if level == 0 else 22,
                shape="dot",
            )
        )

    edges: list[Edge] = []
    seen_partner_edges: set[tuple[str, str]] = set()

    for (source, target, relation), document_ids in relation_map.items():
        if source not in scope or target not in scope:
            continue

        title_docs = ", ".join(sorted(document_ids, key=lambda item: item.casefold()))

        if relation == "partner_of":
            edge_key = tuple(sorted((source, target), key=lambda item: item.casefold()))
            if edge_key in seen_partner_edges:
                continue
            seen_partner_edges.add(edge_key)
            edges.append(
                Edge(
                    source=edge_key[0],
                    target=edge_key[1],
                    label="partner",
                    title=f"Evidence: {title_docs}",
                    color=EDGE_COLOR_PARTNER,
                    arrows="none",
                    dashes=True,
                )
            )
            continue

        if relation != "parent_of":
            # Skip direct grandparent->grandchild links in the visual tree.
            continue

        edges.append(
            Edge(
                source=source,
                target=target,
                label="parent",
                title=f"Evidence: {title_docs}",
                color=EDGE_COLOR_PARENT,
            )
        )

    if not edges:
        st.info("No family links inferred yet from the available documents.")
        return

    config = Config(
        width=960,
        height=620,
        directed=True,
        physics=False,
        hierarchical=True,
        direction="UD",
        sortMethod="directed",
        shakeTowards="roots",
        levelSeparation=180,
        nodeSpacing=220,
        treeSpacing=280,
    )
    agraph(nodes=nodes, edges=edges, config=config)


# ============================================================================
# State + Data Load
# ============================================================================

if "api_base_url" not in st.session_state:
    st.session_state["api_base_url"] = DEFAULT_API_BASE_URL

with st.sidebar:
    api_base_url = st.session_state["api_base_url"]

with st.spinner("Loading people and document context..."):
    try:
        documents = fetch_documents(api_base_url)
    except Exception as exc:  # noqa: BLE001
        show_request_error(exc)
        st.stop()

if not documents:
    st.info("No documents available yet.")
    st.stop()

all_people = extract_people(documents)
if not all_people:
    st.info("No person names were found in document parties.")
    st.stop()

with st.sidebar:
    st.markdown("### Person")
    selected_person = st.selectbox("Choose person", options=all_people, index=0)

relation_map = build_static_relationships(documents, all_people)
mentions = build_person_mentions(documents, selected_person, all_people)
property_rows = build_property_rows(mentions)
relationship_rows = build_relation_rows_for_person(selected_person, relation_map)

family_member_count = len(family_people_with_links(relation_map))


# ============================================================================
# UI
# ============================================================================


metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
metric_col_1.metric("Documents Mentioned", len(mentions))
metric_col_2.metric("Related Properties", len(property_rows))
metric_col_3.metric("Family Members", family_member_count)

summary_bits: list[str] = []
if mentions:
    most_recent = mentions[0]
    summary_bits.append(
        f"Most recent mention: {most_recent.get('document_id') or '-'} ({most_recent.get('date_text') or '-'})"
    )
if family_member_count:
    summary_bits.append(f"Family members in tree: {family_member_count}")
if property_rows:
    summary_bits.append(f"Property involvement records: {len(property_rows)}")
if summary_bits:
    st.info(" | ".join(summary_bits))

section_overview, section_family, section_documents = st.tabs(
    ["Overview", "Family Tree", "Document Mentions"]
)

with section_overview:
    st.markdown("### Possessions and Property Involvement")
    if property_rows:
        st.dataframe(property_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No property links found for this person in document mentions.")

    st.markdown("### Relationship Snapshot")
    if relationship_rows:
        st.dataframe(relationship_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No static family links could be inferred yet for this person.")

with section_family:
    st.markdown("### Family Tree (Document-Derived)")
    st.caption("Built from document party roles and relationship notes.")
    render_static_family_tree(relation_map)

with section_documents:
    st.markdown("### Documents Where This Person Is Mentioned")

    if mentions:
        table_rows = [
            {
                "Document ID": item["document_id"],
                "Date": item["date_text"],
                "Type": item["document_type"],
                "Roles": ", ".join(item["roles"]),
                "Property": item.get("property") or "-",
            }
            for item in mentions
        ]
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        st.markdown("### Mention Details")
        for item in mentions:
            title = (
                f"{item['document_id']} | {item['date_text']} | "
                f"{', '.join(item['roles'])}"
            )
            with st.expander(title):
                st.write(f"Type: {item['document_type']}")
                st.write(f"Property: {item.get('property') or '-'}")
                notes = item.get("notes") or ""
                if notes:
                    st.write(f"Notes: {notes}")
    else:
        st.info("No documents mention this person in parties.")
