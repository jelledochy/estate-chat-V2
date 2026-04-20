from __future__ import annotations

import html
import os
from typing import Any

import httpx
import streamlit as st

DEFAULT_API_BASE_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")
ALL_DOCUMENT_TYPES_OPTION = "All"

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


st.title("Documents")
st.caption("Browse extracted documents and open a full document view.")


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


@st.cache_data(show_spinner=False, ttl=120)
def fetch_document_detail(api_base_url: str, document_id: str) -> dict[str, Any]:
    url = f"{api_base_url.rstrip('/')}" + f"/api/documents/{document_id}"
    with httpx.Client(timeout=30.0) as client:
        response = client.get(url)
        response.raise_for_status()
        payload = response.json()

    return payload if isinstance(payload, dict) else {}


def show_request_error(exc: Exception) -> None:
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
            "Check that FastAPI is running and API URL is correct."
        )
        return

    st.error(f"Unexpected error while loading documents: {exc}")


def normalize_query_param(value: Any) -> str | None:
    if isinstance(value, list):
        value = value[0] if value else None
    normalized = str(value or "").strip()
    return normalized or None


def get_document_id_from_route() -> str | None:
    if hasattr(st, "query_params"):
        return normalize_query_param(st.query_params.get("document_id"))

    raw_values = st.experimental_get_query_params().get("document_id", [])
    return normalize_query_param(raw_values)


def set_document_id_route(document_id: str | None) -> None:
    if hasattr(st, "query_params"):
        if document_id:
            st.query_params["document_id"] = document_id
        elif "document_id" in st.query_params:
            del st.query_params["document_id"]
        return

    if document_id:
        st.experimental_set_query_params(document_id=document_id)
    else:
        st.experimental_set_query_params()


def format_date(value: Any) -> str:
    date_text = str(value or "").strip()
    return date_text or "-"


def format_amount(value: Any) -> str:
    if value is None:
        return "-"
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{amount:,.2f}"


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
    style = DOCUMENT_TYPE_STYLES.get(canonical)
    if style:
        return style["label"]

    text = str(raw_value or "").strip() or canonical
    return text.replace("_", " ").strip().title() or "Document"


def format_document_type_with_icon(raw_value: Any) -> str:
    canonical = canonical_document_type(raw_value)
    style = DOCUMENT_TYPE_STYLES.get(canonical, DOCUMENT_TYPE_STYLES["unknown"])
    return f"{style['icon']} {format_document_type_label(raw_value)}"


def render_document_type_badge_html(raw_value: Any) -> str:
    canonical = canonical_document_type(raw_value)
    style = DOCUMENT_TYPE_STYLES.get(canonical, DOCUMENT_TYPE_STYLES["unknown"])
    label = html.escape(format_document_type_label(raw_value))
    icon = html.escape(style["icon"])
    return (
        "<span class='doc-badge' "
        f"style='background:{style['bg']}; border-color:{style['border']}; color:{style['text']};'>"
        f"{icon} {label}</span>"
    )


def inject_document_styles() -> None:
    st.markdown(
        """
        <style>
        .doc-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border: 1px solid;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
            line-height: 1.1;
            padding: 0.2rem 0.6rem;
            white-space: nowrap;
        }
        .doc-detail-meta {
            color: #5b6470;
            font-size: 0.88rem;
            margin-left: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_parties_inline(raw_parties: Any) -> str:
    if not isinstance(raw_parties, dict):
        return "-"

    entries: list[str] = []
    for raw_role, raw_names in raw_parties.items():
        role = str(raw_role or "").strip()
        if not role:
            continue

        if isinstance(raw_names, list):
            names = [str(name).strip() for name in raw_names if str(name).strip()]
        else:
            name = str(raw_names or "").strip()
            names = [name] if name else []

        if names:
            entries.append(f"{role}: {', '.join(names)}")

    return " | ".join(entries) if entries else "-"


def matches_search(document: dict[str, Any], search_query: str) -> bool:
    query = search_query.casefold()

    parties_text = format_parties_inline(document.get("parties"))
    references = document.get("references")
    references_text = " ".join(str(item).strip() for item in references or [] if str(item).strip())
    haystack = " ".join(
        [
            str(document.get("document_id") or ""),
            str(document.get("filename") or ""),
            str(document.get("document_type") or ""),
            str(document.get("notes") or ""),
            parties_text,
            references_text,
        ]
    ).casefold()
    return query in haystack


inject_document_styles()

api_base_url = DEFAULT_API_BASE_URL

with st.spinner("Loading documents..."):
    try:
        documents = fetch_documents(api_base_url)
    except Exception as exc:  # noqa: BLE001
        show_request_error(exc)
        st.stop()

if not documents:
    st.info("No documents were returned by `/api/documents`.")
    st.stop()

canonical_filter_options = sorted(
    {
        canonical_document_type(item.get("document_type"))
        for item in documents
        if str(item.get("document_type") or "").strip()
    },
    key=lambda value: format_document_type_label(value).casefold(),
)
selected_document_type = st.selectbox(
    "Document type",
    options=[ALL_DOCUMENT_TYPES_OPTION, *canonical_filter_options],
    index=0,
    format_func=lambda value: (
        ALL_DOCUMENT_TYPES_OPTION
        if value == ALL_DOCUMENT_TYPES_OPTION
        else format_document_type_with_icon(value)
    ),
)
search_text = st.text_input(
    "Search",
    value="",
    placeholder="document id, filename, party, note...",
)

filtered_documents: list[dict[str, Any]] = []
for document in documents:
    if selected_document_type != ALL_DOCUMENT_TYPES_OPTION:
        raw_document_type = document.get("document_type")
        if canonical_document_type(raw_document_type) != selected_document_type:
            continue

    normalized_search = search_text.strip()
    if normalized_search and not matches_search(document, normalized_search):
        continue

    filtered_documents.append(document)

st.write(f"Showing {len(filtered_documents)} of {len(documents)} documents")

if not filtered_documents:
    st.warning("No documents match the current filters.")

rows = []
for document in filtered_documents:
    rows.append(
        {
            "Document ID": str(document.get("document_id") or ""),
            "Type": format_document_type_with_icon(document.get("document_type")),
            "Date": format_date(document.get("date")),
            "Amount": format_amount(document.get("amount")),
            "Parties": format_parties_inline(document.get("parties")),
        }
    )

if rows:
    st.dataframe(rows, use_container_width=True, hide_index=True)

selected_document_id = get_document_id_from_route()
if selected_document_id:
    with st.spinner(f"Loading document '{selected_document_id}'..."):
        try:
            detail = fetch_document_detail(api_base_url, selected_document_id)
        except Exception as exc:  # noqa: BLE001
            show_request_error(exc)
            st.stop()

    st.divider()
    st.subheader(f"Document: {selected_document_id}")

    if st.button("Back to list", use_container_width=False):
        set_document_id_route(None)
        st.rerun()

    detail_type = str(detail.get("document_type") or "unknown")
    detail_filename = str(detail.get("filename") or "")
    detail_date = format_date(detail.get("date"))

    st.markdown(
        (
            f"{render_document_type_badge_html(detail_type)}"
            f"<span class='doc-detail-meta'>{html.escape(detail_date)} | "
            f"{html.escape(detail_filename or '-')}</span>"
        ),
        unsafe_allow_html=True,
    )

    detail_parties = detail.get("parties")
    if isinstance(detail_parties, dict) and detail_parties:
        st.markdown("**Parties**")
        for role, raw_names in detail_parties.items():
            role_label = str(role or "").strip()
            if not role_label:
                continue
            if isinstance(raw_names, list):
                names = [str(name).strip() for name in raw_names if str(name).strip()]
            else:
                single_name = str(raw_names or "").strip()
                names = [single_name] if single_name else []

            if names:
                st.write(f"{role_label}: {', '.join(names)}")

    property_details = detail.get("property")
    if isinstance(property_details, dict) and property_details:
        st.markdown("**Property**")
        st.json(property_details)

    references = detail.get("references")
    if isinstance(references, list) and references:
        st.markdown("**References**")
        reference_text = ", ".join(
            str(reference).strip() for reference in references if str(reference).strip()
        )
        st.write(reference_text)

    notes = str(detail.get("notes") or "").strip()
    if notes:
        st.markdown("**Notes**")
        st.write(notes)

    full_text = str(detail.get("full_text") or "")
    st.markdown("**Extracted text**")
    st.text_area("Full text", value=full_text, height=420)

st.markdown("### Open document")
for document in filtered_documents:
    document_id = str(document.get("document_id") or "").strip()
    if not document_id:
        continue

    document_type = str(document.get("document_type") or "unknown")
    document_date = format_date(document.get("date"))
    expander_title = (
        f"{format_document_type_with_icon(document_type)} - "
        f"{document_id} - {document_date}"
    )
    with st.expander(expander_title):
        st.markdown(render_document_type_badge_html(document_type), unsafe_allow_html=True)
        st.write(f"Filename: {document.get('filename') or '-'}")
        st.write(f"Parties: {format_parties_inline(document.get('parties'))}")
        notes = str(document.get("notes") or "").strip()
        if notes:
            st.caption(notes)

        if st.button("View full document", key=f"open-{document_id}", use_container_width=True):
            set_document_id_route(document_id)
            st.rerun()
