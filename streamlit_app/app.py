import os

import httpx
import streamlit as st

st.set_page_config(page_title="Estate Planner Chat", page_icon=":scroll:", layout="wide")

DEFAULT_API_BASE_URL = os.getenv("BACKEND_API_URL", "http://localhost:8001")


def ask_rag(api_base_url: str, question: str, top_k: int) -> dict:
    url = f"{api_base_url.rstrip('/')}/api/chat"
    with httpx.Client(timeout=120.0) as client:
        response = client.post(url, json={"question": question, "top_k": top_k})
        response.raise_for_status()
        return response.json()


def show_request_error(exc: Exception) -> None:
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except Exception:
            detail = exc.response.text
        st.error(f"Backend returned HTTP {exc.response.status_code}. {detail}")
        return

    if isinstance(exc, httpx.RequestError):
        st.error("Could not reach the FastAPI backend. Check that it is running.")
        return

    st.error(f"Unexpected error: {exc}")


st.title("Estate Planner Chat")
st.caption("Ask a question and the FastAPI backend will answer with `backend/RAG.py`.")

with st.sidebar:
    st.header("Settings")
    api_base_url = st.text_input("Backend API URL", value=DEFAULT_API_BASE_URL)
    top_k = st.slider("Retrieved context items", min_value=1, max_value=10, value=5)

    if st.button("Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        sources = message.get("sources") or []
        if sources:
            with st.expander("Sources"):
                for source in sources:
                    document_id = source.get("document_id") or "unknown"
                    document_type = source.get("document_type") or "unknown"
                    pages = source.get("page_numbers") or []
                    page_label = ", ".join(str(page) for page in pages) if pages else "-"
                    st.markdown(f"**{document_id}** | {document_type} | pages: {page_label}")
                    if source.get("excerpt"):
                        st.caption(source["excerpt"])

        if message.get("uncertainty_message"):
            st.warning(message["uncertainty_message"])

prompt = st.chat_input("Ask about the estate documents...")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Running RAG..."):
            try:
                result = ask_rag(api_base_url, prompt, top_k)
            except Exception as exc:
                show_request_error(exc)
                st.stop()

        answer = result.get("answer") or "No answer returned."
        st.write(answer)

        sources = result.get("sources") or []
        if sources:
            with st.expander("Sources"):
                for source in sources:
                    document_id = source.get("document_id") or "unknown"
                    document_type = source.get("document_type") or "unknown"
                    pages = source.get("page_numbers") or []
                    page_label = ", ".join(str(page) for page in pages) if pages else "-"
                    st.markdown(f"**{document_id}** | {document_type} | pages: {page_label}")
                    if source.get("excerpt"):
                        st.caption(source["excerpt"])

        uncertainty_message = result.get("uncertainty_message")
        if uncertainty_message:
            st.warning(uncertainty_message)

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "uncertainty_message": uncertainty_message,
        }
    )
