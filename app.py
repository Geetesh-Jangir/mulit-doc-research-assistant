import streamlit as st
from document_processor import load_pdfs, load_urls, process_documents
from ui import (
    load_css,
    render_header,
    render_status_banner,
    render_sidebar_header,
    render_sidebar_stats,
    render_chat_history,
)
from rag_pipeline import RAGPipeline
import os

# page config
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()

# session state
defaults = {
    "chat_history": [],
    "pipeline": None,
    "docs_loaded": False,
    "doc_count": 0,
    "source_names": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# sidebar
with st.sidebar:
    render_sidebar_header()
    st.divider()

    # api key
    st.markdown(
        '<div class="sidebar-header">⚙ Configuration</div>', unsafe_allow_html=True
    )
    try:
        groq_api_key = st.secrets.get("GROQ_API_KEY") or st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get your free API key at console.groq.com",
        )
    except Exception:
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get your free API key at console.groq.com",
        )

    st.divider()

    # document input
    st.markdown(
        '<div class="sidebar-header">📄 Documents</div>', unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "upload_pdfs", type=["pdf"], accept_multiple_files=True
    )

    url_input = st.text_area(
        "Enter URLs (one per line)",
        placeholder="https://example.com/article\nhttps://another.com/page",
        height=100,
    )

    process_btn = st.button("⚡ Process Documents", use_container_width=True)

    # document processing

    if process_btn:
        if not groq_api_key:
            st.error("please enter your API key first")
        elif not uploaded_files and not url_input.strip():
            st.warning("please upload PDFs or enter URLs.")
        else:
            all_docs = []

            # load pdfs
            if uploaded_files:
                with st.spinner(f"Loading {len(uploaded_files)} PDFs..."):
                    pdf_docs = load_pdfs(uploaded_files)
                    all_docs.extend(pdf_docs)
                st.success(f"✅ Loaded {len(uploaded_files)} PDF(s)")

            # load urls
            if url_input.strip():
                urls = [u.strip() for u in url_input.strip().split("\n") if u.strip()]
                with st.spinner(f"Loading {len(urls)} URL(s)..."):
                    url_docs = load_urls(urls)
                    all_docs.extend(url_docs)
                st.success(f"✅ Loaded {len(urls)} URL(s)")

            if all_docs:
                # chunk and clean
                with st.spinner("chunking and cleaning documents..."):
                    processed = process_documents(all_docs)

                # embedding and vectorestores
                with st.spinner("Building vector stores..."):
                    pipeline = RAGPipeline(groq_api_key=groq_api_key)
                    pipeline.build_vectorstore(processed)

                # save to session
                st.session_state.pipeline = pipeline
                st.session_state.docs_loaded = True
                st.session_state.doc_count = len(processed)
                st.session_state.chat_history = []
                st.session_state.source_names = list(
                    {doc.metadata.get("source", "unknown") for doc in processed}
                )
                st.success(f"Ready! {len(processed)} chunks indexed.")
            else:
                st.error("No documents could be loaded. Check you inputs.")

    # stats panel
    if st.session_state.docs_loaded:
        st.divider()
        render_sidebar_stats(
            doc_count=st.session_state.doc_count,
            source_names=st.session_state.source_names,
        )
    st.divider()

    # clear chat
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# Main page
render_header()

render_status_banner(
    docs_loaded=st.session_state.docs_loaded,
    doc_count=st.session_state.doc_count,
    source_count=len(st.session_state.source_names),
)

st.divider()

# render chat history
render_chat_history(st.session_state.chat_history)

# Chat input

user_input = st.chat_input(
    "Ask a question about your document...", disabled=not st.session_state.docs_loaded
)

if user_input:
    # save user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # get answer from RAG Pipeline
    with st.spinner("🔍 Searching with RAG Fusion + RRF Reranking..."):
        result = st.session_state.pipeline.answer(
            question=user_input,
            chat_history=st.session_state.chat_history[
                :-1
            ],  # exclude the current query
        )

    # save assistant messages with metadata
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "queries": result["queries"],
            "sources": result["sources"],
        }
    )

    st.rerun()
