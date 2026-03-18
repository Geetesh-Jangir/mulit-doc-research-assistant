# 🔬 Multi-Document Research Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mulit-doc-research-assistant.streamlit.app/)

A production-grade RAG (Retrieval-Augmented Generation) application that allows users to upload multiple PDFs and URLs, then ask questions across all documents simultaneously using advanced retrieval techniques.

---

## 🚀 Live Demo
**👉 [Try it here](https://mulit-doc-research-assistant.streamlit.app/)**

---

## ✨ Features

- 📄 Upload multiple PDFs and URLs as knowledge sources
- 🔍 RAG Fusion — generates multiple query variations for broader retrieval
- 🏆 Reciprocal Rank Fusion (RRF) reranking — combines results from all queries
- 💬 Multi-turn conversation with full chat history
- 🚀 Groq LLaMA 3.3-70B for fast, accurate answers
- 🧠 HuggingFace embeddings — runs locally, no extra API needed
- 🎯 Grounded answers — only uses content from your uploaded documents

---

## 🏗️ Architecture

```
User uploads PDFs / URLs
         │
         ▼
Document Processing Pipeline
  ├── PDF Loader (PyPDF)
  ├── URL Loader (WebBaseLoader)
  ├── RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
  └── Deduplication + Noise Filtering
         │
         ▼
HuggingFace Embeddings (all-MiniLM-L6-v2)
         │
         ▼
Chroma Vector Store
         │
         ▼
RAG Fusion Pipeline
  ├── Generate 3 query variations (Groq LLaMA)
  ├── Retrieve docs for each query
  └── RRF Reranking (k=60)
         │
         ▼
Groq LLaMA 3.3-70B → Final Answer
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend UI | Streamlit |
| LLM | Groq (LLaMA 3.3-70B-Versatile) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| RAG Framework | LangChain |
| PDF Loading | PyPDFLoader |
| URL Loading | WebBaseLoader (BeautifulSoup4) |
| Retrieval | RAG Fusion + Reciprocal Rank Fusion |
| Deployment | Streamlit Community Cloud |

---

## ⚙️ How RAG Fusion + RRF Works

```
User Question: "What is attention mechanism?"
       │
       ▼
LLM generates 3 query variations:
  1. "How does attention mechanism work?"
  2. "Explain self-attention in transformers"
  3. "What is the purpose of attention in neural networks?"
       │
       ▼
Retriever runs for EACH query → 3 separate result lists
       │
       ▼
RRF Formula: score = 1 / (rank + 60)
  Doc appearing in multiple lists → higher combined score
       │
       ▼
Top ranked unique documents → LLM → Grounded Answer
```

---

## 🚀 Run Locally

```bash
# Clone repository
git clone https://github.com/Geetesh-Jangir/mulit-doc-research-assistant.git
cd mulit-doc-research-assistant

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

You will need a free [Groq API key](https://console.groq.com).

---

## 📁 Project Structure

```
mulit-doc-research-assistant/
├── app.py                  ← Streamlit entry point + sidebar logic
├── ui.py                   ← All rendering functions
├── rag_pipeline.py         ← RAG Fusion + RRF core logic
├── document_processor.py   ← PDF + URL loading and chunking
├── requirements.txt
├── runtime.txt
├── packages.txt
└── static/
    └── style.css
```

---

## 🔑 Environment Variables

For local development, create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

---

## 📊 Why RAG Fusion Over Basic RAG?

| | Basic RAG | RAG Fusion + RRF |
|--|----------|-----------------|
| Queries used | 1 | 3 (variations) |
| Coverage | Limited | Broader |
| Handles synonyms | ❌ | ✅ |
| Result ranking | Cosine similarity | Combined RRF score |
| Answer quality | Good | Significantly better |
