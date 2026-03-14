import os
import tempfile

# setting up the environment of user agent
os.environ.setdefault("USER_AGENT", "MultiDocResearchAssistant")
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4


from typing import List


def load_pdfs(uploaded_files) -> List[Document]:
    """Load the documents from uploaded pdfs"""
    docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_path)
            docs.extend(loader.load())
        finally:
            os.unlink(tmp_path)
    return docs


def load_urls(urls: List[str]) -> List[Document]:
    """load the document from the urls"""
    docs = []
    for url in urls:
        url = url.strip()
        if not url:
            continue

        try:
            loader = WebBaseLoader(
                web_paths=(url,),
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(
                        name=[
                            "p",
                            "h1",
                            "h2",
                            "h3",
                            "h4",
                            "h5",
                            "article",
                            "section",
                            "main",
                        ]
                    )
                ),
            )
            loaded = loader.load()
            docs.extend(loaded)
        except Exception as e:
            print(f"Error loading url {url}: {e}")
    return docs


def process_documents(docs: List[Document]) -> List[Document]:
    """split, deduplicate and clean documents"""

    # split documents (text_splitter)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)

    # filter out the short chunks
    splits = [doc for doc in splits if len(doc.page_content) > 150]

    # deduplicate
    seen = set()
    unique_splits = []
    for split in splits:
        content = split.page_content.strip()
        if content not in seen:
            seen.add(content)
            unique_splits.append(split)
    return unique_splits
