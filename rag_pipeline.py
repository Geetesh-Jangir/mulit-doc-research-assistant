from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq
from langchain_core.load import dumps, loads
from operator import itemgetter


# prompts

QUERY_GENERATION_TEMPLATE = """You are a helpful assistant that generates multiple search queries.
Given the conversation history and the latest user question, generate 3 different versions 
of the question to retrieve relevant documents from a vector database.

Generate queries that cover different angles and phrasings of the same question.
Output ONLY a numbered list of 3 queries, no explanations.

Example format:
1. query one
2. query two  
3. query three

Conversation History:
{chat_history}

Question: {question}

Output (3 queries):"""

RAG_TEMPLATE = """
You are a helpful research assistant.

Answer the user's question using the provided context.

Rules:
- Prefer information from the context.
- If the context partially contains the answer, use reasoning to explain it.
- If the context does not contain enough information, say what is missing.
- Do not invent facts not related to the context.

Format the answer with:
- A short explanation
- Bullet points when useful
- Highlight important terms using **bold**

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Answer:
"""


def format_chat_history(chat_history: List) -> str:
    """Format chat history to readable string"""
    if not chat_history:
        return "No previous conversations."
    formatted = []
    for chat in chat_history:
        role = "user" if chat["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {chat['content']}")
    return "\n".join(formatted)


def format_docs(docs: List[Document]) -> str:
    """format the docs to structured string for the prompt"""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", f"Document {i+1}")

        # shorten urlr/paths for display
        if len(source) > 60:
            source = "..." + source[-57:]
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def rank_fusion(results: List[List], k: int = 60) -> List[Document]:
    fused_scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked = [
        loads(doc)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked


# RAG Pipeline Class


class RAGPipeline:
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.vectorstore = None
        self.retriever = None
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile", api_key=groq_api_key, temperature=0
        )
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
        self.query_prompt = ChatPromptTemplate.from_template(QUERY_GENERATION_TEMPLATE)
        self.rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    def build_vectorstore(self, documents: List[Document]):
        """Build Chroma vectorstore from processed documents."""

        # clear existing vectorstores if rebuilding
        if self.vectorstore:
            self.vectorstore.delete_collection()

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name="research_docs",
        )

        # creating retriever
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

    def generate_queries(self, question: str, chat_history: List) -> List[str]:
        """Generate multiple query variation using llm"""
        history_str = format_chat_history(chat_history)
        chain = self.query_prompt | self.llm | StrOutputParser()
        result = chain.invoke({"question": question, "chat_history": history_str})

        # remove the numbers and convert the string into a list
        queries = []
        for res in result.strip().split("\n"):
            res = res.strip()
            if res and res[0].isdigit() and ". " in res:
                queries.append(res.split(". ", 1)[1].strip())

        # if parsing fails then parse original question
        if not queries:
            queries = [question]
        return queries

    def retrieve_with_fusion(self, question: str, chat_history: List) -> List[Document]:
        """retreve docs using Rag Fusion and RRF reranking"""
        queries = self.generate_queries(question, chat_history)

        # retrieve docs for each query
        all_results = []
        for query in queries:
            docs = self.retriever.invoke(query)
            # print(doc.)
            all_results.append(docs)

        # reranking
        reranked = rank_fusion(all_results)

        return reranked[:6]

    def answer(self, question: str, chat_history: List) -> dict:
        "full rag pipeline"

        if not self.retriever:
            return {
                "answer": "Please upload documents first before asking questions.",
                "queries": [],
                "sources": [],
            }

        # retrieve with fusion
        docs = self.retrieve_with_fusion(question, chat_history)
        print(len(docs))
        # format context data and chat history
        context = format_docs(docs)
        print(context[:200])
        history_str = format_chat_history(chat_history)

        # generate answer with chain
        chain = self.rag_prompt | self.llm | StrOutputParser()
        answer = chain.invoke(
            {"context": context, "question": question, "chat_history": history_str}
        )

        # collect unique sources
        sources = list({doc.metadata.get("source", "unknown") for doc in docs})

        # get generated_queries for display from the first chain
        queries = self.generate_queries(question, chat_history)

        return {"answer": answer, "queries": queries, "sources": sources, "docs": docs}
