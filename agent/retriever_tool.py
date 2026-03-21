# agent/retriever_tool.py
import time
from langchain.tools import tool
from vectorstore.faiss_store import FaissRetriever
from embeddings.embed import EmbeddingService
from utils.logger import log

# Module-level singletons — loaded once when module is first imported
embedding_service = EmbeddingService()
retriever = FaissRetriever(embedding_service)


def get_retriever() -> FaissRetriever:
    """
    Returns the current retriever instance.
    chat_app.py uses this to call add_documents() when user uploads a file.
    """
    return retriever


@tool(name="retriever_tool")
def retriever_tool(query: str):
    """
    Search the knowledge base and return relevant document chunks.
    """
    log("Retriever tool invoked")
    start = time.time()
    docs = retriever.retrieve(query, top_k=3)
    retrieval_time = time.time() - start
    log(f"Retrieved {len(docs)} documents")
    log(f"Retrieval latency: {retrieval_time:.2f}s")

    cleaned_chunks = []
    sources = []

    for i, doc in enumerate(docs):
        text = " ".join(doc.page_content.split())[:600]
        log(f"Chunk {i+1} retrieved")
        cleaned_chunks.append(text)
        source = doc.metadata.get("source", "Local Document")
        if source not in sources:
            sources.append(source)

    return {
        "chunks": cleaned_chunks,
        "text": "\n\n".join(cleaned_chunks),
        "sources": sources
    }