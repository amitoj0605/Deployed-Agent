import time

from langchain.tools import tool

from vectorstore.faiss_store import FaissRetriever
from embeddings.embed import EmbeddingService

from utils.logger import log


# Module-level singletons — loaded once when the module is first imported.
# Streamlit's @st.cache_resource in chat_app.py ensures the graph (and
# therefore this module) is only imported once per server session.
embedding_service = EmbeddingService()
retriever = FaissRetriever(embedding_service)


@tool
def retriever_tool(query: str):
    """
    Search the knowledge base and return relevant document chunks.
    """

    log("Retriever tool invoked")

    start = time.time()

    # top_k=3: good balance — enough context, minimal embedding overhead
    docs = retriever.retrieve(query, top_k=3)

    retrieval_time = time.time() - start

    log(f"Retrieved {len(docs)} documents")
    log(f"Retrieval latency: {retrieval_time:.2f}s")

    cleaned_chunks = []
    sources = []

    for i, doc in enumerate(docs):

        # 600 chars per chunk × 3 chunks = ~1800 chars total fed to generate_answer
        text = " ".join(doc.page_content.split())[:600]

        log(f"Chunk {i+1} retrieved")

        cleaned_chunks.append(text)

        # Extract source — use metadata["source"] if available, else "Local Document"
        source = doc.metadata.get("source", "Local Document")
        if source not in sources:
            sources.append(source)

    return {
        "chunks": cleaned_chunks,
        "text": "\n\n".join(cleaned_chunks),
        "sources": sources   # list of unique source URLs/file paths
    }