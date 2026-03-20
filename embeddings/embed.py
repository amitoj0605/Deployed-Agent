# embeddings/ollama_embeddings.py
# Switched from OllamaEmbeddings to HuggingFaceEmbeddings (sentence-transformers)
# Reasons:
# 1. Runs without Ollama — works on any cloud server
# 2. all-MiniLM-L6-v2 is fast, lightweight, and strong for semantic search
# 3. 384-dimensional vectors (smaller than nomic-embed-text's 768 — faster + less memory)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List


class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    def get_embeddings(self, docs: List[Document]) -> List[List[float]]:
        """Generate embeddings for a list of Document objects."""
        texts = [doc.page_content for doc in docs]
        return self.embeddings_model.embed_documents(texts)

    def get_query_embedding(self, query: str) -> List[float]:
        """Generate an embedding for a single query string."""
        return self.embeddings_model.embed_query(query)