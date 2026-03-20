# vectorstore/faiss_store.py

import faiss
import pickle
import numpy as np
from typing import List
from langchain_core.documents import Document


class FaissRetriever:
    """
    Load FAISS index and perform semantic search for a query embedding.
    Supports appending new documents without full rebuild.
    """

    def __init__(self, embedding_service,
                 index_path: str = "vectorstore/faiss.index",
                 mapping_path: str = "vectorstore/doc_mapping.pkl"):
        self.embedding_service = embedding_service
        self.index_path = index_path
        self.mapping_path = mapping_path

        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load doc mapping
        with open(mapping_path, "rb") as f:
            self.doc_mapping: List[Document] = pickle.load(f)

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve top-k most relevant document chunks for a query."""
        query_embedding = self.embedding_service.get_query_embedding(query)
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_vector, top_k)
        results = [self.doc_mapping[idx] for idx in indices[0]
                   if idx < len(self.doc_mapping)]
        return results

    def add_documents(self, docs: List[Document]):
        """
        Append new document chunks to the existing FAISS index.
        Does NOT rebuild — preserves all existing vectors.
        Saves updated index and mapping back to disk immediately.
        """
        if not docs:
            return 0

        # Embed new chunks
        embeddings = self.embedding_service.get_embeddings(docs)
        embeddings_np = np.array(embeddings).astype('float32')

        # Append to existing index
        self.index.add(embeddings_np)
        self.doc_mapping.extend(docs)

        # Save updated index and mapping to disk
        faiss.write_index(self.index, self.index_path)
        with open(self.mapping_path, "wb") as f:
            pickle.dump(self.doc_mapping, f)

        return len(docs)

    @property
    def total_chunks(self):
        """Total number of chunks currently in the index."""
        return self.index.ntotal