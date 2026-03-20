# vectorstore/index_builder.py
# IMPORTANT: dimension changed from 768 (nomic-embed-text) to 384 (all-MiniLM-L6-v2)
# You MUST rebuild the FAISS index after switching embedding models

import faiss
import pickle
import numpy as np
from embeddings.embed import EmbeddingService
from langchain_core.documents import Document
from typing import List


class FaissIndexBuilder:
    def __init__(self, embedding_service: EmbeddingService, dimension: int = 384):
        self.embedding_service = embedding_service
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.doc_mapping = []

    def build_index(self, docs: List[Document]):
        print(f"Generating embeddings for {len(docs)} chunks...")
        embeddings = self.embedding_service.get_embeddings(docs)
        embeddings_np = np.array(embeddings).astype('float32')
        self.index.add(embeddings_np)
        self.doc_mapping.extend(docs)
        print(f"Added {len(docs)} vectors to FAISS index.")

    def save_index(self, index_path: str = "vectorstore/faiss.index",
                   mapping_path: str = "vectorstore/doc_mapping.pkl"):
        faiss.write_index(self.index, index_path)
        with open(mapping_path, "wb") as f:
            pickle.dump(self.doc_mapping, f)
        print(f"FAISS index saved to {index_path}")