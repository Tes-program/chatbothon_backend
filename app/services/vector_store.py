# app/services/vector_store.py
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
import chromadb


class VectorStore:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        self.collection = self._setup_collection()

    def _setup_collection(self):
        try:
            return self.chroma_client.create_collection(
                name="document_chunks",
                metadata={"hnsw:space": "cosine"}
            )
        except chromadb.db.base.UniqueConstraintError:
            return self.chroma_client.get_collection(
                name="document_chunks"
            )

    def store_chunks(self, chunks: List[str], document_id: str):
        embeddings = [self.embeddings.embed_query(chunk) for chunk in chunks]

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            self.collection.add(
                embeddings=[embedding],
                documents=[chunk],
                ids=[f"{document_id}_chunk_{i}"]
            )

    def get_relevant_chunks(self, question: str, n_results=3):
        embedding = self.embeddings.embed_query(question)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )
        return results['documents'][0]
