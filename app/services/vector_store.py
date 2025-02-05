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
        try:
            # Debug logging
            print(f"Storing chunks for document: {document_id}")

        # Clear existing chunks for this document
            try:
                self.collection.delete(where={"document_id": document_id})
            except Exception as e:
                print(f"Error clearing existing chunks: {str(e)}")

        # Create embeddings for each chunk
            embeddings = [self.embeddings.embed_query(
                chunk) for chunk in chunks]

        # Store new chunks with metadata
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                self.collection.add(
                    embeddings=[embedding],
                    documents=[chunk],
                    ids=[f"{document_id}_chunk_{i}"],
                    metadatas=[{"document_id": document_id}]
                )

            print(f"Successfully stored {len(chunks)} chunks")
        except Exception as e:
            print(f"Error in store_chunks: {str(e)}")
            raise

    def get_relevant_chunks(self, question: str, document_id: str, n_results=3):
        try:
            # Debug logging
            print(f"Searching for chunks with document_id: {document_id}")

            embedding = self.embeddings.embed_query(question)

            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
                where={"document_id": document_id}
            )

        # Debug logging
            print(f"Query results: {results}")

            if not results['documents'][0]:
                return ["No relevant content found."]

            return results['documents'][0]
        except Exception as e:
            print(f"Error in get_relevant_chunks: {str(e)}")
            return ["Error retrieving relevant chunks."]
