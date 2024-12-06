from typing import List
import os
from langchain.vectorstores import FAISS
from models.property import Property, PropertyMatch
from .base import PropertyVectorStore

class FAISSPropertyStore(PropertyVectorStore):
    def __init__(self, persist_directory: str, embedding_model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(embedding_model_name)
        self.persist_directory = persist_directory
        self.index_file = os.path.join(persist_directory, "index.faiss")
        self.store_file = os.path.join(persist_directory, "store.pkl")

    def load_properties(self, properties: List[Property]) -> None:
        documents = self._create_documents(properties)
        
        # Create FAISS vector store
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Save index
        os.makedirs(self.persist_directory, exist_ok=True)
        self.vector_store.save_local(self.persist_directory)

    def search(self, query: str, top_k: int = 5) -> List[PropertyMatch]:
        if not self.vector_store:
            return []

        # Search documents
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        
        # Convert to PropertyMatch objects
        matches = []
        for doc, score in results:
            property_id = doc.metadata["id"]
            if property_id in self.properties:
                # Convert FAISS distance to similarity score (FAISS returns L2 distance)
                similarity = 1 / (1 + score)
                matches.append(PropertyMatch(
                    property=self.properties[property_id],
                    similarity=similarity
                ))
        
        return matches

    @classmethod
    def load_local(cls, persist_directory: str, embedding_model_name: str = "all-MiniLM-L6-v2") -> 'FAISSPropertyStore':
        """Load existing FAISS index"""
        instance = cls(persist_directory, embedding_model_name)
        if os.path.exists(instance.index_file):
            instance.vector_store = FAISS.load_local(
                persist_directory,
                instance.embeddings
            )
        return instance
