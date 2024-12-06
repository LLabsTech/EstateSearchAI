import os
import shutil
from typing import List
from langchain.vectorstores import FAISS
from models.property import Property, PropertyMatch
from .base import PropertyVectorStore

class FAISSPropertyStore(PropertyVectorStore):
    def __init__(self, persist_directory: str, embedding_model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(embedding_model_name)
        self.persist_directory = persist_directory
        self.index_file = os.path.join(persist_directory, "index.faiss")
        self.store_file = os.path.join(persist_directory, "store.pkl")
        self._initialize_store()

    def _initialize_store(self):
        """Initialize or load existing FAISS store"""
        if os.path.exists(self.index_file) and os.path.exists(self.store_file):
            self.vector_store = FAISS.load_local(
                self.persist_directory,
                self.embeddings
            )

    def needs_loading(self) -> bool:
        """Check if the store needs to be loaded with data"""
        return not (os.path.exists(self.index_file) and
                   os.path.exists(self.store_file))

    def clear(self) -> None:
        """Clear the vector store"""
        if self.vector_store:
            self.vector_store = None
        if self.persist_directory:
            shutil.rmtree(self.persist_directory, ignore_errors=True)
        os.makedirs(self.persist_directory, exist_ok=True)

    def load_properties(self, properties: List[Property]) -> None:
        """Load properties into the vector store"""
        # Clear existing data
        self.clear()
        
        # Create and store new documents
        documents = self._create_documents(properties)
        self.vector_store = FAISS.from_documents(
            documents,
            self.embeddings
        )
        
        # Save to disk
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
