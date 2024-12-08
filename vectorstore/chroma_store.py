from typing import List
import shutil
from langchain_community.vectorstores import Chroma
from models.property import Property, PropertyMatch
from .base import PropertyVectorStore

class ChromaPropertyStore(PropertyVectorStore):
    def __init__(self, persist_directory: str, embedding_model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(embedding_model_name)
        self.persist_directory = persist_directory
        self._initialize_store()

    def _initialize_store(self):
        """Initialize or load existing Chroma store"""
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    def needs_loading(self) -> bool:
        """Check if the store needs to be loaded with data"""
        try:
            count = self.vector_store._collection.count()
            return count == 0
        except Exception:
            return True

    def clear(self) -> None:
        """Clear the vector store"""
        if self.vector_store:
            self.vector_store = None
        if self.persist_directory:
            shutil.rmtree(self.persist_directory, ignore_errors=True)
        self._initialize_store()

    def load_properties(self, properties: List[Property]) -> None:
        """Load properties into the vector store"""
        # Clear existing data
        self.clear()
        
        # Create and store new documents
        documents = self._create_documents(properties)
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.vector_store.persist()

    def search(self, query: str, top_k: int = 5) -> List[PropertyMatch]:
        if not self.vector_store:
            return []

        # Search documents
        results = self.vector_store.similarity_search_with_relevance_scores(query, k=top_k)
        
        # Convert to PropertyMatch objects
        matches = []
        for doc, score in results:
            property_id = doc.metadata["id"]
            if property_id in self.properties:
                matches.append(PropertyMatch(
                    property=self.properties[property_id],
                    similarity=score
                ))
        
        return matches
