from typing import List
from langchain.vectorstores import Chroma
from models.property import Property, PropertyMatch
from .base import PropertyVectorStore

class ChromaPropertyStore(PropertyVectorStore):
    def __init__(self, persist_directory: str, embedding_model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(embedding_model_name)
        self.persist_directory = persist_directory

    def load_properties(self, properties: List[Property]) -> None:
        documents = self._create_documents(properties)
        
        # Create and persist Chroma vector store
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
