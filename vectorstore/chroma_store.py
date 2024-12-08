# vectorstore/chroma_store.py
import os
import shutil
from typing import List
import chromadb
from langchain_chroma import Chroma
from models.property import Property, PropertyMatch
from .base import PropertyVectorStore

class ChromaPropertyStore(PropertyVectorStore):
    def __init__(self, persist_directory: str = None, embedding_model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(embedding_model_name)
        # Initialize in-memory client
        self._initialize_store()

    def _initialize_store(self):
        """Initialize in-memory ChromaDB client and collection"""
        client = chromadb.Client(settings=chromadb.Settings(
            anonymized_telemetry=False,
            is_persistent=False  # Use in-memory storage
        ))
        
        self.vector_store = Chroma(
            client=client,
            collection_name="properties",
            embedding_function=self.embeddings
        )

    def clear(self) -> None:
        """Clear the vector store"""
        if self.vector_store:
            try:
                self.vector_store._collection.delete(where={})
            except:
                pass
        self._initialize_store()

    def load_properties(self, properties: List[Property]) -> None:
        """Load properties into the vector store"""
        try:
            # Clear and reinitialize
            self.clear()
            
            # Create documents
            documents = self._create_documents(properties)
            
            # Create new collection with documents
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                client=self.vector_store._client,
                collection_name="properties"
            )
            
        except Exception as e:
            raise Exception(f"Failed to load properties into ChromaDB: {str(e)}")

    def needs_loading(self) -> bool:
        """Check if the store needs to be loaded with data"""
        try:
            return self.vector_store._collection.count() == 0
        except Exception:
            return True

    def search(self, query: str, top_k: int = 5) -> List[PropertyMatch]:
        if not self.vector_store:
            return []

        try:
            results = self.vector_store.similarity_search_with_relevance_scores(
                query, k=top_k
            )
            
            matches = []
            for doc, score in results:
                property_id = doc.metadata["id"]
                if property_id in self.properties:
                    matches.append(
                        PropertyMatch(
                            property=self.properties[property_id],
                            similarity=score
                        )
                    )
            return matches
            
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return []
