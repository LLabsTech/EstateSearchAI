from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from models.property import Property, PropertyMatch

class PropertyVectorStore(ABC):
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vector_store: Optional[VectorStore] = None
        self.properties: dict[str, Property] = {}

    @abstractmethod
    def needs_loading(self) -> bool:
        """Check if the store needs to be loaded with data"""
        pass

    @abstractmethod
    def load_properties(self, properties: List[Property]) -> None:
        """Load properties into the vector store"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear the vector store"""
        pass

    def _create_documents(self, properties: List[Property]) -> List[Document]:
        """Convert properties to LangChain documents"""
        documents = []
        for prop in properties:
            # Filter out None values and convert numbers to appropriate types
            metadata = {
                "id": prop.id,
                "price": float(prop.price),
                "type": str(prop.type),
                "town": str(prop.town),
                "beds": int(prop.beds) if prop.beds is not None else 0,
                "baths": int(prop.baths) if prop.baths is not None else 0,
                "surface_area": float(prop.surface_area_built) if prop.surface_area_built is not None else 0.0
            }

            doc = Document(
                page_content=prop.to_embedding_text(),
                metadata=metadata
            )
            documents.append(doc)
            self.properties[prop.id] = prop
        return documents
