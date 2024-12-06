from abc import ABC, abstractmethod
from typing import List, Optional
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores.base import VectorStore
from langchain_core.documents import Document
from models.property import Property, PropertyMatch

class PropertyVectorStore(ABC):
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)
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
            doc = Document(
                page_content=prop.to_embedding_text(),
                metadata={
                    "id": prop.id,
                    "price": prop.price,
                    "type": prop.type,
                    "town": prop.town,
                    "beds": prop.beds,
                    "baths": prop.baths
                }
            )
            documents.append(doc)
            self.properties[prop.id] = prop
        return documents
