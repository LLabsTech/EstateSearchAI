import os
import shutil
from typing import List, Dict, Any
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

from .base import PropertyVectorStore
from models.property import Property, PropertyMatch
from config import StorageMode

class ChromaPropertyStore(PropertyVectorStore):
    def __init__(self, persist_directory: str = None, storage_mode: StorageMode = StorageMode.MEMORY, embedding_model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(embedding_model_name)
        
        if storage_mode == StorageMode.DISK and persist_directory:
            self.persist_directory = os.path.abspath(persist_directory)
            self.chroma_data_directory = os.path.join(self.persist_directory, '.chroma')
        else:
            self.persist_directory = None
            self.chroma_data_directory = None
            
        self.storage_mode = storage_mode
        self.vector_store = None

    def _process_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process metadata to ensure it's compatible with ChromaDB"""
        processed = {}
        for key, value in metadata.items():
            if value is None:
                if key in ['beds', 'baths']:
                    processed[key] = '0'
                elif key in ['surface_area_built', 'surface_area_plot', 'price']:
                    processed[key] = '0.0'
                else:
                    processed[key] = ''
            elif isinstance(value, list):
                processed[key] = ', '.join(map(str, value)) if value else ''
            elif isinstance(value, (int, float)):
                processed[key] = str(value)
            elif isinstance(value, bool):
                processed[key] = str(value).lower()
            else:
                processed[key] = str(value)
        
        return processed

    def _create_documents(self, properties: List[Property]) -> List[Document]:
        """Create Langchain documents with complete property data in metadata"""
        documents = []
        for prop in properties:
            description = prop.to_embedding_text()
            metadata = self._process_metadata(prop.dict())
            doc = Document(
                page_content=description,
                metadata=metadata
            )
            documents.append(doc)
            self.properties[str(prop.id)] = prop
        return documents

    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""
        try:
            if self.storage_mode == StorageMode.DISK:
                # Create directory if it doesn't exist
                os.makedirs(self.chroma_data_directory, exist_ok=True)
                
                settings = chromadb.Settings(
                    is_persistent=True,
                    persist_directory=self.chroma_data_directory,
                    anonymized_telemetry=False
                )
                
                client = chromadb.PersistentClient(
                    path=self.chroma_data_directory,
                    settings=settings
                )
            else:
                client = chromadb.Client()

            self.vector_store = Chroma(
                client=client,
                collection_name="properties",
                embedding_function=self.embeddings,
                persist_directory=self.chroma_data_directory if self.storage_mode == StorageMode.DISK else None
            )

        except Exception as e:
            raise Exception(f"Failed to initialize ChromaDB: {str(e)}")

    def clear(self) -> None:
        """Clear the vector store"""
        if self.storage_mode == StorageMode.DISK and self.chroma_data_directory:
            try:
                if os.path.exists(self.chroma_data_directory):
                    shutil.rmtree(self.chroma_data_directory)
                os.makedirs(self.chroma_data_directory, exist_ok=True)
            except Exception as e:
                print(f"Error clearing storage: {e}")
                raise

        self.vector_store = None
        self.properties = {}
        self._initialize_store()

    def needs_loading(self) -> bool:
        """Check if the store needs to be loaded with data"""
        try:
            if self.storage_mode == StorageMode.DISK:
                if not os.path.exists(self.chroma_data_directory):
                    return True
                    
                db_file = os.path.join(self.chroma_data_directory, 'chroma.sqlite3')
                return not os.path.exists(db_file)
            
            return True
            
        except Exception:
            return True

    def load_properties(self, properties: List[Property]) -> None:
        """Load properties into the vector store"""
        try:
            # Create documents with full property data in metadata
            documents = self._create_documents(properties)
            
            # Initialize fresh store
            self._initialize_store()
            
            # Create collection with documents
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                client=self.vector_store._client,
                collection_name="properties",
                persist_directory=self.chroma_data_directory if self.storage_mode == StorageMode.DISK else None
            )
            
            print(f"Successfully stored {len(properties)} properties in vector store")
                
        except Exception as e:
            print(f"Detailed error during property loading: {str(e)}")
            raise Exception(f"Failed to load properties into ChromaDB: {str(e)}")

    def search(self, query: str, top_k: int = 5) -> List[PropertyMatch]:
        """Search for properties matching the query"""
        if not self.vector_store:
            try:
                self._initialize_store()
            except Exception as e:
                print(f"Error initializing store during search: {e}")
                return []

        try:
            results = self.vector_store.similarity_search_with_relevance_scores(
                query, k=top_k
            )
            
            matches = []
            for doc, score in results:
                # Convert stored metadata back to property structure
                property_data = {}
                for key, value in doc.metadata.items():
                    if key == 'features':
                        property_data[key] = value.split(', ') if value else []
                    elif key in ['beds', 'baths']:
                        property_data[key] = int(value) if value else None
                    elif key in ['surface_area_built', 'surface_area_plot', 'price']:
                        property_data[key] = float(value) if value else None
                    elif value == "":
                        property_data[key] = None
                    elif value.lower() in ('true', 'false'):
                        property_data[key] = value.lower() == 'true'
                    else:
                        property_data[key] = value
                
                property_obj = Property(**property_data)
                matches.append(
                    PropertyMatch(
                        property=property_obj,
                        similarity=score
                    )
                )
            return matches
            
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return []
