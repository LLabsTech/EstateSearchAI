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

    def _clean_directory(self, directory: str) -> None:
        """Safely remove and recreate a directory"""
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)

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
            elif isinstance(value, dict):
                # For nested dictionaries like description
                if key == 'desc' and 'es' in value:
                    processed[key] = value['es']
                else:
                    processed[key] = str(value)
            else:
                processed[key] = str(value)
        
        return processed

    def _create_documents(self, properties: List[Property]) -> List[Document]:
        """Create Langchain documents with property data"""
        documents = []
        for prop in properties:
            # Use the property's embedding text method
            text = prop.to_embedding_text()
            
            # Process metadata to ensure compatibility
            metadata = self._process_metadata(prop.dict())
            
            # Create document
            doc = Document(
                page_content=text,
                metadata=metadata
            )
            documents.append(doc)
            
            # Store reference to property
            self.properties[str(prop.id)] = prop
            
        return documents

    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""
        if self.storage_mode == StorageMode.DISK:
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

    def clear(self) -> None:
        """Clear the vector store"""
        if self.storage_mode == StorageMode.DISK and self.chroma_data_directory:
            self._clean_directory(self.chroma_data_directory)

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
        if not properties:
            raise ValueError("No properties provided to load")
            
        try:
            # Create documents and gather data for ChromaDB
            documents = self._create_documents(properties)
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            ids = [str(i) for i in range(len(documents))]
            
            # Initialize fresh store
            self._initialize_store()
            
            # Add texts to ChromaDB
            if texts:  # Only try to add if we have texts
                self.vector_store.add_texts(
                    texts=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"Successfully stored {len(properties)} properties in vector store")
            else:
                raise ValueError("No valid texts to store in vector database")
                
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
                # Reconstruct property data from metadata
                property_data = {}
                for key, value in doc.metadata.items():
                    if key == 'features':
                        property_data[key] = value.split(', ') if value else []
                    elif key == 'desc':
                        property_data[key] = {'es': value} if value else {'es': ''}
                    elif key == 'images':
                        property_data[key] = [{'url': url.strip()} for url in value.split(', ')] if value else []
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