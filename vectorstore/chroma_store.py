import os
import shutil
import stat
import platform
from typing import List
import chromadb
from langchain_chroma import Chroma

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

    def _set_directory_permissions(self, directory: str):
        """Set appropriate permissions based on the operating system"""
        try:
            if platform.system() == "Windows":
                return
            
            # For Unix-like systems (Linux, macOS)
            os.chmod(directory, stat.S_IRWXU | stat.S_IRWXG)
            
            for root, dirs, files in os.walk(directory):
                for d in dirs:
                    os.chmod(os.path.join(root, d), stat.S_IRWXU | stat.S_IRWXG)
                for f in files:
                    os.chmod(os.path.join(root, f), stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP)
                    
        except Exception as e:
            print(f"Warning: Could not set permissions for {directory}: {e}")

    def _setup_storage(self):
        """Set up storage with correct permissions"""
        try:
            # Create main directory if it doesn't exist
            if not os.path.exists(self.chroma_data_directory):
                os.makedirs(self.persist_directory, exist_ok=True)
                os.makedirs(self.chroma_data_directory)
                self._set_directory_permissions(self.persist_directory)
                self._set_directory_permissions(self.chroma_data_directory)
                print(f"Set up ChromaDB directory structure in: {self.chroma_data_directory}")
        except Exception as e:
            raise Exception(f"Failed to setup ChromaDB storage: {str(e)}")

    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""
        try:
            if self.storage_mode == StorageMode.DISK:
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
                client = chromadb.Client(settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    is_persistent=False
                ))

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
                os.makedirs(self.chroma_data_directory)
                self._set_directory_permissions(self.chroma_data_directory)
            except Exception as e:
                print(f"Error clearing storage: {e}")
                raise

        self.vector_store = None
        self._initialize_store()

    def needs_loading(self) -> bool:
        """Check if the store needs to be loaded with data"""
        try:
            if self.storage_mode == StorageMode.DISK:
                if not os.path.exists(self.chroma_data_directory):
                    return True
                    
                # Check if database file exists
                db_file = os.path.join(self.chroma_data_directory, 'chroma.sqlite3')
                return not os.path.exists(db_file)
            
            # For in-memory store, we need loading if no store exists
            return self.vector_store is None
            
        except Exception:
            return True

    def load_properties(self, properties: List[Property]) -> None:
        """Load properties into the vector store"""
        try:
            # Setup storage directories first
            if self.storage_mode == StorageMode.DISK:
                self._setup_storage()
            
            # Create documents
            documents = self._create_documents(properties)
            
            # Initialize fresh store
            self._initialize_store()
            
            # Create new collection with documents
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
