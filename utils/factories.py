import os
import shutil
from config import Config, LLMType, VectorStoreType, StorageMode
from llm.base import LLMHandler
from llm.openai_handler import OpenAIHandler
from llm.claude_handler import ClaudeHandler
from llm.llama_handler import LlamaHandler
from vectorstore.base import PropertyVectorStore
from vectorstore.chroma_store import ChromaPropertyStore
from vectorstore.faiss_store import FAISSPropertyStore

def verify_storage_directory(config: Config) -> None:
    """Verify storage directory exists"""
    if config.storage_mode == StorageMode.DISK:
        persist_dir = os.path.abspath(config.chroma_persist_dir)
        
        # Only create and log if directory doesn't exist
        if not os.path.exists(persist_dir):
            print(f"Setting up persistent storage in: {persist_dir}")
            os.makedirs(persist_dir)
            os.chmod(persist_dir, 0o777)
            print(f"Initialized storage directory: {persist_dir}")

def create_vector_store(config: Config) -> PropertyVectorStore:
    """Create appropriate vector store based on configuration"""
    # Just verify directory exists if needed
    verify_storage_directory(config)
    
    if config.vector_store_type == VectorStoreType.CHROMA:
        return ChromaPropertyStore(
            persist_directory=config.chroma_persist_dir,
            storage_mode=config.storage_mode
        )
    
    elif config.vector_store_type == VectorStoreType.FAISS:
        return FAISSPropertyStore(config.chroma_persist_dir)
    
    raise ValueError(f"Unsupported vector store type: {config.vector_store_type}")

def create_llm_handler(config: Config) -> LLMHandler:
    """Create appropriate LLM handler based on configuration"""
    if config.llm_type == LLMType.GPT:
        if not config.openai_api_key:
            raise ValueError("OpenAI API key is required for GPT")
        return OpenAIHandler(config.openai_api_key)
    
    elif config.llm_type == LLMType.CLAUDE:
        if not config.anthropic_api_key:
            raise ValueError("Anthropic API key is required for Claude")
        return ClaudeHandler(config.anthropic_api_key)
    
    elif config.llm_type == LLMType.LLAMA:
        if not config.llama_model_path:
            raise ValueError("Llama model path is required")
        return LlamaHandler(config.llama_model_path)
    
    raise ValueError(f"Unsupported LLM type: {config.llm_type}")
