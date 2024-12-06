from config import Config, LLMType, VectorStoreType
from llm.base import LLMHandler
from llm.openai_handler import OpenAIHandler
from llm.claude_handler import ClaudeHandler
from llm.llama_handler import LlamaHandler
from vectorstore.base import PropertyVectorStore
from vectorstore.chroma_store import ChromaPropertyStore
from vectorstore.faiss_store import FAISSPropertyStore

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

def create_vector_store(config: Config) -> PropertyVectorStore:
    """Create appropriate vector store based on configuration"""
    if config.vector_store_type == VectorStoreType.CHROMA:
        return ChromaPropertyStore(config.chroma_persist_dir)
    
    elif config.vector_store_type == VectorStoreType.FAISS:
        return FAISSPropertyStore(config.chroma_persist_dir)
    
    raise ValueError(f"Unsupported vector store type: {config.vector_store_type}")
