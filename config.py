import os
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class VectorStoreType(str, Enum):
    CHROMA = "chroma"
    FAISS = "faiss"

class LLMType(str, Enum):
    GPT = "gpt"
    CLAUDE = "claude"
    LLAMA = "llama"

class StorageMode(str, Enum):
    MEMORY = "memory"
    DISK = "disk"

class Config(BaseModel):
    # Telegram
    telegram_token: str = Field(default="", description="Telegram Bot Token")
    
    # Vector Store
    vector_store_type: VectorStoreType = Field(default=VectorStoreType.CHROMA)
    storage_mode: StorageMode = Field(default=StorageMode.MEMORY)
    chroma_persist_dir: str = Field(default="./chroma_db")
    
    # LLM
    llm_type: LLMType = Field(default=LLMType.GPT)
    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)
    llama_model_path: Optional[str] = Field(default=None)
    
    # Server
    port: int = Field(default=5000)

    @classmethod
    def load(cls) -> 'Config':
        # Get environment variables with defaults
        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        if not telegram_token:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")
            
        vector_store_type = os.getenv("VECTOR_STORE_TYPE", "chroma").lower()
        storage_mode = os.getenv("STORAGE_MODE", "memory").lower()
        chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        llm_type = os.getenv("LLM_TYPE", "gpt").lower()
        port = int(os.getenv("PORT", "5000"))
        
        config = cls(
            telegram_token=telegram_token,
            vector_store_type=VectorStoreType(vector_store_type),
            storage_mode=StorageMode(storage_mode),
            chroma_persist_dir=chroma_persist_dir,
            llm_type=LLMType(llm_type),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            llama_model_path=os.getenv("LLAMA_MODEL_PATH"),
            port=port
        )
        
        # Ensure vector store directory exists with proper permissions if using disk storage
        if config.storage_mode == StorageMode.DISK:
            os.makedirs(config.chroma_persist_dir, exist_ok=True)
            os.chmod(config.chroma_persist_dir, 0o755)  # rwxr-xr-x permissions
        
        return config

config = Config.load()
