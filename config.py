import os
from dotenv import load_dotenv
from enum import Enum
from typing import Optional
from pydantic import BaseModel

load_dotenv()

class VectorStoreType(str, Enum):
    CHROMA = "chroma"
    FAISS = "faiss"

class LLMType(str, Enum):
    GPT = "gpt"
    CLAUDE = "claude"
    LLAMA = "llama"

class Config(BaseModel):
    # Telegram
    telegram_token: str
    
    # Vector Store
    vector_store_type: VectorStoreType
    chroma_persist_dir: str
    
    # LLM
    llm_type: LLMType
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    llama_model_path: Optional[str] = None
    
    # Server
    port: int = 5000

    @classmethod
    def load(cls) -> 'Config':
        config = cls(
            telegram_token=os.getenv("TELEGRAM_BOT_TOKEN"),
            vector_store_type=VectorStoreType(os.getenv("VECTOR_STORE_TYPE", "chroma").lower()),
            chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
            llm_type=LLMType(os.getenv("LLM_TYPE", "gpt").lower()),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            llama_model_path=os.getenv("LLAMA_MODEL_PATH"),
            port=int(os.getenv("PORT", "5000"))
        )
        
        # Ensure vector store directory exists with proper permissions
        os.makedirs(config.chroma_persist_dir, exist_ok=True)
        os.chmod(config.chroma_persist_dir, 0o755)  # rwxr-xr-x permissions
        
        return config

config = Config.load()
