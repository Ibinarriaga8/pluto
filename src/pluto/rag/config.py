# config.py
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate


class LLMProvider(Enum):
    GROQ = "groq"
    OLLAMA = "ollama"

@dataclass
class RAGConfig:
    # Loader & Splitter
    urls: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    top_k: int = 4

    # Generator (LLM)
    llm_provider: LLMProvider = LLMProvider.OLLAMA
    llm_model_name: str = "llama3.1:8b"
    llm_temperature: float = 0.0
    llm_top_k: int | None = None
    num_ctx: int = 4096

    # Prompt
    prompt_hub_path: str = "rlm/rag-prompt"
    custom_prompt_template: ChatPromptTemplate | None = None

    # Embeddings & Persistence
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    persist_directory: str = "./chroma_db"
    base_path: str = "./eval_db"

    # Verbose
    verbose: bool = False
