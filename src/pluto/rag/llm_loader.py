# llm_loader.py
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

# own modules
from pluto.rag.config import LLMProvider, RAGConfig


class LLMFactory:
    """
    Factory class to instantiate LLMs based on configuration.
    """

    @staticmethod
    def create_llm(config: RAGConfig) -> BaseChatModel:
        provider = config.llm_provider
        if isinstance(provider, str):
            provider = LLMProvider(provider)

        if provider == LLMProvider.GROQ:
            return ChatGroq(
                model=config.llm_model_name,
                temperature=config.llm_temperature,
            )

        return ChatOllama(
            model=config.llm_model_name,
            temperature=config.llm_temperature,
            top_k=config.llm_top_k,
            num_ctx=getattr(config, "num_ctx", 4096),
        )
