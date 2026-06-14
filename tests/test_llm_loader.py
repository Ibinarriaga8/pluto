# test_llm_loader.py
"""Tests for LLMFactory (all LLM calls mocked via conftest stubs)."""

# own modules
from pluto.rag.config import LLMProvider, RAGConfig
from pluto.rag.llm_loader import LLMFactory


class TestLLMFactoryOllama:
    def test_returns_object(self):
        cfg = RAGConfig(llm_provider=LLMProvider.OLLAMA)
        llm = LLMFactory.create_llm(cfg)
        assert llm is not None

    def test_ollama_is_default_provider(self):
        cfg = RAGConfig()
        llm = LLMFactory.create_llm(cfg)
        assert llm is not None

    def test_ollama_called_with_model_name(self):
        import sys
        ollama_cls = sys.modules["langchain_ollama"].ChatOllama
        ollama_cls.reset_mock()

        cfg = RAGConfig(llm_model_name="llama3.1:8b", llm_provider=LLMProvider.OLLAMA)
        LLMFactory.create_llm(cfg)

        ollama_cls.assert_called_once()
        _, kwargs = ollama_cls.call_args
        assert kwargs["model"] == "llama3.1:8b"

    def test_ollama_called_with_temperature(self):
        import sys
        ollama_cls = sys.modules["langchain_ollama"].ChatOllama
        ollama_cls.reset_mock()

        cfg = RAGConfig(llm_temperature=0.7, llm_provider=LLMProvider.OLLAMA)
        LLMFactory.create_llm(cfg)

        _, kwargs = ollama_cls.call_args
        assert kwargs["temperature"] == 0.7

    def test_ollama_top_k_passed(self):
        import sys
        ollama_cls = sys.modules["langchain_ollama"].ChatOllama
        ollama_cls.reset_mock()

        cfg = RAGConfig(llm_top_k=10, llm_provider=LLMProvider.OLLAMA)
        LLMFactory.create_llm(cfg)

        _, kwargs = ollama_cls.call_args
        assert kwargs["top_k"] == 10

    def test_smoke_multiple_configs(self):
        for temp in [0.0, 0.3, 0.7, 1.0]:
            cfg = RAGConfig(llm_temperature=temp, llm_provider=LLMProvider.OLLAMA)
            LLMFactory.create_llm(cfg)


class TestLLMFactoryGroq:
    def test_groq_returns_object(self):
        cfg = RAGConfig(llm_provider=LLMProvider.GROQ)
        llm = LLMFactory.create_llm(cfg)
        assert llm is not None

    def test_groq_called_with_model_name(self):
        import sys
        groq_cls = sys.modules["langchain_groq"].ChatGroq
        groq_cls.reset_mock()

        cfg = RAGConfig(
            llm_model_name="llama-3.1-8b-instant",
            llm_provider=LLMProvider.GROQ,
        )
        LLMFactory.create_llm(cfg)

        groq_cls.assert_called_once()
        _, kwargs = groq_cls.call_args
        assert kwargs["model"] == "llama-3.1-8b-instant"

    def test_groq_called_with_temperature(self):
        import sys
        groq_cls = sys.modules["langchain_groq"].ChatGroq
        groq_cls.reset_mock()

        cfg = RAGConfig(llm_temperature=0.5, llm_provider=LLMProvider.GROQ)
        LLMFactory.create_llm(cfg)

        _, kwargs = groq_cls.call_args
        assert kwargs["temperature"] == 0.5
