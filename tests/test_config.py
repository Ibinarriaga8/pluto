# test_config.py
"""Tests for RAGConfig and LLMProvider."""
import pytest

# own modules
from pluto.rag.config import LLMProvider, RAGConfig


class TestLLMProvider:
    def test_enum_values(self):
        assert LLMProvider.OLLAMA.value == "ollama"
        assert LLMProvider.GROQ.value == "groq"

    def test_enum_members(self):
        members = {p.name for p in LLMProvider}
        assert "OLLAMA" in members
        assert "GROQ" in members

    def test_enum_from_value(self):
        assert LLMProvider("ollama") is LLMProvider.OLLAMA
        assert LLMProvider("groq") is LLMProvider.GROQ

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            LLMProvider("unknown_provider")


class TestRAGConfigDefaults:
    def test_default_instantiation(self):
        cfg = RAGConfig()
        assert cfg is not None

    def test_default_chunk_size(self):
        assert RAGConfig().chunk_size == 1000

    def test_default_chunk_overlap(self):
        assert RAGConfig().chunk_overlap == 200

    def test_default_top_k(self):
        assert RAGConfig().top_k == 4

    def test_default_temperature(self):
        assert RAGConfig().llm_temperature == 0.0

    def test_default_provider_is_ollama(self):
        assert RAGConfig().llm_provider == LLMProvider.OLLAMA

    def test_default_model_name(self):
        assert RAGConfig().llm_model_name == "llama3.1:8b"

    def test_default_texts_is_empty_list(self):
        cfg = RAGConfig()
        assert cfg.texts == []

    def test_default_texts_are_independent(self):
        """Two default instances must not share the same list object."""
        a = RAGConfig()
        b = RAGConfig()
        assert a.texts is not b.texts

    def test_default_embedding_model(self):
        assert "mpnet" in RAGConfig().embedding_model_name

    def test_default_verbose_is_false(self):
        assert RAGConfig().verbose is False

    def test_default_num_ctx(self):
        assert RAGConfig().num_ctx == 4096

    def test_default_llm_top_k_is_none(self):
        assert RAGConfig().llm_top_k is None

    def test_default_custom_prompt_is_none(self):
        assert RAGConfig().custom_prompt_template is None


class TestRAGConfigCustomValues:
    def test_custom_chunk_size(self):
        cfg = RAGConfig(chunk_size=512)
        assert cfg.chunk_size == 512

    def test_custom_temperature(self):
        cfg = RAGConfig(llm_temperature=0.7)
        assert cfg.llm_temperature == 0.7

    def test_custom_provider_groq(self):
        cfg = RAGConfig(llm_provider=LLMProvider.GROQ)
        assert cfg.llm_provider == LLMProvider.GROQ

    def test_custom_texts(self):
        texts = ["Hello world", "Foo bar"]
        cfg = RAGConfig(texts=texts)
        assert cfg.texts == texts

    def test_custom_top_k(self):
        cfg = RAGConfig(top_k=10)
        assert cfg.top_k == 10

    def test_custom_verbose(self):
        cfg = RAGConfig(verbose=True)
        assert cfg.verbose is True

    def test_mutate_chunk_size(self):
        cfg = RAGConfig(chunk_size=100)
        cfg.chunk_size = 200
        assert cfg.chunk_size == 200

    def test_mutate_temperature(self):
        cfg = RAGConfig(llm_temperature=0.0)
        cfg.llm_temperature = 1.0
        assert cfg.llm_temperature == 1.0

    def test_copy_via_dict(self):
        original = RAGConfig(chunk_size=300, llm_temperature=0.5)
        copy = type(original)(**original.__dict__)
        assert copy.chunk_size == original.chunk_size
        assert copy.llm_temperature == original.llm_temperature
        assert copy is not original

    @pytest.mark.parametrize("chunk_size", [50, 100, 300, 500, 1000, 2000])
    def test_valid_chunk_sizes(self, chunk_size):
        cfg = RAGConfig(chunk_size=chunk_size)
        assert cfg.chunk_size == chunk_size

    @pytest.mark.parametrize("temp", [0.0, 0.1, 0.5, 1.0])
    def test_valid_temperatures(self, temp):
        cfg = RAGConfig(llm_temperature=temp)
        assert cfg.llm_temperature == temp
