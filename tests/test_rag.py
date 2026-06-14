# test_rag.py
import os
import shutil

import pytest

# own modules
from pluto.rag.config import LLMProvider, RAGConfig
from pluto.rag.rag_interface import ChromaRAGInterface, InMemoryRAGInterface


# Define the testing matrix: (Provider, Model)
@pytest.fixture(params=[
    (LLMProvider.OLLAMA, "llama3.1:8b"),
    (LLMProvider.GROQ, "llama-3.1-8b-instant")
])
def provider_setup(request):
    """Fixture to provide LLM provider and model name."""
    provider, model = request.param

    # Professional check: Skip Groq tests if no API key is found
    if provider == LLMProvider.GROQ and not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not found in environment. Skipping Groq tests.")

    return provider, model

# Define the storage matrix: (Interface Class, Label)
@pytest.fixture(params=[
    (InMemoryRAGInterface, "in_memory"),
    (ChromaRAGInterface, "chroma")
])
def storage_setup(request):
    """Fixture to provide the Interface class and its string label."""
    return request.param

def test_rag_interface(provider_setup, storage_setup):
    """
    Integration test using a real Wikipedia article.
    Tests (Ollama/Groq) x (In-Memory/Chroma)
    """
    llm_provider, llm_model = provider_setup
    interface_class, storage_label = storage_setup

    # Use base_path so _sync_persist_directory() routes to an isolated test root
    # (setting persist_directory would be overridden by _sync_persist_directory)
    test_base = f"./test_db_{llm_provider.value}_{storage_label}"
    shutil.rmtree(test_base, ignore_errors=True)

    test_config = RAGConfig(
        urls=["https://en.wikipedia.org/wiki/History_of_Spain"],
        llm_provider=llm_provider,
        llm_model_name=llm_model,
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        base_path=test_base,
    )

    print("\n--- Testing Wikipedia RAG ---")
    print(f"Provider: {llm_provider.name} | Storage: {storage_label}")

    # Initialize the chosen interface
    interface = interface_class(test_config)

    # Execution & Verification
    try:
        query = "Who were the Visigoths in the context of Spanish history?"
        print(f"Querying: {query}")

        response = interface.ask(query)

        assert response is not None

        print(f"Response Preview: {response[:150]}...")

    except Exception as e:
        pytest.fail(f"RAG Matrix Failure [{llm_provider.name} | {storage_label}]: {e}")

    finally:
        shutil.rmtree(test_base, ignore_errors=True)
