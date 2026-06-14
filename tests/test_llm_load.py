# test_llm_load.py

# own modules
from pluto.rag.config import RAGConfig
from pluto.rag.llm_loader import LLMFactory


def test_llm_factory():
    """
    Test to ensure LLMFactory creates a ChatOllama instance with correct parameters.
    """
    config = RAGConfig(
        llm_model_name="llama3.1",
        llm_temperature=0.7,
        llm_top_k = 10,
    )

    llm = LLMFactory.create_llm(config)
    question = "What is the capital of France?"
    response = llm.invoke(question)
    print(f"Question: {question} \nLLM Response: {response.content}")


    assert llm.model == "llama3.1", "LLM model name mismatch"
    assert llm.temperature == 0.7, "LLM temperature mismatch"
