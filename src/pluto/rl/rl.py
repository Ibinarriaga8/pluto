# rl.py
import abc
import logging
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


class Rewarder(abc.ABC):
    @abc.abstractmethod
    def __call__(self, rag_answer: str, gold_answer: str) -> float:
        pass


class DatasetRewardScorer(Rewarder):
    """Semantic similarity reward scorer using cosine similarity."""

    def __init__(self) -> None:
        logger.info("Loading sentence-transformer model for reward computation...")
        self.embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    def _cosine_similarity(self, a: str, b: str) -> float:
        emb_a = self.embedder.encode(a, convert_to_tensor=True)
        emb_b = self.embedder.encode(b, convert_to_tensor=True)
        return float(util.cos_sim(emb_a, emb_b).cpu().numpy()[0][0])

    def __call__(self, rag_answer: str, gold_answer: str) -> float:
        sim = self._cosine_similarity(rag_answer, gold_answer)
        logger.debug("Computed similarity: %.4f", sim)
        return sim


class HumanRewardScorer(Rewarder):
    """Human-in-the-loop reward scoring."""

    def __call__(self, rag_answer: str, gold_answer: str) -> float:
        print("RAG Answer:\n", rag_answer)
        print("Gold Answer:\n", gold_answer)
        while True:
            try:
                score = float(input("Please provide a reward score (0.0 to 1.0): "))
                if 0.0 <= score <= 1.0:
                    return score
                print("Score must be between 0.0 and 1.0.")
            except ValueError:
                print("Invalid input. Please enter a numeric value between 0.0 and 1.0.")


@dataclass(frozen=True)
class RagAction:
    """Action space defined by chunk_size and llm_temperature."""
    chunk_size: int
    llm_temperature: float
