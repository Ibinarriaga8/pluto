# embeders.py
import numpy as np
from sentence_transformers import SentenceTransformer


class QuestionEmbedder:
    """Encapsulates the sentence-transformer used to embed questions."""
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, questions: list[str]) -> np.ndarray:
        if len(questions) == 0:
            raise ValueError("questions must be a non-empty list.")
        embeddings = self.model.encode(
            questions,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return np.asarray(embeddings, dtype=np.float64)

