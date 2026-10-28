from typing import Any, Protocol

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel(Protocol):
    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        ...


def _as_normalized_embeddings(embeddings: Any) -> np.ndarray:
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError("Embedding model must return a 2D array of vectors")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding model returned a zero vector")

    return vectors / norms


class VectorRetriever:
    """Semantic retriever using normalized embeddings and FAISS cosine search."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        model: EmbeddingModel | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required")

        if any(not chunk.get("text", "").strip() for chunk in chunks):
            raise ValueError("Every chunk must contain non-empty text")

        self.chunks = chunks
        self.model = model or SentenceTransformer(model_name)
        embeddings = self._encode([chunk["text"] for chunk in chunks])
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def _encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return _as_normalized_embeddings(embeddings)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the most semantically similar chunks with cosine scores."""
        if not query.strip():
            raise ValueError("Query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_embedding = self._encode([query])
        result_count = min(top_k, len(self.chunks))
        scores, indexes = self.index.search(query_embedding, result_count)

        results: list[dict[str, Any]] = []
        for score, index in zip(scores[0], indexes[0]):
            if index < 0:
                continue

            result = dict(self.chunks[int(index)])
            result["score"] = float(score)
            results.append(result)

        return results
