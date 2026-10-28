from typing import Any, Protocol

import numpy as np
from sentence_transformers import CrossEncoder


class CrossEncoderModel(Protocol):
    def predict(self, sentences: list[list[str]], **kwargs: Any) -> Any:
        ...


class CrossEncoderReranker:
    """Rerank retrieved chunks with a query-passage cross-encoder."""

    def __init__(
        self,
        model: CrossEncoderModel | None = None,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.model = model or CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return candidates ordered by cross-encoder relevance score."""
        if not query.strip():
            raise ValueError("Query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if any(not candidate.get("text", "").strip() for candidate in candidates):
            raise ValueError("Every candidate must contain non-empty text")

        if not candidates:
            return []

        pairs = [[query, candidate["text"]] for candidate in candidates]
        raw_scores = np.asarray(self.model.predict(pairs), dtype=np.float32)

        if raw_scores.ndim != 1 or len(raw_scores) != len(candidates):
            raise ValueError("Cross-encoder returned an invalid score array")

        ranked = []
        for candidate, score in zip(candidates, raw_scores):
            result = dict(candidate)
            result["rerank_score"] = float(score)
            ranked.append(result)

        ranked.sort(key=lambda result: result["rerank_score"], reverse=True)
        return ranked[:top_k]
