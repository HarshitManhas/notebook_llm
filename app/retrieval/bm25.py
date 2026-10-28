import re
from typing import Any

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Convert text to lowercase tokens while preserving technical terms."""
    return re.findall(
        r"[A-Za-z0-9_]+(?:[-.][A-Za-z0-9_]+)*",
        text.lower(),
    )


class BM25Retriever:
    """Keyword retriever over document chunks."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required")

        if any(not chunk.get("text", "").strip() for chunk in chunks):
            raise ValueError("Every chunk must contain non-empty text")

        self.chunks = chunks
        self.index = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the top matching chunks with their BM25 scores."""
        if not query.strip():
            raise ValueError("Query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)
        ranked_indexes = sorted(
            range(len(self.chunks)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results: list[dict[str, Any]] = []
        for index in ranked_indexes[:top_k]:
            result = dict(self.chunks[index])
            result["score"] = float(scores[index])
            results.append(result)

        return results
