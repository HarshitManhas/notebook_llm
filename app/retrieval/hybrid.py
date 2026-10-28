from typing import Any

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.vector_search import EmbeddingModel, VectorRetriever


class HybridRetriever:
    """Combine BM25 and semantic retrieval with reciprocal rank fusion."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        vector_model: EmbeddingModel | None = None,
        vector_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.bm25 = BM25Retriever(chunks)
        self.vector = VectorRetriever(
            chunks,
            model=vector_model,
            model_name=vector_model_name,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 30,
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        """Return deduplicated results from keyword and semantic search."""
        if not query.strip():
            raise ValueError("Query must not be empty")

        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero")

        bm25_results = self.bm25.search(query, top_k=candidate_k)
        vector_results = self.vector.search(query, top_k=candidate_k)

        return reciprocal_rank_fusion(
            [bm25_results, vector_results],
            top_k=top_k,
            rrf_k=rrf_k,
        )
