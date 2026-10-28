import numpy as np
import pytest

from app.retrieval.hybrid import HybridRetriever


class FakeEmbeddingModel:
    vectors = {
        "python programming": [1.0, 0.0],
        "keyword search": [0.0, 1.0],
        "cooking recipe": [-1.0, 0.0],
        "python": [1.0, 0.0],
        "search": [0.0, 1.0],
    }

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        return np.array([self.vectors[text] for text in texts], dtype=np.float32)


def make_chunks() -> list[dict[str, object]]:
    return [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "filename": "python.txt",
            "page_number": None,
            "text": "python programming",
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "doc-2",
            "filename": "search.txt",
            "page_number": None,
            "text": "keyword search",
        },
        {
            "chunk_id": "chunk-3",
            "document_id": "doc-3",
            "filename": "cooking.txt",
            "page_number": None,
            "text": "cooking recipe",
        },
    ]


def test_hybrid_search_returns_fused_results() -> None:
    retriever = HybridRetriever(make_chunks(), vector_model=FakeEmbeddingModel())

    results = retriever.search("python", top_k=2, candidate_k=2, rrf_k=60)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "chunk-1"
    assert "fused_score" in results[0]


def test_hybrid_search_rejects_invalid_query_and_candidate_limit() -> None:
    retriever = HybridRetriever(make_chunks(), vector_model=FakeEmbeddingModel())

    with pytest.raises(ValueError, match="Query must not be empty"):
        retriever.search("")

    with pytest.raises(ValueError, match="candidate_k"):
        retriever.search("python", candidate_k=0)
