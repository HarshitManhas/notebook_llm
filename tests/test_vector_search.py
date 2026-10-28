import numpy as np
import pytest

from app.retrieval.vector_search import VectorRetriever


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


def test_search_returns_semantically_similar_chunk() -> None:
    retriever = VectorRetriever(make_chunks(), model=FakeEmbeddingModel())

    results = retriever.search("python", top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk-1"
    assert results[0]["score"] == pytest.approx(1.0)


def test_search_preserves_metadata_and_limits_results() -> None:
    retriever = VectorRetriever(make_chunks(), model=FakeEmbeddingModel())

    results = retriever.search("search", top_k=2)

    assert len(results) == 2
    assert results[0]["filename"] == "search.txt"
    assert results[0]["score"] >= results[1]["score"]


def test_empty_chunks_are_rejected() -> None:
    with pytest.raises(ValueError, match="At least one chunk"):
        VectorRetriever([])


def test_empty_query_is_rejected() -> None:
    retriever = VectorRetriever(make_chunks(), model=FakeEmbeddingModel())

    with pytest.raises(ValueError, match="Query must not be empty"):
        retriever.search("")


def test_zero_embeddings_are_rejected() -> None:
    class ZeroEmbeddingModel:
        def encode(self, texts: list[str], **_: object) -> np.ndarray:
            return np.zeros((len(texts), 2), dtype=np.float32)

    with pytest.raises(ValueError, match="zero vector"):
        VectorRetriever(make_chunks(), model=ZeroEmbeddingModel())
