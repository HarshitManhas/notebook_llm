import numpy as np
import pytest

from app.reranking.cross_encoder import CrossEncoderReranker


class FakeCrossEncoder:
    def predict(self, sentences: list[list[str]], **_: object) -> np.ndarray:
        scores = {
            "python passage": 0.2,
            "relevant passage": 0.95,
            "weak passage": 0.1,
        }
        return np.array([scores[pair[1]] for pair in sentences], dtype=np.float32)


def make_candidates() -> list[dict[str, object]]:
    return [
        {"chunk_id": "chunk-1", "text": "python passage"},
        {"chunk_id": "chunk-2", "text": "relevant passage"},
        {"chunk_id": "chunk-3", "text": "weak passage"},
    ]


def test_rerank_orders_candidates_by_relevance() -> None:
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())

    results = reranker.rerank("question", make_candidates(), top_k=2)

    assert [result["chunk_id"] for result in results] == [
        "chunk-2",
        "chunk-1",
    ]
    assert results[0]["rerank_score"] == pytest.approx(0.95)


def test_rerank_preserves_candidate_metadata() -> None:
    candidates = [{"chunk_id": "chunk-1", "filename": "source.pdf", "text": "python passage"}]
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())

    results = reranker.rerank("question", candidates)

    assert results[0]["filename"] == "source.pdf"


def test_empty_candidates_return_empty_results() -> None:
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())

    assert reranker.rerank("question", []) == []


def test_invalid_inputs_are_rejected() -> None:
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())

    with pytest.raises(ValueError, match="Query must not be empty"):
        reranker.rerank("", make_candidates())

    with pytest.raises(ValueError, match="top_k"):
        reranker.rerank("question", make_candidates(), top_k=0)

    with pytest.raises(ValueError, match="non-empty"):
        reranker.rerank("question", [{"chunk_id": "empty", "text": ""}])


def test_invalid_score_shape_is_rejected() -> None:
    class InvalidCrossEncoder:
        def predict(self, sentences: list[list[str]], **_: object) -> np.ndarray:
            return np.array([[1.0]], dtype=np.float32)

    reranker = CrossEncoderReranker(model=InvalidCrossEncoder())

    with pytest.raises(ValueError, match="invalid score array"):
        reranker.rerank("question", make_candidates())
