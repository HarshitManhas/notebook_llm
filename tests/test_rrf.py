import pytest

from app.retrieval.rrf import reciprocal_rank_fusion


def result(chunk_id: str, score: float) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "text": chunk_id,
        "score": score,
    }


def test_fusion_rewards_results_seen_by_both_retrievers() -> None:
    results = reciprocal_rank_fusion(
        [
            [result("shared", 1.0), result("keyword-only", 0.5)],
            [result("shared", 0.9), result("vector-only", 0.4)],
        ],
        top_k=3,
        rrf_k=1,
    )

    assert [item["chunk_id"] for item in results] == [
        "shared",
        "keyword-only",
        "vector-only",
    ]
    assert results[0]["fused_score"] == pytest.approx(1.0)


def test_fusion_deduplicates_by_chunk_id() -> None:
    results = reciprocal_rank_fusion(
        [[result("same", 1.0)], [result("same", 0.5)]],
        top_k=5,
    )

    assert len(results) == 1


def test_invalid_fusion_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="top_k"):
        reciprocal_rank_fusion([], top_k=0)

    with pytest.raises(ValueError, match="rrf_k"):
        reciprocal_rank_fusion([], rrf_k=0)

    with pytest.raises(ValueError, match="chunk_id"):
        reciprocal_rank_fusion([[{"text": "missing id"}]])
