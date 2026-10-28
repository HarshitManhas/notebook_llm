import pytest

from app.evaluation.metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)


RANKED_IDS = ["chunk-3", "chunk-1", "chunk-4", "chunk-2"]
RELEVANT_IDS = {"chunk-1", "chunk-2"}


def test_recall_at_k() -> None:
    assert recall_at_k(RANKED_IDS, RELEVANT_IDS, 3) == pytest.approx(0.5)


def test_precision_at_k() -> None:
    assert precision_at_k(RANKED_IDS, RELEVANT_IDS, 4) == pytest.approx(0.5)


def test_hit_rate_at_k() -> None:
    assert hit_rate_at_k(RANKED_IDS, RELEVANT_IDS, 2) == 1.0
    assert hit_rate_at_k(["unrelated"], RELEVANT_IDS, 1) == 0.0


def test_mean_reciprocal_rank() -> None:
    assert mean_reciprocal_rank(
        [
            ["unrelated", "chunk-1"],
            ["chunk-2", "unrelated"],
        ],
        RELEVANT_IDS,
    ) == pytest.approx(0.75)


def test_metrics_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="k"):
        precision_at_k(RANKED_IDS, RELEVANT_IDS, 0)

    with pytest.raises(ValueError, match="relevant"):
        recall_at_k(RANKED_IDS, set(), 2)
