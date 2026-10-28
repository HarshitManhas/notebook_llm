import json
from pathlib import Path

import pytest

from app.evaluation.benchmark import evaluate_retriever, load_dataset


def test_load_dataset_and_evaluate(tmp_path: Path) -> None:
    dataset_path = tmp_path / "eval.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "query": "python",
                "relevant_chunk_ids": ["chunk-1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_dataset(dataset_path)
    metrics = evaluate_retriever(
        dataset,
        lambda query, k: [{"chunk_id": "chunk-1"}],
        k=1,
    )

    assert metrics["recall_at_k"] == 1.0
    assert metrics["precision_at_k"] == 1.0
    assert metrics["hit_rate_at_k"] == 1.0
    assert metrics["mrr"] == 1.0


def test_invalid_dataset_is_rejected(tmp_path: Path) -> None:
    dataset_path = tmp_path / "invalid.jsonl"
    dataset_path.write_text('{"query": "missing labels"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="relevant_chunk_ids"):
        load_dataset(dataset_path)
