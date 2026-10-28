import json
from pathlib import Path
from typing import Any, Callable

from app.evaluation.metrics import (
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
)


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL questions with query and relevant_chunk_ids fields."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON on evaluation dataset line {line_number}"
            ) from error

        if (
            not isinstance(record.get("query"), str)
            or not record["query"].strip()
            or not isinstance(record.get("relevant_chunk_ids"), list)
            or not record["relevant_chunk_ids"]
        ):
            raise ValueError(
                "Each record requires a non-empty query and "
                "relevant_chunk_ids list"
            )
        records.append(record)

    if not records:
        raise ValueError("Evaluation dataset is empty")
    return records


def evaluate_retriever(
    dataset: list[dict[str, Any]],
    search: Callable[[str, int], list[dict[str, Any]]],
    k: int = 5,
) -> dict[str, float]:
    """Evaluate a retriever against labeled relevant chunk IDs."""
    if not dataset:
        raise ValueError("Evaluation dataset is empty")

    per_query: list[dict[str, float]] = []
    reciprocal_ranks: list[float] = []
    for record in dataset:
        results = search(record["query"], k)
        ranked_ids = [result["chunk_id"] for result in results]
        relevant_ids = set(record["relevant_chunk_ids"])
        reciprocal_rank = 0.0
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            if chunk_id in relevant_ids:
                reciprocal_rank = 1.0 / rank
                break
        reciprocal_ranks.append(reciprocal_rank)
        per_query.append(
            {
                "recall": recall_at_k(ranked_ids, relevant_ids, k),
                "precision": precision_at_k(ranked_ids, relevant_ids, k),
                "hit_rate": hit_rate_at_k(ranked_ids, relevant_ids, k),
            }
        )

    return {
        "recall_at_k": sum(item["recall"] for item in per_query) / len(per_query),
        "precision_at_k": sum(item["precision"] for item in per_query)
        / len(per_query),
        "hit_rate_at_k": sum(item["hit_rate"] for item in per_query)
        / len(per_query),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
    }
