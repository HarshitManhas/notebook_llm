from typing import Iterable


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be greater than zero")


def _top_ids(ranked_chunk_ids: Iterable[str], k: int) -> list[str]:
    _validate_k(k)
    return list(ranked_chunk_ids)[:k]


def recall_at_k(
    ranked_chunk_ids: Iterable[str],
    relevant_chunk_ids: set[str],
    k: int,
) -> float:
    if not relevant_chunk_ids:
        raise ValueError("relevant_chunk_ids must not be empty")
    found = set(_top_ids(ranked_chunk_ids, k)) & relevant_chunk_ids
    return len(found) / len(relevant_chunk_ids)


def precision_at_k(
    ranked_chunk_ids: Iterable[str],
    relevant_chunk_ids: set[str],
    k: int,
) -> float:
    retrieved = _top_ids(ranked_chunk_ids, k)
    if not retrieved:
        return 0.0
    return len(set(retrieved) & relevant_chunk_ids) / len(retrieved)


def hit_rate_at_k(
    ranked_chunk_ids: Iterable[str],
    relevant_chunk_ids: set[str],
    k: int,
) -> float:
    return float(bool(set(_top_ids(ranked_chunk_ids, k)) & relevant_chunk_ids))


def mean_reciprocal_rank(
    ranked_lists: Iterable[Iterable[str]],
    relevant_chunk_ids: set[str],
) -> float:
    if not relevant_chunk_ids:
        raise ValueError("relevant_chunk_ids must not be empty")

    reciprocal_ranks = []
    for ranked_ids in ranked_lists:
        reciprocal_rank = 0.0
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            if chunk_id in relevant_chunk_ids:
                reciprocal_rank = 1.0 / rank
                break
        reciprocal_ranks.append(reciprocal_rank)

    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)
