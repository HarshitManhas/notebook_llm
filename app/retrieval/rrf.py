from typing import Any, Iterable


def reciprocal_rank_fusion(
    ranked_lists: Iterable[list[dict[str, Any]]],
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse ranked result lists using reciprocal rank fusion."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero")

    fused: dict[str, dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            chunk_id = item.get("chunk_id")
            if not chunk_id:
                raise ValueError("Every result must contain a chunk_id")

            if chunk_id not in fused:
                fused[chunk_id] = dict(item)
                fused[chunk_id]["fused_score"] = 0.0

            fused[chunk_id]["fused_score"] += 1.0 / (rrf_k + rank)

    ranked_results = sorted(
        fused.values(),
        key=lambda item: item["fused_score"],
        reverse=True,
    )

    return ranked_results[:top_k]
