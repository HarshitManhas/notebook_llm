import re
from typing import Any


_CITATION_PATTERN = re.compile(r"\[S(\d+)\]")


class CitationValidationError(ValueError):
    """Raised when an answer contains invalid or unsupported citations."""


def build_source_context(
    chunks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Assign stable source labels to retrieved chunks."""
    sources: dict[str, dict[str, Any]] = {}

    for index, chunk in enumerate(chunks, start=1):
        _validate_chunk_metadata(chunk)
        source_id = f"S{index}"
        source = dict(chunk)
        source["source_id"] = source_id
        sources[source_id] = source

    return sources


def validate_citations(
    answer: str,
    sources: dict[str, dict[str, Any]],
) -> list[str]:
    """Validate all source IDs cited in an answer and return them in order."""
    if not answer.strip():
        raise CitationValidationError("Answer must not be empty")

    cited_ids = _CITATION_PATTERN.findall(answer)
    if not cited_ids:
        raise CitationValidationError("Answer must contain at least one citation")

    normalized_ids = [f"S{source_number}" for source_number in cited_ids]
    unknown_ids = sorted(set(normalized_ids) - sources.keys())
    if unknown_ids:
        raise CitationValidationError(
            f"Unknown citation IDs: {', '.join(unknown_ids)}"
        )

    return list(dict.fromkeys(normalized_ids))


def _validate_chunk_metadata(chunk: dict[str, Any]) -> None:
    required_fields = ("chunk_id", "document_id", "filename", "text")
    missing_fields = [
        field
        for field in required_fields
        if not isinstance(chunk.get(field), str) or not chunk[field].strip()
    ]
    if missing_fields:
        raise CitationValidationError(
            f"Source is missing required metadata: {', '.join(missing_fields)}"
        )

    page_number = chunk.get("page_number")
    if page_number is not None and (
        not isinstance(page_number, int) or page_number < 1
    ):
        raise CitationValidationError(
            "Source page_number must be None or a positive integer"
        )
