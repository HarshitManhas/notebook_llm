import pytest

from app.citations.validator import (
    CitationValidationError,
    build_source_context,
    validate_citations,
)


def make_chunks() -> list[dict[str, object]]:
    return [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "filename": "source.pdf",
            "page_number": 2,
            "text": "The first source passage.",
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "doc-2",
            "filename": "notes.md",
            "page_number": None,
            "text": "The second source passage.",
        },
    ]


def test_build_source_context_assigns_source_ids() -> None:
    sources = build_source_context(make_chunks())

    assert list(sources) == ["S1", "S2"]
    assert sources["S1"]["source_id"] == "S1"
    assert sources["S1"]["page_number"] == 2


def test_validate_citations_returns_unique_ids_in_appearance_order() -> None:
    sources = build_source_context(make_chunks())

    assert validate_citations(
        "The answer is supported [S2]. It is also supported [S1] [S2].",
        sources,
    ) == ["S2", "S1"]


def test_unknown_citation_is_rejected() -> None:
    sources = build_source_context(make_chunks())

    with pytest.raises(CitationValidationError, match="Unknown citation IDs"):
        validate_citations("Unsupported claim [S3].", sources)


def test_missing_citations_are_rejected() -> None:
    sources = build_source_context(make_chunks())

    with pytest.raises(CitationValidationError, match="at least one citation"):
        validate_citations("This answer has no source.", sources)


def test_invalid_source_metadata_is_rejected() -> None:
    invalid_chunk = dict(make_chunks()[0])
    invalid_chunk["page_number"] = 0

    with pytest.raises(CitationValidationError, match="page_number"):
        build_source_context([invalid_chunk])
