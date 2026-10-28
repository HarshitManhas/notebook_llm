import pytest

from app.retrieval.bm25 import BM25Retriever, tokenize


def make_chunks() -> list[dict[str, object]]:
    return [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "source_path": "data/raw/python.txt",
            "filename": "python.txt",
            "page_number": None,
            "text": "Python is a programming language.",
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "doc-2",
            "source_path": "data/raw/search.txt",
            "filename": "search.txt",
            "page_number": None,
            "text": "BM25 is useful for keyword document search.",
        },
        {
            "chunk_id": "chunk-3",
            "document_id": "doc-3",
            "source_path": "data/raw/cooking.txt",
            "filename": "cooking.txt",
            "page_number": None,
            "text": "A recipe contains ingredients and cooking instructions.",
        },
    ]


def test_tokenize_preserves_technical_terms() -> None:
    assert tokenize("Python error-code v2.1") == [
        "python",
        "error-code",
        "v2.1",
    ]


def test_search_returns_matching_chunk_and_metadata() -> None:
    retriever = BM25Retriever(make_chunks())

    results = retriever.search("keyword document search", top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk-2"
    assert results[0]["filename"] == "search.txt"
    assert results[0]["score"] > 0


def test_search_respects_top_k() -> None:
    retriever = BM25Retriever(make_chunks())

    assert len(retriever.search("Python document cooking", top_k=2)) == 2


def test_empty_query_is_rejected() -> None:
    retriever = BM25Retriever(make_chunks())

    with pytest.raises(ValueError, match="Query must not be empty"):
        retriever.search("")


def test_invalid_top_k_is_rejected() -> None:
    retriever = BM25Retriever(make_chunks())

    with pytest.raises(ValueError, match="top_k"):
        retriever.search("Python", top_k=0)


def test_empty_chunks_are_rejected() -> None:
    with pytest.raises(ValueError, match="At least one chunk"):
        BM25Retriever([])
