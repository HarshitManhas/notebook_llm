from app.ingestion.chunker import chunk_pages


def test_chunk_pages_creates_chunks() -> None:
    pages = [
        {
            "document_id": "doc-1",
            "source_path": "data/raw/doc.txt",
            "filename": "doc.txt",
            "page_number": None,
            "text": "one two three four five six seven eight",
        }
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=4,
        overlap=1,
    )

    assert len(chunks) == 3
    assert chunks[0]["text"] == "one two three four"
    assert chunks[1]["text"] == "four five six seven"
    assert chunks[2]["text"] == "seven eight"