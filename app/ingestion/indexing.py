from typing import Any

from app.ingestion.chunker import chunk_pages
from app.ingestion.parsers import parse_document


def ingest_document(
    file_path: str,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[dict[str, Any]]:
    """Parse a document and convert it into searchable chunks."""
    pages = parse_document(file_path)

    return chunk_pages(
        pages,
        chunk_size=chunk_size,
        overlap=overlap,
    )