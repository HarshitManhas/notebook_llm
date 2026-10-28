from typing import Any


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[dict[str, Any]]:
    """Split parsed pages into overlapping word-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[dict[str, Any]] = []

    for page in pages:
        words = page["text"].split()
        start = 0
        chunk_number = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]

            chunks.append(
                {
                    "chunk_id": (
                        f"{page['document_id']}-"
                        f"{page['page_number'] or 'document'}-"
                        f"{chunk_number}"
                    ),
                    "document_id": page["document_id"],
                    "source_path": page["source_path"],
                    "filename": page["filename"],
                    "page_number": page["page_number"],
                    "text": " ".join(chunk_words),
                }
            )

            chunk_number += 1

            if end == len(words):
                break

            start = end - overlap

    return chunks