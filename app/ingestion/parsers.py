from pathlib import Path
from typing import Any

import fitz


def parse_document(file_path: str) -> list[dict[str, Any]]:
    """Extract text and metadata from a PDF, TXT, or Markdown file."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document does not exist: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _parse_pdf(path)

    if suffix in {".txt", ".md", ".markdown"}:
        return _parse_text_file(path)

    raise ValueError(f"Unsupported file type: {suffix}")


def _parse_pdf(path: Path) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []

    with fitz.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append(
                    {
                        "document_id": path.stem,
                        "source_path": str(path),
                        "filename": path.name,
                        "page_number": page_number,
                        "text": text,
                    }
                )

    return pages


def _parse_text_file(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    return [
        {
            "document_id": path.stem,
            "source_path": str(path),
            "filename": path.name,
            "page_number": None,
            "text": text,
        }
    ]