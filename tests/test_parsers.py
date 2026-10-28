from pathlib import Path

from app.ingestion.parsers import parse_document


def test_parse_text_file(tmp_path: Path) -> None:
    file_path = tmp_path / "example.txt"
    file_path.write_text("Hello document ingestion.", encoding="utf-8")

    pages = parse_document(str(file_path))

    assert len(pages) == 1
    assert pages[0]["text"] == "Hello document ingestion."
    assert pages[0]["page_number"] is None
    assert pages[0]["filename"] == "example.txt"