import json
from pathlib import Path

from app.evaluation.cli import main


def test_cli_prints_metrics(tmp_path: Path, capsys) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "query": "python",
                "relevant_chunk_ids": ["chunk-1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text(
        json.dumps(
            [
                {"chunk_id": "chunk-1", "text": "Python programming"},
                {"chunk_id": "chunk-2", "text": "Cooking recipe"},
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--dataset",
            str(dataset_path),
            "--chunks",
            str(chunks_path),
            "--k",
            "1",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["recall_at_k"] == 1.0


def test_cli_returns_error_for_invalid_k(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "--dataset",
            str(tmp_path / "missing.jsonl"),
            "--chunks",
            str(tmp_path / "missing.json"),
            "--k",
            "0",
        ]
    )

    assert exit_code == 2
    assert "k must be greater than zero" in capsys.readouterr().err
