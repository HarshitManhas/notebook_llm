import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.evaluation.benchmark import evaluate_retriever, load_dataset
from app.retrieval.bm25 import BM25Retriever


def load_chunks(path: str | Path) -> list[dict[str, Any]]:
    chunks_path = Path(path)
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    try:
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in chunks file: {path}") from error

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Chunks file must contain a non-empty JSON list")
    if any(not isinstance(chunk, dict) for chunk in chunks):
        raise ValueError("Every chunk in the chunks file must be an object")
    return chunks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate BM25 retrieval against a JSONL dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to JSONL evaluation dataset.",
    )
    parser.add_argument(
        "--chunks",
        required=True,
        help="Path to a JSON list of indexed chunks.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to evaluate (default: 5).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.k <= 0:
            raise ValueError("k must be greater than zero")
        dataset = load_dataset(args.dataset)
        retriever = BM25Retriever(load_chunks(args.chunks))
        metrics = evaluate_retriever(
            dataset,
            lambda query, k: retriever.search(query, top_k=k),
            k=args.k,
        )
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
