from typing import Any, Protocol

from app.citations.validator import (
    build_source_context,
    validate_citations,
)
from app.generation.prompts import build_grounded_prompt


class LanguageModel(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class AnswerGenerationError(ValueError):
    """Raised when an answer cannot be generated or validated."""


class AnswerGenerator:
    """Generate evidence-grounded answers through an injected LLM adapter."""

    def __init__(self, model: LanguageModel) -> None:
        self.model = model

    def generate(
        self,
        question: str,
        chunks: list[dict[str, Any]],
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Generate an answer and validate its source citations."""
        try:
            sources = build_source_context(chunks)
            prompt = build_grounded_prompt(question, sources, history)
            answer = self.model.generate(prompt)
        except (TypeError, KeyError, ValueError) as error:
            raise AnswerGenerationError(str(error)) from error

        if not isinstance(answer, str):
            raise AnswerGenerationError("Language model must return a string")

        try:
            cited_source_ids = validate_citations(answer, sources)
        except ValueError as error:
            raise AnswerGenerationError(str(error)) from error

        return {
            "answer": answer,
            "citations": [
                sources[source_id]
                for source_id in cited_source_ids
            ],
        }


class EvidenceOnlyModel:
    """Deterministic local fallback for development without an LLM provider."""

    def generate(self, prompt: str) -> str:
        evidence_marker = "Evidence:\n"
        evidence = prompt.split(evidence_marker, 1)[-1].strip()
        first_source = evidence.split("]", 1)[0] + "]"
        first_passage = evidence.split("\n", 1)[-1].split("\n\n", 1)[0]
        return f"{first_passage} {first_source}"
