import pytest

from app.generation.answer_generator import (
    AnswerGenerationError,
    AnswerGenerator,
)
from app.generation.prompts import build_grounded_prompt


def make_chunks() -> list[dict[str, object]]:
    return [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "filename": "source.pdf",
            "page_number": 2,
            "text": "Python is a programming language.",
        }
    ]


class ValidModel:
    def __init__(self) -> None:
        self.prompt = ""

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return "Python is a programming language [S1]."


def test_prompt_contains_question_evidence_and_safety_rules() -> None:
    prompt = build_grounded_prompt("What is Python?", {"S1": make_chunks()[0]})

    assert "What is Python?" in prompt
    assert "[S1]" in prompt
    assert "untrusted document content" in prompt
    assert "source.pdf, page 2" in prompt


def test_answer_generator_returns_validated_citations() -> None:
    model = ValidModel()
    result = AnswerGenerator(model).generate("What is Python?", make_chunks())

    assert result["answer"].endswith("[S1].")
    assert result["citations"][0]["source_id"] == "S1"
    assert "[S1]" in model.prompt


def test_invalid_model_citation_is_rejected() -> None:
    class InvalidModel:
        def generate(self, prompt: str) -> str:
            return "This claim is unsupported [S9]."

    with pytest.raises(AnswerGenerationError, match="Unknown citation IDs"):
        AnswerGenerator(InvalidModel()).generate("Question", make_chunks())


def test_model_must_return_a_string() -> None:
    class InvalidModel:
        def generate(self, prompt: str) -> int:
            return 42

    with pytest.raises(AnswerGenerationError, match="must return a string"):
        AnswerGenerator(InvalidModel()).generate("Question", make_chunks())


def test_empty_question_is_rejected() -> None:
    with pytest.raises(AnswerGenerationError, match="Question must not be empty"):
        AnswerGenerator(ValidModel()).generate("", make_chunks())


def test_prompt_includes_previous_conversation() -> None:
    prompt = build_grounded_prompt(
        "What language is it?",
        {"S1": make_chunks()[0]},
        history=[
            {"role": "user", "content": "What is this?"},
            {"role": "assistant", "content": "It is a programming language [S1]."},
        ],
    )

    assert "Previous conversation:" in prompt
    assert "What is this?" in prompt
