import json
from urllib import request

import pytest

from app.generation.ollama import OllamaGenerationError, OllamaModel


class FakeResponse:
    def __init__(self, body: dict[str, object]) -> None:
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_ollama_model_sends_configured_request(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(http_request, timeout):
        captured["url"] = http_request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(http_request.data.decode("utf-8"))
        return FakeResponse({"response": "Answer [S1]"})

    monkeypatch.setattr(request, "urlopen", fake_urlopen)

    answer = OllamaModel(
        model="gemma3:4b",
        base_url="http://localhost:11434",
        timeout=30,
    ).generate("Question")

    assert answer == "Answer [S1]"
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["timeout"] == 30
    assert captured["body"] == {
        "model": "gemma3:4b",
        "prompt": "Question",
        "stream": False,
    }


def test_ollama_model_rejects_empty_prompt() -> None:
    with pytest.raises(OllamaGenerationError, match="Prompt"):
        OllamaModel().generate("")


def test_ollama_model_rejects_empty_response(monkeypatch) -> None:
    monkeypatch.setattr(
        request,
        "urlopen",
        lambda http_request, timeout: FakeResponse({"response": ""}),
    )

    with pytest.raises(OllamaGenerationError, match="no text"):
        OllamaModel().generate("Question")
