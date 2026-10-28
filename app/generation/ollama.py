import json
import os
from urllib import error, request


class OllamaGenerationError(RuntimeError):
    """Raised when Ollama cannot generate an answer."""


class OllamaModel:
    """Language-model adapter for the local Ollama HTTP API."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "gemma3:4b")
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        ).rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise OllamaGenerationError("Prompt must not be empty")

        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }
        ).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaGenerationError(
                f"Could not connect to Ollama at {self.base_url}: {exc}"
            ) from exc

        answer = body.get("response")
        if not isinstance(answer, str) or not answer.strip():
            raise OllamaGenerationError(
                "Ollama returned no text response"
            )

        return answer.strip()
