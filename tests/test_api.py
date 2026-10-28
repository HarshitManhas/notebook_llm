from uuid import uuid4

from fastapi.testclient import TestClient

import app.main as main
from app.main import app


client = TestClient(
    app,
    headers={"X-API-Key": "local-development-key"},
)


def make_chunk(text: str = "Python is a programming language.") -> dict[str, object]:
    return {
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "filename": "source.txt",
        "page_number": None,
        "text": text,
    }


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_frontend_is_served() -> None:
    response = TestClient(app).get("/frontend/")

    assert response.status_code == 200
    assert "Notebook LLM" in response.text
    assert "/frontend/login.html" in response.text


def test_source_url_targets_pdf_page() -> None:
    assert main.build_source_url(
        {
            "document_id": "doc-1",
            "filename": "paper.pdf",
            "page_number": 4,
        }
    ) == "/api/documents/doc-1/file#page=4"


def test_source_url_does_not_add_page_to_text_file() -> None:
    assert main.build_source_url(
        {
            "document_id": "doc-1",
            "filename": "notes.txt",
            "page_number": None,
        }
    ) == "/api/documents/doc-1/file"


def test_protected_endpoint_requires_api_key() -> None:
    response = TestClient(app).get("/api/notebooks")

    assert response.status_code == 401


def test_register_and_login_return_bearer_tokens() -> None:
    email = f"account-test-{uuid4().hex}@example.com"
    register_response = client.post(
        "/api/auth/register",
        json={"email": email, "password": "strong-password"},
    )

    assert register_response.status_code == 201
    assert register_response.json()["token_type"] == "bearer"

    login_response = client.post(
        "/api/auth/login",
        json={"email": email, "password": "strong-password"},
    )

    assert login_response.status_code == 200
    assert login_response.json()["access_token"]


def test_login_rejects_wrong_password() -> None:
    email = f"wrong-password-{uuid4().hex}@example.com"
    register_response = client.post(
        "/api/auth/register",
        json={"email": email, "password": "strong-password"},
    )
    assert register_response.status_code == 201

    response = client.post(
        "/api/auth/login",
        json={
            "email": email,
            "password": "wrong-password",
        },
    )

    assert response.status_code == 401


def test_search_endpoint_returns_ranked_results() -> None:
    response = client.post(
        "/api/search",
        json={
            "query": "Python",
            "chunks": [
                make_chunk(),
                {
                    **make_chunk("Cooking recipes contain ingredients."),
                    "chunk_id": "chunk-2",
                },
            ],
            "top_k": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["chunk_id"] == "chunk-1"
    assert isinstance(body["results"][0]["score"], float)


def test_search_endpoint_rejects_empty_chunks() -> None:
    response = client.post(
        "/api/search",
        json={"query": "Python", "chunks": []},
    )

    assert response.status_code == 422


def test_search_endpoint_rejects_empty_query() -> None:
    response = client.post(
        "/api/search",
        json={"query": "", "chunks": [make_chunk()]},
    )

    assert response.status_code == 422


def test_upload_list_and_search_document() -> None:
    upload_response = client.post(
        "/api/documents",
        files={"file": ("uploaded.txt", b"Python document search", "text/plain")},
    )

    assert upload_response.status_code == 201
    document = upload_response.json()
    assert document["filename"] == "uploaded.txt"
    assert document["chunk_count"] == 1

    list_response = client.get("/api/documents")
    assert list_response.status_code == 200
    assert any(
        item["document_id"] == document["document_id"]
        for item in list_response.json()
    )

    search_response = client.post(
        "/api/documents/search",
        json={
            "query": "Python",
            "document_ids": [document["document_id"]],
        },
    )
    assert search_response.status_code == 200
    assert search_response.json()["results"][0]["document_id"] == document[
        "document_id"
    ]


def test_upload_rejects_unsupported_file_type() -> None:
    response = client.post(
        "/api/documents",
        files={"file": ("uploaded.exe", b"not a document", "application/octet-stream")},
    )

    assert response.status_code == 400


def test_uploaded_source_file_can_be_viewed() -> None:
    upload_response = client.post(
        "/api/documents",
        files={"file": ("view.txt", b"Source content", "text/plain")},
    )
    assert upload_response.status_code == 201
    document_id = upload_response.json()["document_id"]

    response = client.get(f"/api/documents/{document_id}/file")

    assert response.status_code == 200
    assert response.content == b"Source content"
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    assert response.headers["content-disposition"].startswith("inline;")
    assert "view.txt" in response.headers["content-disposition"]


def test_stored_search_rejects_unknown_document() -> None:
    response = client.post(
        "/api/documents/search",
        json={"query": "Python", "document_ids": ["missing"]},
    )

    assert response.status_code == 400


def test_document_can_be_deleted() -> None:
    upload_response = client.post(
        "/api/documents",
        files={"file": ("delete-me.txt", b"Delete this source.", "text/plain")},
    )
    assert upload_response.status_code == 201
    document_id = upload_response.json()["document_id"]

    delete_response = client.delete(f"/api/documents/{document_id}")

    assert delete_response.status_code == 204
    assert all(
        item["document_id"] != document_id
        for item in client.get("/api/documents").json()
    )


def test_notebook_creation_upload_and_scoped_search() -> None:
    notebook_response = client.post(
        "/api/notebooks",
        json={"name": "Python research"},
    )

    assert notebook_response.status_code == 201
    notebook = notebook_response.json()

    upload_response = client.post(
        f"/api/documents?notebook_id={notebook['notebook_id']}",
        files={"file": ("notebook.txt", b"Python notebook content", "text/plain")},
    )
    assert upload_response.status_code == 201

    search_response = client.post(
        "/api/documents/search",
        json={
            "query": "Python",
            "notebook_id": notebook["notebook_id"],
        },
    )
    assert search_response.status_code == 200
    assert search_response.json()["results"][0]["filename"] == "notebook.txt"

    notebooks_response = client.get("/api/notebooks")
    matching = [
        item
        for item in notebooks_response.json()
        if item["notebook_id"] == notebook["notebook_id"]
    ]
    assert matching[0]["document_count"] == 1


def test_notebook_name_is_required() -> None:
    response = client.post("/api/notebooks", json={"name": ""})

    assert response.status_code == 422


def test_chat_endpoint_returns_grounded_answer_and_citation(
    monkeypatch,
) -> None:
    notebook_response = client.post(
        "/api/notebooks",
        json={"name": "Chat test"},
    )
    notebook_id = notebook_response.json()["notebook_id"]

    upload_response = client.post(
        f"/api/documents?notebook_id={notebook_id}",
        files={"file": ("chat.txt", b"Python is a language.", "text/plain")},
    )
    assert upload_response.status_code == 201

    class FakeOllamaModel:
        def generate(self, prompt: str) -> str:
            return "Python is a language. [S1]"

    monkeypatch.setattr(main, "get_language_model", lambda: FakeOllamaModel())

    response = client.post(
        "/api/chat",
        json={
            "query": "What is Python?",
            "notebook_id": notebook_id,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "[S1]" in body["answer"]
    assert body["citations"][0]["source_id"] == "S1"


def test_chat_endpoint_rejects_empty_notebook() -> None:
    response = client.post(
        "/api/chat",
        json={"query": "What is Python?", "notebook_id": "missing"},
    )

    assert response.status_code == 400


def test_chat_session_stores_conversation(monkeypatch) -> None:
    notebook_response = client.post(
        "/api/notebooks",
        json={"name": "Session test"},
    )
    notebook_id = notebook_response.json()["notebook_id"]
    upload_response = client.post(
        f"/api/documents?notebook_id={notebook_id}",
        files={"file": ("session.txt", b"Python is useful.", "text/plain")},
    )
    assert upload_response.status_code == 201

    session_response = client.post(
        "/api/chat/sessions",
        json={"notebook_id": notebook_id},
    )
    assert session_response.status_code == 201
    session_id = session_response.json()["session_id"]

    class FakeOllamaModel:
        def generate(self, prompt: str) -> str:
            return "Python is useful. [S1]"

    monkeypatch.setattr(main, "get_language_model", lambda: FakeOllamaModel())

    chat_response = client.post(
        "/api/chat",
        json={"query": "What is Python?", "session_id": session_id},
    )
    assert chat_response.status_code == 200

    messages_response = client.get(
        f"/api/chat/sessions/{session_id}/messages"
    )
    assert messages_response.status_code == 200
    assert [message["role"] for message in messages_response.json()] == [
        "user",
        "assistant",
    ]
