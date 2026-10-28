from pathlib import Path

from app.documents import DocumentStore


def test_documents_persist_across_store_instances(tmp_path: Path) -> None:
    database_path = tmp_path / "documents.sqlite3"

    first_store = DocumentStore(str(database_path))
    document = first_store.add_document(
        "notes.txt",
        b"Python persistence works.",
    )

    second_store = DocumentStore(str(database_path))

    assert second_store.list_documents() == [
        {
            "document_id": document["document_id"],
            "filename": "notes.txt",
            "chunk_count": 1,
        }
    ]
    results = second_store.search("persistence")
    assert results[0]["document_id"] == document["document_id"]


def test_chat_history_persists(tmp_path: Path) -> None:
    database_path = tmp_path / "chat.sqlite3"
    store = DocumentStore(str(database_path))
    notebook = store.create_notebook("Research")
    session = store.create_chat_session(notebook["notebook_id"])
    store.add_chat_message(session["session_id"], "user", "Question")
    store.add_chat_message(session["session_id"], "assistant", "Answer [S1]")

    reopened = DocumentStore(str(database_path))

    assert reopened.get_chat_history(session["session_id"]) == [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer [S1]"},
    ]


def test_documents_are_isolated_by_owner_and_can_be_deleted(tmp_path: Path) -> None:
    store = DocumentStore(str(tmp_path / "ownership.sqlite3"))
    first = store.create_notebook("First", owner_id="user-1")
    second = store.create_notebook("Second", owner_id="user-2")
    document = store.add_document(
        "private.txt",
        b"Private research",
        notebook_id=first["notebook_id"],
        owner_id="user-1",
    )
    store.add_document(
        "other.txt",
        b"Other research",
        notebook_id=second["notebook_id"],
        owner_id="user-2",
    )

    assert len(store.list_documents(owner_id="user-1")) == 1
    assert store.list_documents(
        notebook_id=second["notebook_id"],
        owner_id="user-1",
    ) == []

    store.delete_document(document["document_id"], owner_id="user-1")
    assert store.list_documents(owner_id="user-1") == []
    try:
        store.get_chunks(
            notebook_id=first["notebook_id"],
            owner_id="user-1",
        )
    except ValueError as error:
        assert str(error) == "No documents have been uploaded"
    else:
        raise AssertionError("Expected empty notebook retrieval to fail")


def test_notebook_document_counts_are_isolated_by_owner(tmp_path: Path) -> None:
    store = DocumentStore(str(tmp_path / "notebook_counts.sqlite3"))
    first = store.create_notebook("First", owner_id="user-1")
    second = store.create_notebook("Second", owner_id="user-2")
    store.add_document(
        "private.txt",
        b"Private research",
        notebook_id=first["notebook_id"],
        owner_id="user-1",
    )

    assert store.list_notebooks(owner_id="user-2") == [
        {
            "notebook_id": second["notebook_id"],
            "name": "Second",
            "document_count": 0,
        }
    ]
