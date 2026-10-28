import sqlite3
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.ingestion.indexing import ingest_document
from app.retrieval.bm25 import BM25Retriever


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class DocumentStore:
    """SQLite-backed document store for uploaded files and chunks."""

    def __init__(self, db_path: str = "data/notebookllm.sqlite3") -> None:
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self.storage_dir = Path(db_path).parent / "uploads"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS notebooks (
                    notebook_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    notebook_id TEXT REFERENCES notebooks(notebook_id),
                    file_path TEXT,
                    owner_id TEXT NOT NULL DEFAULT 'local-development-user'
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(document_id)
                        ON DELETE CASCADE,
                    source_path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    page_number INTEGER,
                    text TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    notebook_id TEXT NOT NULL REFERENCES notebooks(notebook_id),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES chat_sessions(session_id)
                        ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            document_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(documents)"
                ).fetchall()
            }
            if "notebook_id" not in document_columns:
                connection.execute(
                    "ALTER TABLE documents ADD COLUMN notebook_id TEXT"
                )
            if "file_path" not in document_columns:
                connection.execute(
                    "ALTER TABLE documents ADD COLUMN file_path TEXT"
                )
            if "owner_id" not in document_columns:
                connection.execute(
                    "ALTER TABLE documents ADD COLUMN owner_id TEXT "
                    "NOT NULL DEFAULT 'local-development-user'"
                )
            notebook_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(notebooks)"
                ).fetchall()
            }
            if "owner_id" not in notebook_columns:
                connection.execute(
                    "ALTER TABLE notebooks ADD COLUMN owner_id TEXT "
                    "NOT NULL DEFAULT 'local-development-user'"
                )

    def create_user(self, email: str, password_hash: str) -> dict[str, Any]:
        user = {"user_id": uuid4().hex, "email": email.strip().lower()}
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO users(user_id, email, password_hash) VALUES (?, ?, ?)",
                    (user["user_id"], user["email"], password_hash),
                )
        except sqlite3.IntegrityError as error:
            raise ValueError("An account with that email already exists") from error
        return user

    def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT user_id, email, password_hash FROM users WHERE email = ?",
                (email.strip().lower(),),
            ).fetchone()
        return dict(row) if row else None

    def create_chat_session(
        self,
        notebook_id: str,
        owner_id: str = "local-development-user",
    ) -> dict[str, Any]:
        session_id = uuid4().hex
        with self._connect() as connection:
            notebook = connection.execute(
                "SELECT 1 FROM notebooks WHERE notebook_id = ? AND owner_id = ?",
                (notebook_id, owner_id),
            ).fetchone()
            if notebook is None:
                raise ValueError(f"Unknown notebook ID: {notebook_id}")
            connection.execute(
                """
                INSERT INTO chat_sessions(session_id, notebook_id)
                VALUES (?, ?)
                """,
                (session_id, notebook_id),
            )
        return {"session_id": session_id, "notebook_id": notebook_id}

    def get_chat_session(
        self,
        session_id: str,
        owner_id: str = "local-development-user",
    ) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT s.session_id, s.notebook_id
                FROM chat_sessions s
                JOIN notebooks n ON n.notebook_id = s.notebook_id
                WHERE s.session_id = ? AND n.owner_id = ?
                """,
                (session_id, owner_id),
            ).fetchone()
        if row is None:
            raise ValueError(f"Unknown chat session ID: {session_id}")
        return dict(row)

    def get_chat_history(
        self,
        session_id: str,
        owner_id: str = "local-development-user",
    ) -> list[dict[str, str]]:
        self.get_chat_session(session_id, owner_id=owner_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY message_id
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        owner_id: str = "local-development-user",
    ) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("Chat message role must be user or assistant")
        if not content.strip():
            raise ValueError("Chat message content must not be empty")
        self.get_chat_session(session_id, owner_id=owner_id)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO chat_messages(session_id, role, content)
                VALUES (?, ?, ?)
                """,
                (session_id, role, content),
            )

    def create_notebook(
        self,
        name: str,
        owner_id: str = "local-development-user",
    ) -> dict[str, Any]:
        if not name.strip():
            raise ValueError("Notebook name must not be empty")

        notebook = {
            "notebook_id": uuid4().hex,
            "name": name.strip(),
            "owner_id": owner_id,
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO notebooks(notebook_id, name, owner_id)
                VALUES (?, ?, ?)
                """,
                (
                    notebook["notebook_id"],
                    notebook["name"],
                    owner_id,
                ),
            )
        return notebook

    def list_notebooks(
        self,
        owner_id: str = "local-development-user",
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT n.notebook_id, n.name, COUNT(d.document_id) AS document_count
                FROM notebooks n
                LEFT JOIN documents d
                    ON d.notebook_id = n.notebook_id
                    AND d.owner_id = n.owner_id
                WHERE n.owner_id = ?
                GROUP BY n.notebook_id
                ORDER BY n.rowid
                """,
                (owner_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_document(
        self,
        filename: str,
        content: bytes,
        notebook_id: str | None = None,
        owner_id: str = "local-development-user",
    ) -> dict[str, Any]:
        extension = Path(filename).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                "Unsupported file type. Use PDF, TXT, or Markdown."
            )

        if not content:
            raise ValueError("Uploaded document is empty")

        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError("Uploaded document exceeds the 10 MB limit")

        if notebook_id is not None:
            with self._connect() as connection:
                notebook = connection.execute(
                    """
                    SELECT 1 FROM notebooks
                    WHERE notebook_id = ? AND owner_id = ?
                    """,
                    (notebook_id, owner_id),
                ).fetchone()
            if notebook is None:
                raise ValueError(f"Unknown notebook ID: {notebook_id}")

        document_id = uuid4().hex
        temporary_path: str | None = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=extension,
                delete=False,
            ) as temporary_file:
                temporary_file.write(content)
                temporary_path = temporary_file.name

            chunks = ingest_document(temporary_path)
        except (OSError, ValueError) as error:
            raise ValueError(f"Could not ingest document: {error}") from error
        finally:
            if temporary_path is not None:
                Path(temporary_path).unlink(missing_ok=True)

        if not chunks:
            raise ValueError("Document contains no extractable text")

        for index, chunk in enumerate(chunks):
            chunk["document_id"] = document_id
            chunk["chunk_id"] = f"{document_id}-{index}"
            chunk["filename"] = filename
            chunk["source_path"] = filename

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents(
                    document_id, filename, chunk_count, notebook_id, owner_id
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (document_id, filename, len(chunks), notebook_id, owner_id),
            )
            connection.executemany(
                """
                INSERT INTO chunks(
                    chunk_id, document_id, source_path, filename,
                    page_number, text
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk["chunk_id"],
                        chunk["document_id"],
                        chunk["source_path"],
                        chunk["filename"],
                        chunk["page_number"],
                        chunk["text"],
                    )
                    for chunk in chunks
                ],
            )
            stored_path = self.storage_dir / f"{document_id}{extension}"
            stored_path.write_bytes(content)
            connection.execute(
                "UPDATE documents SET file_path = ? WHERE document_id = ?",
                (str(stored_path), document_id),
            )

        return {
            "document_id": document_id,
            "filename": filename,
            "chunk_count": len(chunks),
        }

    def get_document_file(
        self,
        document_id: str,
        owner_id: str = "local-development-user",
    ) -> tuple[str, str]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT d.filename, d.file_path
                FROM documents d
                WHERE d.document_id = ?
                AND d.owner_id = ?
                """,
                (document_id, owner_id),
            ).fetchone()
        if row is None or not row["file_path"]:
            raise ValueError(f"Unknown document ID: {document_id}")
        path = Path(row["file_path"])
        if not path.is_file():
            raise ValueError("The original source file is no longer available")
        return str(path), row["filename"]

    def delete_document(
        self,
        document_id: str,
        owner_id: str = "local-development-user",
    ) -> None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT file_path
                FROM documents
                WHERE document_id = ? AND owner_id = ?
                """,
                (document_id, owner_id),
            ).fetchone()
            if row is None:
                raise ValueError(f"Unknown document ID: {document_id}")
            connection.execute(
                "DELETE FROM documents WHERE document_id = ? AND owner_id = ?",
                (document_id, owner_id),
            )

        if row["file_path"]:
            Path(row["file_path"]).unlink(missing_ok=True)

    def list_documents(
        self,
        notebook_id: str | None = None,
        owner_id: str = "local-development-user",
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT d.document_id, d.filename, d.chunk_count
                FROM documents d
                WHERE d.owner_id = ?
                AND (? IS NULL OR d.notebook_id = ?)
                ORDER BY d.rowid
                """,
                (owner_id, notebook_id, notebook_id),
            ).fetchall()

        return [dict(row) for row in rows]

    def search(
        self,
        query: str,
        document_ids: list[str] | None = None,
        notebook_id: str | None = None,
        top_k: int = 5,
        owner_id: str = "local-development-user",
    ) -> list[dict[str, Any]]:
        from app.retrieval.bm25 import BM25Retriever

        chunks = self.get_chunks(
            document_ids=document_ids,
            notebook_id=notebook_id,
            owner_id=owner_id,
        )
        return BM25Retriever(chunks).search(query, top_k=top_k)

    def get_chunks(
        self,
        document_ids: list[str] | None = None,
        notebook_id: str | None = None,
        owner_id: str = "local-development-user",
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if notebook_id is not None:
                notebook = connection.execute(
                    "SELECT 1 FROM notebooks WHERE notebook_id = ? AND owner_id = ?",
                    (notebook_id, owner_id),
                ).fetchone()
                if notebook is None:
                    raise ValueError(f"Unknown notebook ID: {notebook_id}")

            available_ids = {
                row["document_id"]
                for row in connection.execute(
                    """
                    SELECT d.document_id
                    FROM documents d
                    WHERE d.owner_id = ?
                    AND (? IS NULL OR d.notebook_id = ?)
                    """,
                    (owner_id, notebook_id, notebook_id),
                ).fetchall()
            }

            selected_ids = document_ids or list(available_ids)
            unknown_ids = sorted(set(selected_ids) - available_ids)
            if unknown_ids:
                raise ValueError(
                    f"Unknown document IDs: {', '.join(unknown_ids)}"
                )

            if not selected_ids:
                raise ValueError("No documents have been uploaded")

            placeholders = ",".join("?" for _ in selected_ids)
            rows = connection.execute(
                f"""
                SELECT c.chunk_id, c.document_id, c.source_path, c.filename,
                       page_number, text
                FROM chunks c
                JOIN documents d ON d.document_id = c.document_id
                WHERE c.document_id IN ({placeholders})
                AND d.owner_id = ?
                ORDER BY c.rowid
                """,
                [*selected_ids, owner_id],
            ).fetchall()

        chunks = [dict(row) for row in rows]
        if not chunks:
            raise ValueError("No documents have been uploaded")

        return chunks
