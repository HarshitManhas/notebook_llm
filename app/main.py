import mimetypes
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.documents import DocumentStore
from app.generation.answer_generator import (
    AnswerGenerationError,
    AnswerGenerator,
)
from app.generation.ollama import OllamaModel
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.reranking.cross_encoder import CrossEncoderReranker
from app.security.auth import (
    create_access_token,
    hash_password,
    require_identity,
    verify_password,
)


app = FastAPI(
    title="Notebook LLM",
    version="0.1.0",
    description="Document search and question-answering backend.",
)
document_store = DocumentStore()
app.mount(
    "/frontend",
    StaticFiles(directory="frontend", html=True),
    name="frontend",
)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_reranking_model() -> CrossEncoder:
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@lru_cache(maxsize=1)
def get_language_model() -> OllamaModel:
    return OllamaModel()


def build_source_url(citation: dict[str, Any]) -> str:
    """Build a source URL with PDF page navigation when available."""
    url = f"/api/documents/{citation['document_id']}/file"
    if (
        Path(citation["filename"]).suffix.lower() == ".pdf"
        and citation.get("page_number") is not None
    ):
        return f"{url}#page={citation['page_number']}"
    return url


class ChunkInput(BaseModel):
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    text: str = Field(min_length=1)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    chunks: list[ChunkInput] = Field(min_length=1)
    top_k: int = Field(default=5, ge=1)


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int | None
    text: str
    score: float
    fused_score: float | None = None
    rerank_score: float | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]


class StoredDocument(BaseModel):
    document_id: str
    filename: str
    chunk_count: int


class NotebookCreate(BaseModel):
    name: str = Field(min_length=1)


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3)
    password: str = Field(min_length=8)


class LoginRequest(RegisterRequest):
    pass


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class Notebook(BaseModel):
    notebook_id: str
    name: str
    document_count: int


class StoredSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    document_ids: list[str] | None = None
    notebook_id: str | None = None
    mode: Literal["bm25", "hybrid"] = "bm25"
    top_k: int = Field(default=5, ge=1)
    candidate_k: int = Field(default=30, ge=1)


class ChatRequest(StoredSearchRequest):
    session_id: str | None = None


class ChatSessionCreate(BaseModel):
    notebook_id: str = Field(min_length=1)


class ChatSession(BaseModel):
    session_id: str
    notebook_id: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Citation(BaseModel):
    source_id: str
    chunk_id: str
    document_id: str
    filename: str
    page_number: int | None
    text: str
    source_url: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/auth/register", response_model=TokenResponse, status_code=201)
def register(request: RegisterRequest) -> TokenResponse:
    try:
        user = document_store.create_user(
            request.email,
            hash_password(request.password),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return TokenResponse(access_token=create_access_token(user["user_id"]))


@app.post("/api/auth/login", response_model=TokenResponse)
def login(request: LoginRequest) -> TokenResponse:
    user = document_store.get_user_by_email(request.email)
    if user is None or not verify_password(
        request.password,
        user["password_hash"],
    ):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return TokenResponse(access_token=create_access_token(user["user_id"]))


@app.post(
    "/api/notebooks",
    response_model=Notebook,
    status_code=201,
    dependencies=[Depends(require_identity)],
)
def create_notebook(
    request: NotebookCreate,
    user_id: str = Depends(require_identity),
) -> Notebook:
    try:
        notebook = document_store.create_notebook(request.name, owner_id=user_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    notebook["document_count"] = 0
    return Notebook(**notebook)


@app.get(
    "/api/notebooks",
    response_model=list[Notebook],
    dependencies=[Depends(require_identity)],
)
def list_notebooks(
    user_id: str = Depends(require_identity),
) -> list[Notebook]:
    return [
        Notebook(**notebook)
        for notebook in document_store.list_notebooks(owner_id=user_id)
    ]


@app.post(
    "/api/chat/sessions",
    response_model=ChatSession,
    status_code=201,
    dependencies=[Depends(require_identity)],
)
def create_chat_session(
    request: ChatSessionCreate,
    user_id: str = Depends(require_identity),
) -> ChatSession:
    try:
        return ChatSession(
            **document_store.create_chat_session(
                request.notebook_id,
                owner_id=user_id,
            )
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get(
    "/api/chat/sessions/{session_id}/messages",
    response_model=list[ChatMessage],
    dependencies=[Depends(require_identity)],
)
def get_chat_messages(
    session_id: str,
    user_id: str = Depends(require_identity),
) -> list[ChatMessage]:
    try:
        history = document_store.get_chat_history(session_id, owner_id=user_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return [ChatMessage(**message) for message in history]


@app.post(
    "/api/documents",
    response_model=StoredDocument,
    status_code=201,
    dependencies=[Depends(require_identity)],
)
async def upload_document(
    file: UploadFile = File(...),
    notebook_id: str | None = None,
    user_id: str = Depends(require_identity),
) -> StoredDocument:
    if not file.filename:
        raise HTTPException(status_code=400, detail="A filename is required")

    try:
        content = await file.read()
        document = document_store.add_document(
            file.filename,
            content,
            notebook_id=notebook_id,
            owner_id=user_id,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return StoredDocument(**document)


@app.get(
    "/api/documents",
    response_model=list[StoredDocument],
    dependencies=[Depends(require_identity)],
)
def list_documents(
    notebook_id: str | None = None,
    user_id: str = Depends(require_identity),
) -> list[StoredDocument]:
    return [
        StoredDocument(**document)
        for document in document_store.list_documents(
            notebook_id=notebook_id,
            owner_id=user_id,
        )
    ]


@app.get(
    "/api/documents/{document_id}/file",
    dependencies=[Depends(require_identity)],
)
def get_document_file(
    document_id: str,
    user_id: str = Depends(require_identity),
) -> FileResponse:
    try:
        path, filename = document_store.get_document_file(
            document_id,
            owner_id=user_id,
        )
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return FileResponse(
        path=path,
        filename=filename,
        media_type=media_type,
        content_disposition_type="inline",
    )


@app.delete(
    "/api/documents/{document_id}",
    status_code=204,
    dependencies=[Depends(require_identity)],
)
def delete_document(
    document_id: str,
    user_id: str = Depends(require_identity),
) -> None:
    try:
        document_store.delete_document(document_id, owner_id=user_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post(
    "/api/documents/search",
    response_model=SearchResponse,
    dependencies=[Depends(require_identity)],
)
def search_documents(
    request: StoredSearchRequest,
    user_id: str = Depends(require_identity),
) -> SearchResponse:
    try:
        chunks = document_store.get_chunks(
            document_ids=request.document_ids,
            notebook_id=request.notebook_id,
            owner_id=user_id,
        )
        if request.mode == "bm25":
            results = BM25Retriever(chunks).search(
                request.query,
                top_k=request.top_k,
            )
        else:
            candidates = HybridRetriever(
                chunks,
                vector_model=get_embedding_model(),
            ).search(
                request.query,
                top_k=request.candidate_k,
                candidate_k=request.candidate_k,
            )
            results = CrossEncoderReranker(
                model=get_reranking_model(),
            ).rerank(
                request.query,
                candidates,
                top_k=request.top_k,
            )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return SearchResponse(results=[SearchResult(**result) for result in results])


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    dependencies=[Depends(require_identity)],
)
def chat(
    request: ChatRequest,
    user_id: str = Depends(require_identity),
) -> ChatResponse:
    try:
        session = None
        history: list[dict[str, str]] = []
        if request.session_id is not None:
            session = document_store.get_chat_session(
                request.session_id,
                owner_id=user_id,
            )
            if (
                request.notebook_id is not None
                and request.notebook_id != session["notebook_id"]
            ):
                raise ValueError("Chat session does not belong to this notebook")
            request_notebook_id = session["notebook_id"]
            history = document_store.get_chat_history(
                request.session_id,
                owner_id=user_id,
            )
        else:
            request_notebook_id = request.notebook_id

        chunks = document_store.get_chunks(
            document_ids=request.document_ids,
            notebook_id=request_notebook_id,
            owner_id=user_id,
        )
        if request.mode == "bm25":
            retrieved = BM25Retriever(chunks).search(
                request.query,
                top_k=request.top_k,
            )
        else:
            candidates = HybridRetriever(
                chunks,
                vector_model=get_embedding_model(),
            ).search(
                request.query,
                top_k=request.candidate_k,
                candidate_k=request.candidate_k,
            )
            retrieved = CrossEncoderReranker(
                model=get_reranking_model(),
            ).rerank(
                request.query,
                candidates,
                top_k=request.top_k,
            )

        result = AnswerGenerator(get_language_model()).generate(
            request.query,
            retrieved,
            history=history,
        )
        if request.session_id is not None:
            document_store.add_chat_message(
                request.session_id,
                "user",
                request.query,
                owner_id=user_id,
            )
            document_store.add_chat_message(
                request.session_id,
                "assistant",
                result["answer"],
                owner_id=user_id,
            )
        for citation in result["citations"]:
            citation["source_url"] = build_source_url(citation)
    except (AnswerGenerationError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return ChatResponse(**result)


@app.post(
    "/api/search",
    response_model=SearchResponse,
    dependencies=[Depends(require_identity)],
)
def search(
    request: SearchRequest,
    _: str = Depends(require_identity),
) -> SearchResponse:
    chunks: list[dict[str, Any]] = [
        chunk.model_dump() for chunk in request.chunks
    ]

    try:
        results = BM25Retriever(chunks).search(
            request.query,
            top_k=request.top_k,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return SearchResponse(results=[SearchResult(**result) for result in results])
