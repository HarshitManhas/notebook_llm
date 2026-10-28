let accessToken = localStorage.getItem("notebook_access_token") || "";
let notebooks = [];
const selectedNotebookStorageKey = "notebook_selected_notebook_id";
let selectedNotebookId = localStorage.getItem(selectedNotebookStorageKey) || "";

const status = document.querySelector("#status");
const notebookList = document.querySelector("#notebook-list");
const notebookCount = document.querySelector("#notebook-count");
const selectedName = document.querySelector("#selected-name");
const selectedCount = document.querySelector("#selected-count");
const documentList = document.querySelector("#document-list");
const messages = document.querySelector("#messages");

function headers(json = false) {
  const result = accessToken
    ? { Authorization: `Bearer ${accessToken}` }
    : { "X-API-Key": "local-development-key" };
  if (json) result["Content-Type"] = "application/json";
  return result;
}

function setStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle("error", isError);
}

function readableError(body, fallback) {
  if (typeof body.detail === "string") return body.detail;
  if (Array.isArray(body.detail)) return body.detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
  return fallback;
}

async function openSource(event, path) {
  event.preventDefault();
  try {
    const response = await fetch(path, { headers: headers() });
    if (!response.ok) throw new Error(`Source request failed (${response.status})`);
    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);
    const overlay = document.createElement("div");
    overlay.className = "source-viewer";
    overlay.innerHTML = '<div class="source-viewer-header"><strong>Source preview</strong><button type="button" class="ghost-button">Close</button></div><iframe title="Source preview"></iframe>';
    const close = () => {
      URL.revokeObjectURL(objectUrl);
      overlay.remove();
    };
    overlay.querySelector("button").addEventListener("click", close);
    overlay.addEventListener("click", (overlayEvent) => {
      if (overlayEvent.target === overlay) close();
    });
    overlay.querySelector("iframe").src = objectUrl;
    document.body.append(overlay);
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { ...headers(typeof options.body === "string"), ...(options.headers || {}) },
  });
  const body = await response.json().catch(() => ({}));
  if (response.status === 401) {
    localStorage.removeItem("notebook_access_token");
    window.location.href = "/frontend/login.html";
    throw new Error("Your session has expired.");
  }
  if (!response.ok) throw new Error(readableError(body, `Request failed (${response.status})`));
  return body;
}

function renderNotebooks() {
  notebookList.replaceChildren();
  notebookCount.textContent = notebooks.length;
  if (!notebooks.length) {
    notebookList.innerHTML = '<p class="empty-list">Create your first notebook.</p>';
    return;
  }
  for (const notebook of notebooks) {
    const button = document.createElement("button");
    button.className = `notebook-item${notebook.notebook_id === selectedNotebookId ? " active" : ""}`;
    button.type = "button";
    button.innerHTML = `<strong></strong><small>${notebook.document_count} ${notebook.document_count === 1 ? "source" : "sources"}</small>`;
    button.querySelector("strong").textContent = notebook.name;
    button.addEventListener("click", () => selectNotebook(notebook.notebook_id));
    notebookList.append(button);
  }
}

async function loadNotebooks() {
  notebooks = await api("/api/notebooks");
  if (!notebooks.some((notebook) => notebook.notebook_id === selectedNotebookId)) {
    selectedNotebookId = notebooks[0]?.notebook_id || "";
  }
  renderNotebooks();
  if (selectedNotebookId) await selectNotebook(selectedNotebookId, false);
  else renderDocuments([]);
}

async function selectNotebook(notebookId, announce = true) {
  selectedNotebookId = notebookId;
  if (selectedNotebookId) {
    localStorage.setItem(selectedNotebookStorageKey, selectedNotebookId);
  } else {
    localStorage.removeItem(selectedNotebookStorageKey);
  }
  renderNotebooks();
  const notebook = notebooks.find((item) => item.notebook_id === notebookId);
  selectedName.textContent = notebook?.name || "Select a notebook";
  selectedCount.textContent = notebook ? `${notebook.document_count} sources` : "";
  await loadDocuments();
  if (announce) setStatus(`Selected ${notebook.name}.`);
}

async function loadDocuments() {
  const documents = await api(`/api/documents?notebook_id=${encodeURIComponent(selectedNotebookId)}`);
  renderDocuments(documents);
}

function renderDocuments(documents) {
  documentList.replaceChildren();
  if (!selectedNotebookId) {
    documentList.innerHTML = '<p class="empty-list">Select a notebook to see its sources.</p>';
    return;
  }
  if (!documents.length) {
    documentList.innerHTML = '<p class="empty-list">No sources yet. Upload a document above.</p>';
    return;
  }
  for (const source of documents) {
    const row = document.createElement("div");
    row.className = "document-row";
    row.innerHTML = '<span class="file-icon">▧</span><div class="document-info"><strong></strong><small></small></div><button class="view-button" type="button">View</button><button class="delete-button" type="button">Remove</button>';
    row.querySelector("strong").textContent = source.filename;
    row.querySelector("small").textContent = `${source.chunk_count} indexed ${source.chunk_count === 1 ? "passage" : "passages"}`;
    row.querySelector(".view-button").addEventListener("click", (event) => {
      openSource(event, `/api/documents/${source.document_id}/file`);
    });
    row.querySelector(".delete-button").addEventListener("click", () => removeDocument(source));
    documentList.append(row);
  }
}

async function removeDocument(source) {
  if (!window.confirm(`Remove "${source.filename}" from your workspace?`)) return;
  try {
    await api(`/api/documents/${source.document_id}`, { method: "DELETE" });
    await loadNotebooks();
    setStatus("Source removed.");
  } catch (error) {
    setStatus(error.message, true);
  }
}

document.querySelector("#notebook-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = document.querySelector("#notebook-name");
  if (!input.value.trim()) return;
  try {
    const notebook = await api("/api/notebooks", {
      method: "POST",
      body: JSON.stringify({ name: input.value.trim() }),
    });
    input.value = "";
    selectedNotebookId = notebook.notebook_id;
    await loadNotebooks();
    setStatus("Notebook created.");
  } catch (error) {
    setStatus(error.message, true);
  }
});

document.querySelector("#upload-document").addEventListener("click", async () => {
  const file = document.querySelector("#document-file").files[0];
  if (!selectedNotebookId) return setStatus("Select a notebook first.", true);
  if (!file) return setStatus("Choose a document first.", true);
  const form = new FormData();
  form.append("file", file);
  try {
    await api(`/api/documents?notebook_id=${encodeURIComponent(selectedNotebookId)}`, { method: "POST", body: form });
    document.querySelector("#document-file").value = "";
    await loadNotebooks();
    setStatus("Source uploaded and indexed.");
  } catch (error) {
    setStatus(error.message, true);
  }
});

document.querySelector("#chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = document.querySelector("#question");
  if (!selectedNotebookId) return setStatus("Select a notebook first.", true);
  if (!input.value.trim()) return;
  const question = input.value.trim();
  input.value = "";
  const userMessage = document.createElement("div");
  userMessage.className = "message user";
  userMessage.textContent = question;
  messages.querySelector(".empty-chat")?.remove();
  messages.append(userMessage);
  try {
    const result = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ query: question, notebook_id: selectedNotebookId, mode: "bm25", top_k: 5 }),
    });
    const answer = document.createElement("div");
    answer.className = "message assistant";
    answer.textContent = result.answer;
    if (result.citations.length) {
      const citations = document.createElement("div");
      citations.className = "citation-list";
      citations.append("Sources: ");
      result.citations.forEach((source, index) => {
        const link = document.createElement("a");
        link.href = source.source_url || "#";
        link.textContent = `[${source.source_id}] ${source.filename}`;
        if (source.source_url) {
          link.addEventListener("click", (event) => openSource(event, source.source_url));
        }
        citations.append(link);
        if (index < result.citations.length - 1) citations.append(", ");
      });
      answer.append(citations);
    }
    messages.append(answer);
    messages.scrollTop = messages.scrollHeight;
  } catch (error) {
    setStatus(error.message, true);
  }
});

document.querySelector("#logout").addEventListener("click", () => {
  localStorage.removeItem("notebook_access_token");
  localStorage.removeItem(selectedNotebookStorageKey);
  window.location.href = "/frontend/login.html";
});

loadNotebooks().catch((error) => setStatus(error.message, true));
