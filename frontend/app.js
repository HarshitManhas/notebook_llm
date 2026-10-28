const apiKey = document.querySelector("#api-key");
const notebookSelect = document.querySelector("#notebook-select");
const status = document.querySelector("#status");
const messages = document.querySelector("#messages");
let accessToken = localStorage.getItem("notebook_access_token") || "";

function headers(json = false) {
  const result = accessToken
    ? { Authorization: `Bearer ${accessToken}` }
    : { "X-API-Key": apiKey.value || "local-development-key" };
  if (json) result["Content-Type"] = "application/json";
  return result;
}

async function authenticate(path) {
  const email = document.querySelector("#email").value.trim();
  const password = document.querySelector("#password").value;
  if (!email || !password) return setStatus("Enter email and password.", true);
  try {
    const result = await api(path, {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
    accessToken = result.access_token;
    localStorage.setItem("notebook_access_token", accessToken);
    setStatus("Signed in.");
    await loadNotebooks();
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function api(path, options = {}) {
  const isJsonBody = typeof options.body === "string";
  const response = await fetch(path, {
    ...options,
    headers: { ...headers(isJsonBody), ...(options.headers || {}) },
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(body.detail)
      ? body.detail
          .map((item) => typeof item === "object" ? item.msg || JSON.stringify(item) : item)
          .join("; ")
      : typeof body.detail === "object"
        ? JSON.stringify(body.detail)
        : body.detail;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return body;
}

async function openSource(event, path) {
  event.preventDefault();
  try {
    const response = await fetch(path, { headers: headers() });
    if (!response.ok) throw new Error(`Source request failed (${response.status})`);
    const blob = await response.blob();
    window.open(URL.createObjectURL(blob), "_blank", "noopener");
  } catch (error) {
    setStatus(error.message, true);
  }
}

function setStatus(message, isError = false) {
  status.textContent = message;
  status.style.color = isError ? "#ff8d8d" : "";
}

async function loadNotebooks() {
  const notebooks = await api("/api/notebooks");
  notebookSelect.replaceChildren(new Option("Select a notebook", ""));
  for (const notebook of notebooks) {
    notebookSelect.add(new Option(
      `${notebook.name} (${notebook.document_count} documents)`,
      notebook.notebook_id,
    ));
  }
}

document.querySelector("#create-notebook").addEventListener("click", async () => {
  const name = document.querySelector("#notebook-name").value.trim();
  if (!name) return setStatus("Enter a notebook name.", true);
  try {
    await api("/api/notebooks", {
      method: "POST",
      body: JSON.stringify({ name }),
    });
    document.querySelector("#notebook-name").value = "";
    await loadNotebooks();
    setStatus("Notebook created.");
  } catch (error) {
    setStatus(error.message, true);
  }
});

document.querySelector("#register").addEventListener("click", () => {
  authenticate("/api/auth/register");
});

document.querySelector("#login").addEventListener("click", () => {
  authenticate("/api/auth/login");
});

document.querySelector("#upload-document").addEventListener("click", async () => {
  const file = document.querySelector("#document-file").files[0];
  if (!notebookSelect.value) return setStatus("Select a notebook first.", true);
  if (!file) return setStatus("Choose a document first.", true);
  const form = new FormData();
  form.append("file", file);
  try {
    await api(`/api/documents?notebook_id=${encodeURIComponent(notebookSelect.value)}`, {
      method: "POST",
      body: form,
    });
    await loadNotebooks();
    setStatus("Document uploaded.");
  } catch (error) {
    setStatus(error.message, true);
  }
});

document.querySelector("#chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = document.querySelector("#question").value.trim();
  if (!notebookSelect.value) return setStatus("Select a notebook first.", true);
  if (!question) return;
  messages.insertAdjacentHTML("beforeend", `<div class="message user">${question}</div>`);
  document.querySelector("#question").value = "";
  try {
    const result = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        query: question,
        notebook_id: notebookSelect.value,
        mode: "bm25",
        top_k: 5,
      }),
    });
    const citationText = result.citations.map((source) => {
      const link = source.source_url
        ? ` <a href="${source.source_url}" onclick="openSource(event, '${source.source_url}')">open source</a>`
        : "";
      return `[${source.source_id}] ${source.filename}${link}`;
    }).join(", ");
    messages.insertAdjacentHTML(
      "beforeend",
      `<div class="message">${result.answer}\n\nSources: ${citationText}</div>`,
    );
  } catch (error) {
    setStatus(error.message, true);
  }
});

loadNotebooks().catch((error) => setStatus(error.message, true));
