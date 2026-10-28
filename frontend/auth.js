const status = document.querySelector("#status");
const apiKeyInput = document.querySelector("#api-key");

function setStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle("error", isError);
}

function requestHeaders() {
  const key = apiKeyInput?.value || "local-development-key";
  return { "X-API-Key": key, "Content-Type": "application/json" };
}

async function submitAuth(path) {
  const email = document.querySelector("#email").value.trim();
  const password = document.querySelector("#password").value;
  if (!email || !password) {
    setStatus("Enter your email and password.", true);
    return;
  }
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: requestHeaders(),
      body: JSON.stringify({ email, password }),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(typeof body.detail === "string" ? body.detail : "Authentication failed");
    }
    localStorage.setItem("notebook_access_token", body.access_token);
    window.location.href = "/frontend/workspace.html";
  } catch (error) {
    setStatus(error.message, true);
  }
}

const form = document.querySelector("#login-form") || document.querySelector("#register-form");
if (form) {
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    submitAuth(form.id === "login-form" ? "/api/auth/login" : "/api/auth/register");
  });
}
