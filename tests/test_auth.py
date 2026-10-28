from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.security.auth import require_api_key


def test_valid_api_key_is_accepted(monkeypatch) -> None:
    monkeypatch.setenv("NOTEBOOK_API_KEY", "test-key")
    app = FastAPI()
    app.get("/protected", dependencies=[])(require_api_key)

    # Exercise the dependency directly because route registration is not
    # needed to verify its authentication behavior.
    require_api_key("test-key")


def test_invalid_api_key_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("NOTEBOOK_API_KEY", "test-key")

    try:
        require_api_key("wrong-key")
    except Exception as error:
        assert getattr(error, "status_code", None) == 401
    else:
        raise AssertionError("Invalid API key was accepted")
