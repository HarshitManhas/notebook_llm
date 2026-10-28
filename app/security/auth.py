import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Annotated

from fastapi import Header, HTTPException, status


def require_api_key(
    x_api_key: str | None = Header(default=None),
) -> None:
    expected_key = os.getenv("NOTEBOOK_API_KEY", "local-development-key")
    if not x_api_key or x_api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


def hash_password(password: str) -> str:
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt, 200_000
    )
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, digest_hex = stored_hash.split("$", 1)
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode(),
            bytes.fromhex(salt_hex),
            200_000,
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def _token_secret() -> bytes:
    return os.getenv(
        "NOTEBOOK_TOKEN_SECRET",
        "local-development-secret",
    ).encode()


def create_access_token(user_id: str, expires_in: int = 86_400) -> str:
    payload = {"sub": user_id, "exp": int(time.time()) + expires_in}
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode()
    ).decode().rstrip("=")
    signature = hmac.new(
        _token_secret(), encoded.encode(), hashlib.sha256
    ).hexdigest()
    return f"{encoded}.{signature}"


def get_user_from_token(token: str) -> str:
    try:
        encoded, signature = token.split(".", 1)
        expected = hmac.new(
            _token_secret(), encoded.encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError
        payload = json.loads(
            base64.urlsafe_b64decode(
                encoded + "=" * (-len(encoded) % 4)
            )
        )
        if int(payload["exp"]) < int(time.time()):
            raise ValueError
        return str(payload["sub"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
        ) from None


def require_identity(
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> str:
    if authorization and authorization.startswith("Bearer "):
        return get_user_from_token(authorization[7:])

    expected_key = os.getenv("NOTEBOOK_API_KEY", "local-development-key")
    if x_api_key == expected_key:
        return "local-development-user"

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
    )
