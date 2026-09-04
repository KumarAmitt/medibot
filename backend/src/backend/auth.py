from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.config import get_settings
from backend.rbac import ROLE_LABELS, ROLES

DEMO_USERS: dict[str, dict[str, str]] = {
    "dr.mehta": {
        "password": "doctor",
        "role": "doctor",
        "display_name": "Dr. Mehta",
    },
    "nurse.priya": {
        "password": "nurse",
        "role": "nurse",
        "display_name": "Nurse Priya",
    },
    "billing.ravi": {
        "password": "billing_executive",
        "role": "billing_executive",
        "display_name": "Ravi — Billing",
    },
    "tech.anand": {
        "password": "technician",
        "role": "technician",
        "display_name": "Anand — Technician",
    },
    "admin.sys": {
        "password": "admin",
        "role": "admin",
        "display_name": "System Admin",
    },
}

_bearer = HTTPBearer(auto_error=False)


def authenticate(username: str, password: str) -> dict[str, str]:
    user = DEMO_USERS.get(username.strip().lower())
    if not user or user["password"] != password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    return {
        "username": username.strip().lower(),
        "role": user["role"],
        "display_name": user["display_name"],
    }


def create_token(username: str, role: str) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": username, "role": role, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_token(token: str) -> dict[str, str]:
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired. Please log in again.",
        ) from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session token",
        ) from exc

    role = payload.get("role")
    username = payload.get("sub")
    if role not in ROLES or not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session token",
        )
    return {"username": username, "role": role}


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict[str, str]:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization token",
        )
    user = decode_token(credentials.credentials)
    user["display_name"] = ROLE_LABELS.get(user["role"], user["role"])
    return user
