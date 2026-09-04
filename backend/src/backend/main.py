from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from backend.ssl_util import configure_model_downloads

configure_model_downloads()

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.auth import authenticate, create_token, get_current_user
from backend.config import get_settings
from backend.models import (
    ChatRequest,
    ChatResponse,
    CollectionsResponse,
    HealthResponse,
    LoginRequest,
    LoginResponse,
)
from backend.rag import answer_question
from backend.rbac import COLLECTION_LABELS, ROLES, collections_for_role
from backend.vectorstore import chunk_count, ensure_collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_collection()
    count = chunk_count()
    if count == 0:
        logger.warning(
            "Qdrant collection is empty. Run `uv run medibot-ingest` before chatting."
        )
    else:
        logger.info("Qdrant ready with %s chunks", count)
    yield


app = FastAPI(
    title="MediBot",
    description="Role-aware hybrid RAG assistant for MediAssist Health Network",
    version="1.0.0",
    lifespan=lifespan,
)

settings = get_settings()
origins = (
    ["*"]
    if settings.cors_origins.strip() == "*"
    else [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False if origins == ["*"] else True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        count = chunk_count()
        qdrant_ok = True
    except Exception:
        logger.exception("Qdrant health check failed")
        count = 0
        qdrant_ok = False
    groq_ok = bool(get_settings().groq_api_key)
    return HealthResponse(
        status="ok" if qdrant_ok and groq_ok else "degraded",
        qdrant=qdrant_ok,
        chunk_count=count,
        groq_configured=groq_ok,
    )


@app.post("/login", response_model=LoginResponse)
def login(body: LoginRequest) -> LoginResponse:
    user = authenticate(body.username, body.password)
    token = create_token(user["username"], user["role"])
    return LoginResponse(
        access_token=token,
        role=user["role"],
        username=user["username"],
        display_name=user["display_name"],
    )


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, user: dict[str, str] = Depends(get_current_user)) -> ChatResponse:
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = answer_question(question, user["role"])
    return ChatResponse(**result)


@app.get("/collections/{role}", response_model=CollectionsResponse)
def collections(role: str) -> CollectionsResponse:
    if role not in ROLES:
        raise HTTPException(status_code=404, detail=f"Unknown role: {role}")
    return CollectionsResponse(
        role=role,
        collections=collections_for_role(role),
        collection_labels=COLLECTION_LABELS,
    )


def run() -> None:
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)


def main() -> None:
    run()
