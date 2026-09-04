from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str
    display_name: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)


class Source(BaseModel):
    source_document: str
    section_title: str
    collection: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    retrieval_type: str
    role: str
    blocked: bool = False


class CollectionsResponse(BaseModel):
    role: str
    collections: list[str]
    collection_labels: dict[str, str]


class HealthResponse(BaseModel):
    status: str
    qdrant: bool
    chunk_count: int
    groq_configured: bool
