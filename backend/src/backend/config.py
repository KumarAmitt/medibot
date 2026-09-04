from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = BACKEND_ROOT / "mediassist_data"
QDRANT_PATH = BACKEND_ROOT / "qdrant_data"
DB_PATH = DATA_DIR / "db" / "mediassist.db"

COLLECTION_NAME = "medibot_chunks"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"
DENSE_DIM = 384

HYBRID_CANDIDATES = 10
RERANK_TOP_K = 3


class Settings(BaseSettings):
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    jwt_secret: str = "medibot-dev-secret-change-in-production"
    jwt_expire_minutes: int = 480
    cors_origins: str = "*"

    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
