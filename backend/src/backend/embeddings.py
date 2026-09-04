from __future__ import annotations

import logging
from functools import lru_cache

from qdrant_client.models import SparseVector

from backend.config import DENSE_MODEL, SPARSE_MODEL
from backend.local_text import BM25Index, dense_embed
from backend.ssl_util import configure_model_downloads

configure_model_downloads()

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Dense + sparse embeddings stored together in Qdrant.

    Prefers FastEmbed (BGE + Qdrant/bm25) when Hugging Face is reachable.
    Falls back to hashed dense vectors + corpus BM25 so ingest still works
    on networks that block model downloads.
    """

    def __init__(self) -> None:
        self.backend = "local"
        self._dense = None
        self._sparse = None
        self._bm25 = BM25Index.load()
        import os

        use_fastembed = os.getenv("MEDIBOT_FASTEMBED", "").lower() in {"1", "true", "yes"}
        if use_fastembed:
            try:
                from fastembed import SparseTextEmbedding, TextEmbedding

                self._dense = TextEmbedding(model_name=DENSE_MODEL)
                self._sparse = SparseTextEmbedding(model_name=SPARSE_MODEL)
                self.backend = "fastembed"
                logger.info("Using FastEmbed dense=%s sparse=%s", DENSE_MODEL, SPARSE_MODEL)
            except Exception as exc:
                logger.warning(
                    "FastEmbed unavailable (%s). Using local BM25 + hashed dense vectors.",
                    exc,
                )
        else:
            logger.info("Using local BM25 + hashed dense vectors (set MEDIBOT_FASTEMBED=1 to use FastEmbed)")

    def fit(self, texts: list[str]) -> None:
        if self.backend == "local":
            self._bm25.fit(texts)
            logger.info("Fitted local BM25 on %s chunks", len(texts))

    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        if self._dense is not None:
            return [vector.tolist() for vector in self._dense.embed(texts)]
        return [dense_embed(text) for text in texts]

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        if self._sparse is not None:
            vectors: list[SparseVector] = []
            for vector in self._sparse.embed(texts):
                vectors.append(
                    SparseVector(
                        indices=vector.indices.tolist(),
                        values=vector.values.tolist(),
                    )
                )
            return vectors
        return [self._bm25.document_vector(text) for text in texts]

    def embed_query(self, text: str) -> tuple[list[float], SparseVector]:
        dense = self.embed_dense([text])[0]
        if self._sparse is not None:
            sparse = self.embed_sparse([text])[0]
        else:
            if not self._bm25.document_count:
                self._bm25 = BM25Index.load()
            sparse = self._bm25.query_vector(text)
        return dense, sparse


@lru_cache
def get_embeddings() -> EmbeddingService:
    return EmbeddingService()
