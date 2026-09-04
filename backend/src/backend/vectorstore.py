from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PayloadSchemaType,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from backend.config import (
    COLLECTION_NAME,
    DENSE_DIM,
    DENSE_VECTOR_NAME,
    HYBRID_CANDIDATES,
    QDRANT_PATH,
    SPARSE_VECTOR_NAME,
)


@lru_cache
def get_qdrant() -> QdrantClient:
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(QDRANT_PATH))


def ensure_collection(client: QdrantClient | None = None) -> None:
    client = client or get_qdrant()
    if client.collection_exists(COLLECTION_NAME):
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(),
        },
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="access_roles",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="collection",
        field_schema=PayloadSchemaType.KEYWORD,
    )


def chunk_count(client: QdrantClient | None = None) -> int:
    client = client or get_qdrant()
    if not client.collection_exists(COLLECTION_NAME):
        return 0
    return int(client.count(COLLECTION_NAME, exact=True).count)


def rbac_filter(role: str) -> Filter:
    """Qdrant-level metadata filter: only chunks whose access_roles contain this role."""
    return Filter(
        must=[FieldCondition(key="access_roles", match=MatchValue(value=role))]
    )


def hybrid_search(
    dense: list[float],
    sparse: SparseVector,
    role: str,
    limit: int = HYBRID_CANDIDATES,
) -> list[dict]:
    """
    Single Qdrant hybrid query: dense + BM25 prefetches fused with RRF.
    The RBAC filter is applied on both prefetches and the fusion query so
    restricted chunks never leave the vector store.
    """
    client = get_qdrant()
    ensure_collection(client)
    access_filter = rbac_filter(role)
    if not sparse.indices:
        sparse = SparseVector(indices=[0], values=[0.0])

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(
                query=dense,
                using=DENSE_VECTOR_NAME,
                filter=access_filter,
                limit=limit,
            ),
            Prefetch(
                query=sparse,
                using=SPARSE_VECTOR_NAME,
                filter=access_filter,
                limit=limit,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        query_filter=access_filter,
        limit=limit,
        with_payload=True,
    )

    hits: list[dict] = []
    for point in results.points:
        payload = point.payload or {}
        hits.append(
            {
                "id": str(point.id),
                "score": float(point.score or 0.0),
                "text": payload.get("text", ""),
                "source_document": payload.get("source_document", ""),
                "collection": payload.get("collection", ""),
                "access_roles": payload.get("access_roles", []),
                "section_title": payload.get("section_title", ""),
                "chunk_type": payload.get("chunk_type", "text"),
            }
        )
    return hits
