from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from qdrant_client.models import SparseVector

from backend.config import DENSE_DIM, QDRANT_PATH

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9.+%/_-]{0,}", re.I)
STATS_PATH = QDRANT_PATH / "bm25_stats.json"
K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if token.strip()]


def term_index(term: str) -> int:
    digest = hashlib.md5(term.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def dense_embed(text: str, dim: int = DENSE_DIM) -> list[float]:
    """Signed feature-hashing over tokens and character trigrams (no model download)."""
    tokens = tokenize(text)
    features = list(tokens)
    compact = re.sub(r"\s+", " ", text.lower())
    for index in range(max(len(compact) - 2, 0)):
        features.append(compact[index : index + 3])

    vector = [0.0] * dim
    for feature in features:
        hashed = int(hashlib.md5(feature.encode("utf-8")).hexdigest(), 16)
        slot = hashed % dim
        sign = 1.0 if (hashed >> 17) & 1 else -1.0
        vector[slot] += sign

    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class BM25Index:
    def __init__(self) -> None:
        self.document_count = 0
        self.avg_length = 1.0
        self.df: dict[str, int] = {}

    def fit(self, texts: list[str]) -> None:
        lengths: list[int] = []
        df: Counter[str] = Counter()
        for text in texts:
            tokens = tokenize(text)
            lengths.append(len(tokens) or 1)
            df.update(set(tokens))
        self.document_count = len(texts)
        self.avg_length = (sum(lengths) / len(lengths)) if lengths else 1.0
        self.df = dict(df)
        self.save()

    def idf(self, term: str) -> float:
        document_frequency = self.df.get(term, 0)
        return math.log(
            (self.document_count - document_frequency + 0.5) / (document_frequency + 0.5) + 1.0
        )

    def document_vector(self, text: str) -> SparseVector:
        tokens = tokenize(text)
        length = len(tokens) or 1
        counts = Counter(tokens)
        indices: list[int] = []
        values: list[float] = []
        for term, tf in counts.items():
            saturation = (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * length / self.avg_length))
            weight = saturation * self.idf(term)
            if weight <= 0:
                continue
            indices.append(term_index(term))
            values.append(float(weight))
        return SparseVector(indices=indices, values=values)

    def query_vector(self, text: str) -> SparseVector:
        indices: list[int] = []
        values: list[float] = []
        for term in set(tokenize(text)):
            weight = self.idf(term)
            if weight <= 0:
                continue
            indices.append(term_index(term))
            values.append(float(weight))
        return SparseVector(indices=indices, values=values)

    def save(self) -> None:
        QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        STATS_PATH.write_text(
            json.dumps(
                {
                    "document_count": self.document_count,
                    "avg_length": self.avg_length,
                    "df": self.df,
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls) -> BM25Index:
        instance = cls()
        if not STATS_PATH.exists():
            return instance
        payload = json.loads(STATS_PATH.read_text(encoding="utf-8"))
        instance.document_count = int(payload.get("document_count") or 0)
        instance.avg_length = float(payload.get("avg_length") or 1.0)
        instance.df = {str(key): int(value) for key, value in (payload.get("df") or {}).items()}
        return instance


def lexical_cross_score(query: str, passage: str) -> float:
    """Joint query–passage score used when a neural cross-encoder is unavailable."""
    query_tokens = tokenize(query)
    passage_tokens = tokenize(passage)
    if not query_tokens or not passage_tokens:
        return 0.0
    query_set = set(query_tokens)
    passage_set = set(passage_tokens)
    overlap = query_set & passage_set
    precision = len(overlap) / len(passage_set)
    recall = len(overlap) / len(query_set)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    query_lower = query.lower()
    passage_lower = passage.lower()
    phrase_bonus = 0.25 if query_lower in passage_lower else 0.0
    bigrams = set(zip(query_tokens, query_tokens[1:]))
    passage_bigrams = set(zip(passage_tokens, passage_tokens[1:]))
    if bigrams:
        phrase_bonus += 0.35 * (len(bigrams & passage_bigrams) / len(bigrams))
    return f1 + phrase_bonus
