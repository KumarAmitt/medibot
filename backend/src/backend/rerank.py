from __future__ import annotations

import json
import logging
import re
from functools import lru_cache

from backend.config import RERANK_TOP_K
from backend.local_text import lexical_cross_score
from backend.ssl_util import configure_model_downloads

configure_model_downloads()

logger = logging.getLogger(__name__)


@lru_cache
def _ranker():
    from flashrank import Ranker

    return Ranker(model_name="ms-marco-MiniLM-L-12-v2")


def _rerank_flashrank(question: str, candidates: list[dict], top_k: int) -> list[dict] | None:
    import os

    if os.getenv("MEDIBOT_FLASHRANK", "").lower() not in {"1", "true", "yes"}:
        return None
    try:
        from flashrank import RerankRequest

        passages = [
            {"id": index, "text": candidate["text"], "meta": candidate}
            for index, candidate in enumerate(candidates)
        ]
        ranked = _ranker().rerank(RerankRequest(query=question, passages=passages))
        return _take(ranked, candidates, top_k, score_key="score")
    except Exception as exc:
        logger.warning("FlashRank unavailable (%s)", exc)
        return None


def _rerank_groq(question: str, candidates: list[dict], top_k: int) -> list[dict] | None:
    """Joint (query, passage) scoring via Groq — used when a local cross-encoder cannot load."""
    try:
        from langchain_core.prompts import ChatPromptTemplate

        from backend.llm import invoke_llm

        blocks = []
        for index, candidate in enumerate(candidates):
            excerpt = candidate["text"][:1200]
            blocks.append(f"[{index}] {excerpt}")
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a relevance reranker. Score each passage jointly against the "
                    "question from 0 to 1. Return ONLY a JSON array of "
                    '{{"id": int, "score": float}} objects, one per passage.',
                ),
                ("human", "Question: {question}\n\nPassages:\n{passages}"),
            ]
        )
        raw = invoke_llm(
            prompt,
            {"question": question, "passages": "\n\n".join(blocks)},
            temperature=0,
        )
        match = re.search(r"\[[\s\S]*\]", raw)
        if not match:
            return None
        ranked = json.loads(match.group(0))
        return _take(ranked, candidates, top_k, score_key="score")
    except Exception as exc:
        logger.warning("Groq rerank failed (%s)", exc)
        return None


def _rerank_lexical(question: str, candidates: list[dict], top_k: int) -> list[dict]:
    scored = []
    for candidate in candidates:
        item = dict(candidate)
        item["rerank_score"] = lexical_cross_score(question, candidate["text"])
        scored.append(item)
    scored.sort(key=lambda item: item["rerank_score"], reverse=True)
    logger.info(
        "Lexical cross-scores: %s",
        [
            (item.get("source_document"), item.get("section_title"), round(item["rerank_score"], 4))
            for item in scored
        ],
    )
    return scored[:top_k]


def _take(ranked: list[dict], candidates: list[dict], top_k: int, score_key: str) -> list[dict]:
    logger.info(
        "Reranker scores: %s",
        [
            (
                (item.get("meta") or candidates[int(item["id"])]).get("source_document"),
                (item.get("meta") or candidates[int(item["id"])]).get("section_title"),
                round(float(item.get(score_key, 0.0)), 4),
            )
            for item in ranked
        ],
    )
    selected: list[dict] = []
    for item in ranked[:top_k]:
        source = item.get("meta") or candidates[int(item["id"])]
        chunk = dict(source)
        chunk["rerank_score"] = float(item.get(score_key, 0.0))
        selected.append(chunk)
    return selected


def rerank(question: str, candidates: list[dict], top_k: int = RERANK_TOP_K) -> list[dict]:
    """Score each candidate jointly with the query, then keep only the top-k."""
    if not candidates:
        return []
    for strategy in (_rerank_flashrank, _rerank_groq):
        ranked = strategy(question, candidates, top_k)
        if ranked:
            return ranked
    return _rerank_lexical(question, candidates, top_k)
