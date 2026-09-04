from __future__ import annotations

import logging
import re

from langchain_core.prompts import ChatPromptTemplate

from backend.config import HYBRID_CANDIDATES, RERANK_TOP_K
from backend.embeddings import get_embeddings
from backend.llm import extractive_answer, invoke_llm
from backend.models import Source
from backend.rbac import (
    SQL_RAG_ROLES,
    infer_restricted_collections,
    rbac_refusal_message,
    sql_refusal_message,
)
from backend.rerank import rerank
from backend.sql_rag import sql_rag_chain
from backend.vectorstore import hybrid_search

logger = logging.getLogger(__name__)

ANALYTICAL_RE = re.compile(
    r"\b(how many|count of|number of|total (number|amount|claims|tickets)|"
    r"average|avg\b|sum of|percentage|proportion|breakdown|"
    r"which (equipment )?category|which department|"
    r"most (open|escalated|claims|tickets)|last month|this month|"
    r"statistics|how much)\b",
    re.I,
)
SQL_DOMAIN_RE = re.compile(
    r"\b(claims?|tickets?|maintenance|insurer|claimed amount|"
    r"approved amount|equipment category|open tickets|escalated)\b",
    re.I,
)
DOC_HINT_RE = re.compile(
    r"\b(protocol|procedure|dosage|guideline|manual|handbook|policy|"
    r"sla|pre-auth|cannula|calibration steps|code of conduct|"
    r"leave|formulary|infection control)\b",
    re.I,
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are MediBot, the internal assistant for MediAssist Health Network.\n"
            "Answer ONLY from the provided source excerpts. If they are insufficient, say so.\n"
            "Never invent clinical, billing, or equipment details that are not in the excerpts.\n"
            "The user role is {role}. Restricted collections have already been filtered out "
            "at retrieval time — you cannot and must not speculate about them.\n"
            "Write a clear, practical answer. Mention the source document names you used.",
        ),
        (
            "human",
            "Question: {question}\n\nSource excerpts:\n{context}",
        ),
    ]
)


def is_analytical_question(question: str) -> bool:
    if DOC_HINT_RE.search(question) and not SQL_DOMAIN_RE.search(question):
        return False
    return bool(ANALYTICAL_RE.search(question))


def _format_context(chunks: list[dict]) -> str:
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{index}] Document: {chunk['source_document']} | "
            f"Section: {chunk['section_title']} | "
            f"Collection: {chunk['collection']} | "
            f"Type: {chunk.get('chunk_type', 'text')}\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(blocks)


def _sources_from_chunks(chunks: list[dict]) -> list[Source]:
    seen: set[tuple[str, str, str]] = set()
    sources: list[Source] = []
    for chunk in chunks:
        key = (chunk["source_document"], chunk["section_title"], chunk["collection"])
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            Source(
                source_document=chunk["source_document"],
                section_title=chunk["section_title"],
                collection=chunk["collection"],
            )
        )
    return sources


def hybrid_rag(question: str, role: str) -> tuple[str, list[Source], bool]:
    embeddings = get_embeddings()
    dense, sparse = embeddings.embed_query(question)
    candidates = hybrid_search(dense, sparse, role=role, limit=HYBRID_CANDIDATES)
    logger.info("Hybrid retrieval returned %s candidates for role=%s", len(candidates), role)

    restricted = infer_restricted_collections(question, role)
    if not candidates:
        return rbac_refusal_message(role, restricted), [], True

    top_chunks = rerank(question, candidates, top_k=RERANK_TOP_K)
    if not top_chunks:
        return rbac_refusal_message(role, restricted), [], True

    best_score = float(top_chunks[0].get("rerank_score") or 0.0)
    # FlashRank MiniLM scores are typically higher for true matches; a very weak
    # top hit plus a restricted-collection ask is treated as an access refusal.
    if restricted and best_score < 0.2:
        return rbac_refusal_message(role, restricted), [], True

    context = _format_context(top_chunks)
    try:
        answer = invoke_llm(
            ANSWER_PROMPT,
            {"question": question, "role": role, "context": context},
        )
    except Exception as exc:
        logger.warning("Groq answer failed (%s); using extractive fallback", exc)
        answer = extractive_answer(question, top_chunks)
    return answer, _sources_from_chunks(top_chunks), False


def answer_question(question: str, role: str) -> dict:
    if is_analytical_question(question):
        if role not in SQL_RAG_ROLES:
            return {
                "answer": sql_refusal_message(role),
                "sources": [],
                "retrieval_type": "sql_rag",
                "role": role,
                "blocked": True,
            }
        answer = sql_rag_chain(question)
        sources = [
            Source(
                source_document="mediassist.db",
                section_title="claims / maintenance_tickets",
                collection="billing" if "claim" in question.lower() else "equipment",
            )
        ]
        return {
            "answer": answer,
            "sources": sources,
            "retrieval_type": "sql_rag",
            "role": role,
            "blocked": False,
        }

    answer, sources, blocked = hybrid_rag(question, role)
    return {
        "answer": answer,
        "sources": sources,
        "retrieval_type": "hybrid_rag",
        "role": role,
        "blocked": blocked,
    }
