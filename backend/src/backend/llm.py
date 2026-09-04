from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from backend.config import get_settings
from backend.local_text import tokenize

logger = logging.getLogger(__name__)


def _http_client():
    try:
        import certifi
        import httpx

        return httpx.Client(verify=certifi.where(), timeout=60.0)
    except Exception:
        return None


@lru_cache
def get_chat_model(temperature: float = 0.1) -> ChatGroq:
    settings = get_settings()
    kwargs = {
        "api_key": settings.groq_api_key,
        "model": settings.groq_model,
        "temperature": temperature,
    }
    client = _http_client()
    if client is not None:
        kwargs["http_client"] = client
    return ChatGroq(**kwargs)


def invoke_llm(prompt: ChatPromptTemplate, variables: dict, temperature: float = 0.1) -> str:
    chain = prompt | get_chat_model(temperature) | StrOutputParser()
    return chain.invoke(variables).strip()


def extractive_answer(question: str, chunks: list[dict]) -> str:
    """Compose a cited answer from retrieved chunks when Groq is unreachable."""
    query_tokens = set(tokenize(question))
    sentences: list[tuple[float, str, dict]] = []
    for chunk in chunks:
        for raw in chunk["text"].replace("\n", " ").split(". "):
            sentence = raw.strip(" .")
            if len(sentence) < 40:
                continue
            tokens = set(tokenize(sentence))
            score = len(query_tokens & tokens) / max(len(query_tokens), 1)
            sentences.append((score, sentence, chunk))
    sentences.sort(key=lambda item: item[0], reverse=True)
    picked = []
    seen = set()
    for score, sentence, chunk in sentences:
        if score <= 0 or sentence in seen:
            continue
        seen.add(sentence)
        picked.append((sentence, chunk))
        if len(picked) == 4:
            break
    if not picked:
        picked = [(chunks[0]["text"][:500], chunks[0])]

    lines = [
        "Based on your authorised sources:",
        "",
    ]
    for sentence, chunk in picked:
        lines.append(f"- {sentence}. ({chunk['source_document']} — {chunk['section_title']})")
    return "\n".join(lines)
