from __future__ import annotations

import logging
import uuid
from pathlib import Path

from qdrant_client.models import PointStruct

from backend.ssl_util import configure_model_downloads

configure_model_downloads()

from backend.config import (
    COLLECTION_NAME,
    DATA_DIR,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
)
from backend.embeddings import get_embeddings
from backend.rbac import access_roles_for_collection
from backend.vectorstore import chunk_count, ensure_collection, get_qdrant

logger = logging.getLogger(__name__)

COLLECTION_FOLDERS = ("general", "clinical", "nursing", "billing", "equipment")
SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown"}
_DOCLING_PDF_DISABLED = False


def _infer_chunk_type(chunk) -> str:
    items = getattr(getattr(chunk, "meta", None), "doc_items", None) or []
    labels = " ".join(str(getattr(item, "label", "")).lower() for item in items)
    if "table" in labels:
        return "table"
    if "code" in labels:
        return "code"
    if any(token in labels for token in ("title", "section_header", "heading")):
        return "heading"
    return "text"


def _section_title(chunk) -> str:
    headings = getattr(getattr(chunk, "meta", None), "headings", None) or []
    if headings:
        return str(headings[-1]).strip() or "Document"
    return "Document"


def _chunk_with_docling(path: Path) -> list[dict]:
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    document = converter.convert(str(path)).document
    try:
        from docling.chunking import HybridChunker

        chunker = HybridChunker()
        records: list[dict] = []
        for chunk in chunker.chunk(dl_doc=document):
            text = chunker.contextualize(chunk=chunk).strip()
            if not text:
                continue
            records.append(
                {
                    "text": text,
                    "section_title": _section_title(chunk),
                    "chunk_type": _infer_chunk_type(chunk),
                }
            )
        if records:
            return records
    except Exception:
        logger.warning("HybridChunker unavailable for %s — using Docling markdown export", path.name)

    markdown = document.export_to_markdown()
    return _chunk_markdown_text(markdown, path.stem.replace("_", " ").title())


def _split_by_size(text: str, max_chars: int = 1400) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            cut = text.rfind("\n", start, end)
            if cut <= start:
                cut = text.rfind(". ", start, end)
            if cut > start:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            parts.append(piece)
        start = end
    return parts


def _chunk_markdown_text(raw: str, default_title: str) -> list[dict]:
    sections: list[tuple[str, list[str], str]] = []
    heading = default_title
    heading_stack: list[str] = []
    buffer: list[str] = []
    chunk_type = "text"

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if not body:
            return
        title = heading_stack[-1] if heading_stack else heading
        prefix = " > ".join(heading_stack) if heading_stack else heading
        for part in _split_by_size(f"{prefix}\n\n{body}"):
            inferred = "table" if "|" in body and "---" in body else chunk_type
            if body.startswith("```") or "\n```" in body:
                inferred = "code"
            sections.append((part, [title], inferred))

    for line in raw.splitlines():
        if line.startswith("#"):
            flush()
            buffer = []
            title = line.lstrip("#").strip()
            level = len(line) - len(line.lstrip("#"))
            heading_stack = heading_stack[: max(level - 1, 0)]
            heading_stack.append(title)
            chunk_type = "heading"
        else:
            if chunk_type == "heading" and line.strip():
                chunk_type = "text"
            buffer.append(line)
    flush()
    return [
        {"text": text, "section_title": titles[-1] if titles else heading, "chunk_type": ctype}
        for text, titles, ctype in sections
    ]


def _chunk_markdown_fallback(path: Path) -> list[dict]:
    return _chunk_markdown_text(
        path.read_text(encoding="utf-8", errors="replace"),
        path.stem.replace("_", " ").title(),
    )


def _chunk_pdf_fallback(path: Path) -> list[dict]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    records: list[dict] = []
    current_heading = path.stem.replace("_", " ").title()
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if not body:
            return
        contextualized = f"{current_heading}\n\n{body}"
        for part in _split_by_size(contextualized):
            records.append(
                {
                    "text": part,
                    "section_title": current_heading,
                    "chunk_type": "table" if part.count("|") >= 4 else "text",
                }
            )

    for page in reader.pages:
        page_text = page.extract_text() or ""
        for line in page_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            looks_like_heading = (
                stripped.isupper()
                or (len(stripped) < 80 and stripped[:1].isdigit() and "." in stripped[:4])
            )
            if looks_like_heading and buffer:
                flush()
                buffer = []
                current_heading = stripped
            else:
                buffer.append(stripped)
        flush()
        buffer = []
    return records


def parse_document(path: Path) -> list[dict]:
    global _DOCLING_PDF_DISABLED
    is_pdf = path.suffix.lower() == ".pdf"
    if not (is_pdf and _DOCLING_PDF_DISABLED):
        try:
            chunks = _chunk_with_docling(path)
            if chunks:
                logger.info("Docling parsed %s into %s chunks", path.name, len(chunks))
                return chunks
        except Exception as exc:
            message = str(exc)
            if is_pdf and (
                "CERTIFICATE_VERIFY_FAILED" in message
                or "LocalEntryNotFoundError" in message
                or "huggingface" in message.lower()
            ):
                _DOCLING_PDF_DISABLED = True
                logger.warning(
                    "Docling PDF models are unavailable (Hugging Face SSL). "
                    "Using the hierarchical pypdf fallback for remaining PDFs."
                )
            else:
                logger.exception("Docling failed for %s — using fallback parser", path.name)

    if path.suffix.lower() in {".md", ".markdown"}:
        return _chunk_markdown_fallback(path)
    return _chunk_pdf_fallback(path)


def discover_documents() -> list[tuple[str, Path]]:
    documents: list[tuple[str, Path]] = []
    for collection in COLLECTION_FOLDERS:
        folder = DATA_DIR / collection
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                documents.append((collection, path))
    return documents


def ingest_all(*, force: bool = False) -> int:
    client = get_qdrant()
    ensure_collection(client)
    existing = chunk_count(client)
    if existing and not force:
        logger.info("Qdrant already has %s chunks — skip ingest (use --force to rebuild)", existing)
        return existing

    if force and existing:
        client.delete_collection(COLLECTION_NAME)
        ensure_collection(client)

    documents = discover_documents()
    if not documents:
        raise RuntimeError(f"No documents found under {DATA_DIR}")

    parsed: list[tuple[str, Path, list[dict]]] = []
    corpus: list[str] = []
    for collection, path in documents:
        chunks = parse_document(path)
        if not chunks:
            logger.warning("No chunks produced for %s", path.name)
            continue
        parsed.append((collection, path, chunks))
        corpus.extend(chunk["text"] for chunk in chunks)
        logger.info("Prepared %s chunks from %s [%s]", len(chunks), path.name, collection)

    embeddings = get_embeddings()
    embeddings.fit(corpus)

    points: list[PointStruct] = []
    for collection, path, chunks in parsed:
        roles = access_roles_for_collection(collection)
        texts = [chunk["text"] for chunk in chunks]
        dense_vectors = embeddings.embed_dense(texts)
        sparse_vectors = embeddings.embed_sparse(texts)
        for index, (chunk, dense, sparse) in enumerate(
            zip(chunks, dense_vectors, sparse_vectors, strict=True)
        ):
            points.append(
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{collection}:{path.name}:{index}")),
                    vector={
                        DENSE_VECTOR_NAME: dense,
                        SPARSE_VECTOR_NAME: sparse,
                    },
                    payload={
                        "text": chunk["text"],
                        "source_document": path.name,
                        "collection": collection,
                        "access_roles": roles,
                        "section_title": chunk["section_title"],
                        "chunk_type": chunk["chunk_type"],
                    },
                )
            )

    batch_size = 64
    for start in range(0, len(points), batch_size):
        client.upsert(collection_name=COLLECTION_NAME, points=points[start : start + batch_size])
        logger.info("Upserted %s / %s points", min(start + batch_size, len(points)), len(points))

    total = chunk_count(client)
    logger.info("Ingestion complete — %s chunks in %s", total, COLLECTION_NAME)
    return total


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import argparse

    parser = argparse.ArgumentParser(description="Ingest MediAssist documents into Qdrant")
    parser.add_argument("--force", action="store_true", help="Rebuild the collection from scratch")
    args = parser.parse_args()
    total = ingest_all(force=args.force)
    print(f"Indexed {total} chunks")


if __name__ == "__main__":
    main()
