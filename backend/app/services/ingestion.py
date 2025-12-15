import hashlib
import logging
import uuid
from pathlib import Path
from typing import List, Optional, Sequence
import numpy as np
from app.utils.loaders.pdf_loader import load_pdf
from app.utils.loaders.docx_loader import load_docx
from app.utils.loaders.txt_loader import load_txt
from app.utils.chunking import make_chunks, Chunk
from app.services.llm import embed_texts, EMBED_MODEL
from app.data.vector_store import VectorStore, StoredChunk
from app.api.schemas import IngestResult, ProofSpan

logger = logging.getLogger(__name__)

_CHUNK_TARGET_CHARS = 1400
_CHUNK_OVERLAP = 200


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _detect_loader(path: str):
    ext = Path(path).suffix.lower()
    if ext == ".pdf": return load_pdf
    if ext in [".docx"]: return load_docx
    return load_txt

def _ingest_single(path: str, store: VectorStore) -> IngestResult:
    logger.info("Starting ingestion for %s", path)
    loader = _detect_loader(path)
    logger.debug("Selected loader %s", loader.__name__)
    logger.info("Loading document contents")
    pages = loader(path)
    logger.info("Loaded %s with %d pages", Path(path).name, len(pages))
    doc_id = uuid.uuid4().hex[:12]
    store.add_document(
        doc_id,
        path=str(Path(path).resolve()),
        title=Path(path).name,
        info={"pages": len(pages)},
    )  # Add document to sql table
    logger.info("Registered document %s in metadata store", doc_id)

    # chunk
    # Slightly larger chunks and overlap improve coherence and recall
    chunks: List[Chunk] = make_chunks(
        doc_id,
        pages,
        target_chars=_CHUNK_TARGET_CHARS,
        overlap=_CHUNK_OVERLAP,
    )
    logger.info(
        "Created %d chunks (target_chars=%d, overlap=%d)",
        len(chunks),
        _CHUNK_TARGET_CHARS,
        _CHUNK_OVERLAP,
    )

    # embed
    texts = [c["text"] for c in chunks]
    hashes = [_hash_text(text) for text in texts]
    cached = store.get_cached_embeddings(hashes)
    logger.info(
        "Embedding %d chunks with model %s (cache hits=%d, misses=%d)",
        len(chunks),
        EMBED_MODEL,
        len(cached),
        len(hashes) - len(cached),
    )

    missing_indices: List[int] = [i for i, h in enumerate(hashes) if h not in cached]
    missing_texts = [texts[i] for i in missing_indices]

    if missing_texts:
        new_vectors = embed_texts(missing_texts)
        for idx, vec in zip(missing_indices, new_vectors):
            cached[hashes[idx]] = vec.astype("float32", copy=False)
        store.add_cached_embeddings(
            {hashes[idx]: cached[hashes[idx]] for idx in missing_indices}
        )
        logger.info("Cached %d new embeddings", len(missing_indices))
    elif not hashes:
        logger.info("No chunks to embed for %s", path)

    if hashes:
        vectors = np.stack([cached[h] for h in hashes]).astype("float32", copy=False)
    else:
        vectors = np.zeros((0, store.vector_dimension), dtype="float32")
    logger.info("Prepared embeddings with shape %s", getattr(vectors, "shape", None))

    # persist
    stored = [
        StoredChunk(
            chunk_id=c["chunk_id"],
            doc_id=c["doc_id"],
            page=c["page"],
            start=c["start"],
            end=c["end"],
            text=c["text"],
        )
        for c in chunks
    ]
    logger.info("Persisting chunks to vector store")
    store.add_chunks(stored, vectors)
    logger.info("Finished ingestion for %s (doc_id=%s)", path, doc_id)
    return IngestResult(doc_id=doc_id, num_chunks=len(chunks))


def ingest_documents(paths: Sequence[str], store: Optional[VectorStore] = None) -> List[IngestResult]:
    if not paths:
        logger.info("No paths provided for ingestion; skipping")
        return []
    store = store or VectorStore()
    results: List[IngestResult] = []
    for path in paths:
        results.append(_ingest_single(path, store))
    logger.info("Ingested %d document(s)", len(results))
    return results
