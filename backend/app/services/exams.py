from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from app.data.vector_store import VectorStore
from app.data.db_repository import StoredExam


@dataclass
class ExamWorkspace:
    """
    Local-first exam workspace (like a ChatGPT thread):
    - binds together a set of documents (doc_ids)
    - has a persistent event log
    """

    exam: StoredExam
    doc_ids: List[str]


class ImmutableExamError(ValueError):
    """Raised when a caller attempts to mutate an already-bootstrapped exam."""

    def __init__(self, exam_id: str, message: Optional[str] = None) -> None:
        self.exam_id = exam_id
        super().__init__(
            message
            or (
                "Exam content is immutable after initial bootstrap ingestion. "
                "Create a new exam to ingest additional documents."
            )
        )


def create_exam(
    *,
    store: Optional[VectorStore] = None,
    user_id: str,
    title: str,
    mode: str = "mastery",
    info: Optional[Dict[str, Any]] = None,
) -> str:
    store = store or VectorStore()
    return store.db.create_exam(user_id=user_id, title=title, mode=mode, info=info)


def ensure_exam_ingest_allowed(
    *,
    store: Optional[VectorStore] = None,
    exam_id: str,
) -> None:
    """
    Enforce immutable exam policy:
    - first bootstrap ingest is allowed when exam has no attached docs
    - re-ingest / add-documents on existing exam is rejected
    """
    store = store or VectorStore()
    exam = store.db.get_exam(exam_id)
    if exam is None:
        raise ValueError(f"Exam not found: {exam_id}")
    existing_doc_ids = store.db.list_exam_documents(exam_id=exam_id)
    if existing_doc_ids:
        raise ImmutableExamError(exam_id=exam_id)


def load_exam(*, store: Optional[VectorStore] = None, exam_id: str) -> ExamWorkspace:
    store = store or VectorStore()
    exam = store.db.get_exam(exam_id)
    if exam is None:
        raise ValueError(f"Exam not found: {exam_id}")
    doc_ids = store.db.list_exam_documents(exam_id=exam_id)
    return ExamWorkspace(exam=exam, doc_ids=doc_ids)


def attach_documents(
    *,
    store: Optional[VectorStore] = None,
    exam_id: str,
    doc_ids: Sequence[str],
) -> None:
    store = store or VectorStore()
    ensure_exam_ingest_allowed(store=store, exam_id=exam_id)
    store.db.attach_documents_to_exam(exam_id=exam_id, doc_ids=doc_ids)


def log_event(
    *,
    store: Optional[VectorStore] = None,
    user_id: str,
    exam_id: str,
    type: str,
    payload: Optional[Dict[str, Any]] = None,
) -> str:
    store = store or VectorStore()
    return store.db.add_event(user_id=user_id, exam_id=exam_id, type=type, payload=payload)


