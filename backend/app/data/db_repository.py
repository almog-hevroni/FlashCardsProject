"""
Database Repository Layer using SQLAlchemy ORM

This module provides the high-level database interface for the FlashCards application.
It uses SQLAlchemy ORM for all database operations, making it easy to
switch between SQLite (development) and PostgreSQL (cloud).

The dataclasses (StoredChunk, StoredExam, etc.) are kept for backwards
compatibility with existing code.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Iterable, Sequence
from dataclasses import dataclass
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select, delete, and_
from sqlalchemy.orm import Session

from app.data.db_engine import get_db, init_db, SessionLocal, DATABASE_URL
from app.data.models import (
    User, Document, Chunk, EmbeddingCache, VectorIndexMap,
    Exam, ExamDocument, Topic, TopicChunk, TopicEvidence, TopicVector,
    Card, CardProof, CardReview, Event, QuestionIndexEntry,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class StoredChunk:
    chunk_id: str
    doc_id: str
    page: int
    start: int
    end: int
    text: str


@dataclass
class StoredExam:
    exam_id: str
    user_id: str
    title: str
    mode: str
    created_at: str
    updated_at: str
    info: Dict[str, Any]


@dataclass
class StoredTopic:
    topic_id: str
    exam_id: str
    label: str
    created_at: str
    info: Dict[str, Any]


@dataclass
class StoredCard:
    card_id: str
    exam_id: str
    topic_id: str
    question: str
    answer: str
    difficulty: int
    created_at: str
    status: str
    info: Dict[str, Any]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _datetime_to_str(dt: Optional[datetime]) -> str:
    """Convert datetime to ISO string."""
    if dt is None:
        return ""
    return dt.isoformat() if isinstance(dt, datetime) else str(dt)


# =============================================================================
# DBRepository - Main Database Repository Class
# =============================================================================

class DBRepository:
    """
    Database repository layer using SQLAlchemy ORM.
    
    Works with any SQLAlchemy-supported database (SQLite, PostgreSQL, MySQL).
    
    Note: The actual database connection is configured via the DATABASE_URL
    environment variable. The db_path parameter is kept for backwards
    compatibility but the actual path is determined by DATABASE_URL or
    VECTOR_STORE_PATH environment variables.
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to database file (kept for backwards compatibility).
                     Actual path determined by DATABASE_URL or VECTOR_STORE_PATH.
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize tables using SQLAlchemy models
        init_db()
        
        # Log the actual database being used
        if DATABASE_URL.startswith("sqlite"):
            actual_path = DATABASE_URL.replace("sqlite:///", "")
            if str(db_path) != actual_path:
                logger.debug(
                    "Note: DBRepository initialized with path=%s but actual DB is %s "
                    "(controlled by DATABASE_URL/VECTOR_STORE_PATH env vars)",
                    db_path, actual_path
                )
        logger.info("SQLAlchemy database initialized")
    
    def _get_session(self) -> Session:
        """Get a new database session."""
        return SessionLocal()
    
    def close(self):
        """Close the database connection."""
        # SQLAlchemy handles connection pooling, nothing to do here
        pass
    
    # =========================================================================
    # DOCUMENT METHODS
    # =========================================================================
    
    def add_document(self, doc_id: str, path: str, title: str, info: dict):
        """Add or update a document."""
        with get_db() as db:
            doc = db.query(Document).filter(Document.doc_id == doc_id).first()
            if doc:
                doc.path = path
                doc.title = title
                doc.info = info
            else:
                doc = Document(doc_id=doc_id, path=path, title=title, info=info)
                db.add(doc)
    
    def add_chunks(self, chunk_rows: Iterable[StoredChunk]):
        """Add or update chunks."""
        with get_db() as db:
            for c in chunk_rows:
                chunk = db.query(Chunk).filter(Chunk.chunk_id == c.chunk_id).first()
                if chunk:
                    chunk.doc_id = c.doc_id
                    chunk.page = c.page
                    chunk.start = c.start
                    chunk.end = c.end
                    chunk.text = c.text
                else:
                    chunk = Chunk(
                        chunk_id=c.chunk_id,
                        doc_id=c.doc_id,
                        page=c.page,
                        start=c.start,
                        end=c.end,
                        text=c.text,
                    )
                    db.add(chunk)
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[StoredChunk]:
        """Get a chunk by its ID."""
        with get_db() as db:
            chunk = db.query(Chunk).filter(Chunk.chunk_id == chunk_id).first()
            if not chunk:
                return None
            return StoredChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                page=chunk.page or 0,
                start=chunk.start or 0,
                end=chunk.end or 0,
                text=chunk.text or "",
            )

    def get_chunk_by_vector_index(self, vector_index: int) -> Optional[StoredChunk]:
        """Get chunk by vector index using the mapping table."""
        with get_db() as db:
            mapping = db.query(VectorIndexMap).filter(
                VectorIndexMap.vector_index == vector_index
            ).first()
            if not mapping:
                return None
            # Extract chunk_id while session is still open
            chunk_id = mapping.chunk_id
            chunk = db.query(Chunk).filter(Chunk.chunk_id == chunk_id).first()
            if not chunk:
                return None
            # Extract all values while session is still open
            return StoredChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                page=chunk.page or 0,
                start=chunk.start or 0,
                end=chunk.end or 0,
                text=chunk.text or "",
            )
    
    def list_chunks_by_doc(self, doc_id: str) -> List[StoredChunk]:
        """List all chunks for a document."""
        with get_db() as db:
            chunks = db.query(Chunk).filter(Chunk.doc_id == doc_id).order_by(
                Chunk.page, Chunk.start
            ).all()
            return [
                StoredChunk(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    page=c.page or 0,
                    start=c.start or 0,
                    end=c.end or 0,
                    text=c.text or "",
                )
                for c in chunks
            ]

    def get_chunks_by_ids(self, chunk_ids: Sequence[str]) -> Dict[str, StoredChunk]:
        """Fetch chunks by IDs in one query (used by Pinecone-backed retrieval)."""
        ids = [c for c in chunk_ids if c]
        if not ids:
            return {}
        with get_db() as db:
            rows = db.query(Chunk).filter(Chunk.chunk_id.in_(ids)).all()
            out: Dict[str, StoredChunk] = {}
            for chunk in rows:
                out[chunk.chunk_id] = StoredChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    page=chunk.page or 0,
                    start=chunk.start or 0,
                    end=chunk.end or 0,
                    text=chunk.text or "",
                )
            return out
    
    # =========================================================================
    # VECTOR INDEX MAPPING METHODS
    # =========================================================================
    
    def add_vector_index_mapping(self, *, start_index: int, chunk_ids: Sequence[str]) -> None:
        """Add mapping from vector indices to chunk IDs."""
        ids = [c for c in chunk_ids if c]
        if not ids:
            return
        with get_db() as db:
            for i, cid in enumerate(ids):
                idx = start_index + i
                existing = db.query(VectorIndexMap).filter(
                    VectorIndexMap.vector_index == idx
                ).first()
                if existing:
                    existing.chunk_id = cid
                else:
                    db.add(VectorIndexMap(vector_index=idx, chunk_id=cid))
    
    def get_vector_index_by_chunk_id(self, chunk_id: str) -> Optional[int]:
        """Get vector index for a chunk ID."""
        with get_db() as db:
            mapping = db.query(VectorIndexMap).filter(
                VectorIndexMap.chunk_id == chunk_id
            ).first()
            return mapping.vector_index if mapping else None
    
    def list_vector_indices_by_chunk_ids(self, chunk_ids: Sequence[str]) -> Dict[str, int]:
        """Get vector indices for multiple chunk IDs."""
        ids = [c for c in chunk_ids if c]
        if not ids:
            return {}
        with get_db() as db:
            mappings = db.query(VectorIndexMap).filter(
                VectorIndexMap.chunk_id.in_(ids)
            ).all()
            # Extract values while session is still open
            return {m.chunk_id: m.vector_index for m in mappings}
    
    # =========================================================================
    # USER METHODS
    # =========================================================================
    
    def ensure_user(self, user_id: str, *, name: Optional[str] = None) -> None:
        """Create user if not exists, optionally update name."""
        with get_db() as db:
            user = db.query(User).filter(User.user_id == user_id).first()
            if user:
                if name is not None:
                    user.name = name
            else:
                db.add(User(user_id=user_id, name=name))
    
    # =========================================================================
    # EXAM METHODS
    # =========================================================================
    
    def create_exam(
        self,
        *,
        user_id: str,
        title: str,
        mode: str = "mastery",
        info: Optional[Dict[str, Any]] = None,
        exam_id: Optional[str] = None,
    ) -> str:
        """Create an exam workspace. Returns exam_id."""
        self.ensure_user(user_id)
        eid = exam_id or uuid.uuid4().hex[:16]
        with get_db() as db:
            existing = db.query(Exam).filter(Exam.exam_id == eid).first()
            if existing:
                existing.user_id = user_id
                existing.title = title
                existing.mode = mode
                existing.info = info or {}
            else:
                db.add(Exam(
                    exam_id=eid,
                    user_id=user_id,
                    title=title,
                    mode=mode,
                    info=info or {},
                ))
        return eid
    
    def update_exam(self, exam_id: str) -> None:
        """Update exam's updated_at timestamp."""
        with get_db() as db:
            exam = db.query(Exam).filter(Exam.exam_id == exam_id).first()
            if exam:
                exam.updated_at = datetime.now(timezone.utc)
    
    def get_exam(self, exam_id: str) -> Optional[StoredExam]:
        """Get exam by ID."""
        with get_db() as db:
            exam = db.query(Exam).filter(Exam.exam_id == exam_id).first()
            if not exam:
                return None
            return StoredExam(
                exam_id=exam.exam_id,
                user_id=exam.user_id,
                title=exam.title,
                mode=exam.mode,
                created_at=_datetime_to_str(exam.created_at),
                updated_at=_datetime_to_str(exam.updated_at),
                info=exam.info or {},
            )
    
    def list_exams(self, *, user_id: str, limit: int = 50) -> List[StoredExam]:
        """List exams for a user."""
        with get_db() as db:
            exams = db.query(Exam).filter(Exam.user_id == user_id).order_by(
                Exam.updated_at.desc()
            ).limit(limit).all()
            return [
                StoredExam(
                    exam_id=e.exam_id,
                    user_id=e.user_id,
                    title=e.title,
                    mode=e.mode,
                    created_at=_datetime_to_str(e.created_at),
                    updated_at=_datetime_to_str(e.updated_at),
                    info=e.info or {},
                )
                for e in exams
            ]
    
    def attach_documents_to_exam(self, *, exam_id: str, doc_ids: Sequence[str]) -> None:
        """Attach documents to an exam."""
        ids = [d for d in doc_ids if d]
        if not ids:
            return
        with get_db() as db:
            for doc_id in ids:
                existing = db.query(ExamDocument).filter(
                    ExamDocument.exam_id == exam_id,
                    ExamDocument.doc_id == doc_id
                ).first()
                if not existing:
                    db.add(ExamDocument(exam_id=exam_id, doc_id=doc_id))
        self.update_exam(exam_id)
    
    def list_exam_documents(self, *, exam_id: str) -> List[str]:
        """List document IDs attached to an exam."""
        with get_db() as db:
            docs = db.query(ExamDocument).filter(
                ExamDocument.exam_id == exam_id
            ).order_by(ExamDocument.added_at).all()
            return [d.doc_id for d in docs]
    
    # =========================================================================
    # TOPIC METHODS
    # =========================================================================
    
    def upsert_topic(
        self,
        *,
        topic_id: str,
        exam_id: str,
        label: str,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or update a topic."""
        with get_db() as db:
            topic = db.query(Topic).filter(Topic.topic_id == topic_id).first()
            if topic:
                topic.exam_id = exam_id
                topic.label = label
                topic.info = info or {}
            else:
                db.add(Topic(
                    topic_id=topic_id,
                    exam_id=exam_id,
                    label=label,
                    info=info or {},
                ))
    
    def list_topics(self, *, exam_id: str) -> List[StoredTopic]:
        """List topics for an exam."""
        with get_db() as db:
            topics = db.query(Topic).filter(Topic.exam_id == exam_id).order_by(
                Topic.created_at
            ).all()
            return [
                StoredTopic(
                    topic_id=t.topic_id,
                    exam_id=t.exam_id,
                    label=t.label,
                    created_at=_datetime_to_str(t.created_at),
                    info=t.info or {},
                )
                for t in topics
            ]
    
    def delete_topics_for_exam(self, *, exam_id: str) -> None:
        """Delete all topics and related data for an exam."""
        topics = self.list_topics(exam_id=exam_id)
        if not topics:
            return
        topic_ids = [t.topic_id for t in topics]
        with get_db() as db:
            # Delete related records first
            db.query(TopicEvidence).filter(TopicEvidence.topic_id.in_(topic_ids)).delete(
                synchronize_session=False
            )
            db.query(TopicChunk).filter(TopicChunk.topic_id.in_(topic_ids)).delete(
                synchronize_session=False
            )
            db.query(TopicVector).filter(TopicVector.topic_id.in_(topic_ids)).delete(
                synchronize_session=False
            )
            db.query(Topic).filter(Topic.exam_id == exam_id).delete(
                synchronize_session=False
            )
    
    def replace_topic_chunks(self, *, topic_id: str, chunk_ids: Sequence[str]) -> None:
        """Replace all chunk assignments for a topic."""
        ids = [c for c in chunk_ids if c]
        with get_db() as db:
            db.query(TopicChunk).filter(TopicChunk.topic_id == topic_id).delete(
                synchronize_session=False
            )
            for cid in ids:
                db.add(TopicChunk(topic_id=topic_id, chunk_id=cid))
    
    def list_chunk_ids_for_topic(self, *, topic_id: str) -> List[str]:
        """Get chunk IDs for a topic."""
        with get_db() as db:
            chunks = db.query(TopicChunk).filter(TopicChunk.topic_id == topic_id).all()
            return [c.chunk_id for c in chunks]
    
    def list_topic_chunks_for_exam(self, *, exam_id: str) -> Dict[str, List[str]]:
        """Get all topic→chunks mappings for an exam."""
        with get_db() as db:
            results = db.query(TopicChunk.topic_id, TopicChunk.chunk_id).join(
                Topic, Topic.topic_id == TopicChunk.topic_id
            ).filter(Topic.exam_id == exam_id).all()
            out: Dict[str, List[str]] = {}
            for topic_id, chunk_id in results:
                out.setdefault(topic_id, []).append(chunk_id)
            return out
    
    def replace_topic_evidence(
        self,
        *,
        topic_id: str,
        evidence: Sequence[Dict[str, Any]],
    ) -> None:
        """Replace all evidence for a topic."""
        with get_db() as db:
            db.query(TopicEvidence).filter(TopicEvidence.topic_id == topic_id).delete(
                synchronize_session=False
            )
            for ev in evidence:
                db.add(TopicEvidence(
                    evidence_id=ev.get("evidence_id") or uuid.uuid4().hex[:20],
                    topic_id=topic_id,
                    doc_id=str(ev.get("doc_id") or ""),
                    page=int(ev.get("page") or 0),
                    start=int(ev.get("start") or 0),
                    end=int(ev.get("end") or 0),
                    text=str(ev.get("text") or ""),
                ))
    
    # =========================================================================
    # TOPIC VECTOR METHODS
    # =========================================================================
    
    def upsert_topic_vector(self, *, topic_id: str, vector: np.ndarray) -> None:
        """Store or update topic centroid vector."""
        arr = vector.astype("float32", copy=False).reshape(-1)
        with get_db() as db:
            existing = db.query(TopicVector).filter(TopicVector.topic_id == topic_id).first()
            if existing:
                existing.dim = int(arr.size)
                existing.vector = arr.tobytes()
            else:
                db.add(TopicVector(
                    topic_id=topic_id,
                    dim=int(arr.size),
                    vector=arr.tobytes(),
                ))
    
    def get_topic_vector(self, *, topic_id: str) -> Optional[np.ndarray]:
        """Get topic centroid vector."""
        with get_db() as db:
            tv = db.query(TopicVector).filter(TopicVector.topic_id == topic_id).first()
            if not tv:
                return None
            try:
                arr = np.frombuffer(tv.vector, dtype="float32")
                if int(tv.dim) != arr.size:
                    return None
                return arr
            except Exception:
                return None
    
    def list_topic_vectors_for_exam(self, *, exam_id: str) -> Dict[str, np.ndarray]:
        """Get all topic vectors for an exam."""
        with get_db() as db:
            results = db.query(TopicVector.topic_id, TopicVector.dim, TopicVector.vector).join(
                Topic, Topic.topic_id == TopicVector.topic_id
            ).filter(Topic.exam_id == exam_id).all()
            out: Dict[str, np.ndarray] = {}
            for topic_id, dim, blob in results:
                try:
                    arr = np.frombuffer(blob, dtype="float32")
                    if int(dim) == arr.size:
                        out[topic_id] = arr
                except Exception:
                    continue
            return out
    
    # =========================================================================
    # CARD METHODS
    # =========================================================================
    
    def upsert_card(
        self,
        *,
        card_id: str,
        exam_id: str,
        topic_id: str,
        question: str,
        answer: str,
        difficulty: int = 1,
        status: str = "active",
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or update a card."""
        with get_db() as db:
            card = db.query(Card).filter(Card.card_id == card_id).first()
            if card:
                card.exam_id = exam_id
                card.topic_id = topic_id
                card.question = question
                card.answer = answer
                card.difficulty = difficulty
                card.status = status
                card.info = info or {}
            else:
                db.add(Card(
                    card_id=card_id,
                    exam_id=exam_id,
                    topic_id=topic_id,
                    question=question,
                    answer=answer,
                    difficulty=difficulty,
                    status=status,
                    info=info or {},
                ))
    
    def replace_card_proofs(self, *, card_id: str, proofs: Sequence[Dict[str, Any]]) -> None:
        """Replace all proofs for a card."""
        with get_db() as db:
            db.query(CardProof).filter(CardProof.card_id == card_id).delete(
                synchronize_session=False
            )
            for p in proofs:
                db.add(CardProof(
                    proof_id=p.get("proof_id") or uuid.uuid4().hex[:20],
                    card_id=card_id,
                    doc_id=str(p.get("doc_id") or ""),
                    page=int(p.get("page") or 0),
                    start=int(p.get("start") or 0),
                    end=int(p.get("end") or 0),
                    text=str(p.get("text") or ""),
                    score=float(p.get("score") or 0.0),
                ))
    
    def list_cards_for_exam(self, *, exam_id: str, limit: int = 200) -> List[StoredCard]:
        """List cards for an exam."""
        with get_db() as db:
            cards = db.query(Card).filter(Card.exam_id == exam_id).order_by(
                Card.created_at
            ).limit(limit).all()
            return [
                StoredCard(
                    card_id=c.card_id,
                    exam_id=c.exam_id,
                    topic_id=c.topic_id,
                    question=c.question,
                    answer=c.answer,
                    difficulty=c.difficulty,
                    created_at=_datetime_to_str(c.created_at),
                    status=c.status,
                    info=c.info or {},
                )
                for c in cards
            ]
    
    def list_cards_for_topic(
        self, *, topic_id: str, status: str = "active", limit: int = 200
    ) -> List[StoredCard]:
        """List cards for a topic."""
        with get_db() as db:
            cards = db.query(Card).filter(
                Card.topic_id == topic_id,
                Card.status == status
            ).order_by(Card.created_at).limit(limit).all()
            return [
                StoredCard(
                    card_id=c.card_id,
                    exam_id=c.exam_id,
                    topic_id=c.topic_id,
                    question=c.question,
                    answer=c.answer,
                    difficulty=c.difficulty,
                    created_at=_datetime_to_str(c.created_at),
                    status=c.status,
                    info=c.info or {},
                )
                for c in cards
            ]
    
    def get_cards_with_question_vector_idx(
        self, *, exam_id: str
    ) -> Dict[int, StoredCard]:
        """Get cards with question_vector_idx in their info."""
        cards = self.list_cards_for_exam(exam_id=exam_id, limit=10000)
        mapping: Dict[int, StoredCard] = {}
        for card in cards:
            vector_idx = card.info.get("question_vector_idx")
            if vector_idx is not None:
                try:
                    mapping[int(vector_idx)] = card
                except (ValueError, TypeError):
                    continue
        return mapping

    # =========================================================================
    # QUESTION INDEX METHODS (SQL audit/source-of-truth)
    # =========================================================================

    def add_question_index_entry(
        self,
        *,
        question_id: str,
        exam_id: str,
        topic_id: str,
        question_text: str,
        difficulty: Optional[int],
        embedding: np.ndarray,
    ) -> None:
        """Insert a question_index row (question must exist before answering begins)."""
        arr = embedding.astype("float32", copy=False).reshape(-1)
        with get_db() as db:
            existing = db.query(QuestionIndexEntry).filter(
                QuestionIndexEntry.question_id == question_id
            ).first()
            if existing:
                existing.exam_id = exam_id
                existing.topic_id = topic_id
                existing.question_text = question_text
                existing.difficulty = difficulty
                existing.dim = int(arr.size)
                existing.embedding = arr.tobytes()
            else:
                db.add(
                    QuestionIndexEntry(
                        question_id=question_id,
                        exam_id=exam_id,
                        topic_id=topic_id,
                        question_text=question_text,
                        difficulty=difficulty,
                        dim=int(arr.size),
                        embedding=arr.tobytes(),
                    )
                )
    
    # =========================================================================
    # EVENT METHODS
    # =========================================================================
    
    def add_event(
        self,
        *,
        user_id: str,
        exam_id: str,
        type: str,
        payload: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
    ) -> str:
        """Add an event to the exam history."""
        self.ensure_user(user_id)
        eid = event_id or uuid.uuid4().hex[:20]
        with get_db() as db:
            db.add(Event(
                event_id=eid,
                user_id=user_id,
                exam_id=exam_id,
                type=type,
                payload=payload or {},
            ))
        self.update_exam(exam_id)
        return eid
    
    # =========================================================================
    # EMBEDDING CACHE METHODS
    # =========================================================================
    
    def get_cached_embeddings(self, hashes: List[str]) -> Dict[str, np.ndarray]:
        """Get cached embeddings by hash."""
        if not hashes:
            return {}
        with get_db() as db:
            cached = db.query(EmbeddingCache).filter(
                EmbeddingCache.hash.in_(hashes)
            ).all()
            result: Dict[str, np.ndarray] = {}
            for c in cached:
                try:
                    arr = np.frombuffer(c.vector, dtype="float32")
                    if arr.size == c.dim:
                        result[c.hash] = arr
                except Exception:
                    continue
            return result
    
    def add_cached_embeddings(self, mapping: Dict[str, np.ndarray]) -> None:
        """Cache embeddings."""
        if not mapping:
            return
        with get_db() as db:
            for key, vec in mapping.items():
                arr = vec.astype("float32", copy=False)
                existing = db.query(EmbeddingCache).filter(EmbeddingCache.hash == key).first()
                if existing:
                    existing.dim = arr.size
                    existing.vector = arr.tobytes()
                else:
                    db.add(EmbeddingCache(
                        hash=key,
                        dim=arr.size,
                        vector=arr.tobytes(),
                    ))
