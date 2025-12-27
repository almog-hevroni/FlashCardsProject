import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Iterable, Sequence
from dataclasses import dataclass
import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

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

class SQLiteDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Use a non-zero timeout and WAL mode to reduce "database is locked" errors
        # when multiple threads/processes read/write concurrently.
        self.conn = sqlite3.connect(self.db_path, timeout=5.0)
        try:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            self.conn.execute("PRAGMA busy_timeout=5000;")
        except Exception as exc:
            logger.warning("SQLite PRAGMA setup failed; continuing without WAL/busy_timeout. Error=%s", repr(exc))
        self._ensure_schema()

    def _ensure_schema(self):
        cur = self.conn.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS documents(
            doc_id TEXT PRIMARY KEY,
            path TEXT,
            title TEXT,
            info TEXT
        );
        CREATE TABLE IF NOT EXISTS chunks(
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            page INTEGER,
            start INTEGER,
            end INTEGER,
            text TEXT
        );
        CREATE TABLE IF NOT EXISTS embedding_cache(
            hash TEXT PRIMARY KEY,
            dim INTEGER,
            vector BLOB
        );

        -- -----------------------------
        -- Stable mapping: vector index position -> chunk_id
        -- -----------------------------
        CREATE TABLE IF NOT EXISTS vector_index_map(
            vector_index INTEGER PRIMARY KEY,
            chunk_id TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_vector_index_map_chunk_id ON vector_index_map(chunk_id);

        -- -----------------------------
        -- Topics (per exam)
        -- -----------------------------
        CREATE TABLE IF NOT EXISTS topics(
            topic_id TEXT PRIMARY KEY,
            exam_id TEXT NOT NULL,
            label TEXT NOT NULL,
            created_at TEXT DEFAULT (CURRENT_TIMESTAMP),
            info TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_topics_exam_id ON topics(exam_id);

        CREATE TABLE IF NOT EXISTS topic_chunks(
            topic_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            PRIMARY KEY (topic_id, chunk_id)
        );
        CREATE INDEX IF NOT EXISTS idx_topic_chunks_chunk_id ON topic_chunks(chunk_id);

        -- Evidence spans proving the topic label is grounded in document text
        CREATE TABLE IF NOT EXISTS topic_evidence(
            evidence_id TEXT PRIMARY KEY,
            topic_id TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            page INTEGER,
            start INTEGER,
            end INTEGER,
            text TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_topic_evidence_topic_id ON topic_evidence(topic_id);

        -- -----------------------------
        -- Exams / users
        -- -----------------------------
        CREATE TABLE IF NOT EXISTS users(
            user_id TEXT PRIMARY KEY,
            name TEXT,
            created_at TEXT DEFAULT (CURRENT_TIMESTAMP)
        );

        CREATE TABLE IF NOT EXISTS exams(
            exam_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'mastery',
            created_at TEXT DEFAULT (CURRENT_TIMESTAMP),
            updated_at TEXT DEFAULT (CURRENT_TIMESTAMP),
            info TEXT
        );

        CREATE TABLE IF NOT EXISTS exam_documents(
            exam_id TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            added_at TEXT DEFAULT (CURRENT_TIMESTAMP),
            PRIMARY KEY (exam_id, doc_id)
        );

        CREATE TABLE IF NOT EXISTS events(
            event_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            exam_id TEXT NOT NULL,
            type TEXT NOT NULL,
            payload TEXT,
            created_at TEXT DEFAULT (CURRENT_TIMESTAMP)
        );

        CREATE INDEX IF NOT EXISTS idx_exams_user_id ON exams(user_id);
        CREATE INDEX IF NOT EXISTS idx_exam_documents_exam_id ON exam_documents(exam_id);
        CREATE INDEX IF NOT EXISTS idx_events_exam_id ON events(exam_id);
        """)
        self.conn.commit()

    def add_document(self, doc_id: str, path: str, title: str, info: dict):
        self.conn.execute(
            "INSERT OR REPLACE INTO documents(doc_id, path, title, info) VALUES(?,?,?,?)",
            (doc_id, path, title, json.dumps(info, ensure_ascii=False)))
        self.conn.commit()

    def add_chunks(self, chunk_rows: Iterable[StoredChunk]):
        rows = [(c.chunk_id, c.doc_id, c.page, c.start, c.end, c.text) for c in chunk_rows]
        self.conn.executemany(
            "INSERT OR REPLACE INTO chunks(chunk_id, doc_id, page, start, end, text) VALUES(?,?,?,?,?,?)", rows)
        self.conn.commit()

    # -----------------------------
    # Exam helpers
    # -----------------------------
    def ensure_user(self, user_id: str, *, name: Optional[str] = None) -> None:
        """
        Create user row if it doesn't exist. Safe to call repeatedly.
        """
        self.conn.execute(
            "INSERT OR IGNORE INTO users(user_id, name) VALUES(?,?)",
            (user_id, name),
        )
        if name is not None:
            self.conn.execute(
                "UPDATE users SET name=? WHERE user_id=?",
                (name, user_id),
            )
        self.conn.commit()

    def create_exam(
        self,
        *,
        user_id: str,
        title: str,
        mode: str = "mastery",
        info: Optional[Dict[str, Any]] = None,
        exam_id: Optional[str] = None,
    ) -> str:
        """
        Create an exam workspace (like a ChatGPT thread). Returns exam_id.
        """
        self.ensure_user(user_id)
        eid = exam_id or uuid.uuid4().hex[:16]
        now = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(info or {}, ensure_ascii=False)
        self.conn.execute(
            "INSERT OR REPLACE INTO exams(exam_id, user_id, title, mode, created_at, updated_at, info) "
            "VALUES(?,?,?,?,?,?,?)",
            (eid, user_id, title, mode, now, now, payload),
        )
        self.conn.commit()
        return eid

    def update_exam(self, exam_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE exams SET updated_at=? WHERE exam_id=?",
            (now, exam_id),
        )
        self.conn.commit()

    # -----------------------------
    # Topic helpers
    # -----------------------------
    def upsert_topic(
        self,
        *,
        topic_id: str,
        exam_id: str,
        label: str,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = json.dumps(info or {}, ensure_ascii=False)
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT OR REPLACE INTO topics(topic_id, exam_id, label, created_at, info) VALUES(?,?,?,?,?)",
            (topic_id, exam_id, label, now, payload),
        )
        self.conn.commit()

    def list_topics(self, *, exam_id: str) -> List[StoredTopic]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT topic_id, exam_id, label, created_at, info FROM topics WHERE exam_id=? ORDER BY created_at ASC",
            (exam_id,),
        )
        rows = cur.fetchall()
        out: List[StoredTopic] = []
        for topic_id, exam_id_val, label, created_at, info_raw in rows:
            try:
                info = json.loads(info_raw) if isinstance(info_raw, str) and info_raw else {}
            except Exception:
                info = {}
            out.append(
                StoredTopic(
                    topic_id=str(topic_id),
                    exam_id=str(exam_id_val),
                    label=str(label),
                    created_at=str(created_at),
                    info=info,
                )
            )
        return out

    def replace_topic_chunks(self, *, topic_id: str, chunk_ids: Sequence[str]) -> None:
        ids = [c for c in chunk_ids if c]
        cur = self.conn.cursor()
        cur.execute("DELETE FROM topic_chunks WHERE topic_id=?", (topic_id,))
        if ids:
            rows = [(topic_id, cid) for cid in ids]
            self.conn.executemany(
                "INSERT OR IGNORE INTO topic_chunks(topic_id, chunk_id) VALUES(?,?)",
                rows,
            )
        self.conn.commit()

    def replace_topic_evidence(
        self,
        *,
        topic_id: str,
        evidence: Sequence[Dict[str, Any]],
    ) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM topic_evidence WHERE topic_id=?", (topic_id,))
        rows = []
        for ev in evidence:
            evidence_id = ev.get("evidence_id") or uuid.uuid4().hex[:20]
            rows.append(
                (
                    str(evidence_id),
                    topic_id,
                    str(ev.get("doc_id") or ""),
                    int(ev.get("page") or 0),
                    int(ev.get("start") or 0),
                    int(ev.get("end") or 0),
                    str(ev.get("text") or ""),
                )
            )
        if rows:
            self.conn.executemany(
                "INSERT OR REPLACE INTO topic_evidence(evidence_id, topic_id, doc_id, page, start, end, text) "
                "VALUES(?,?,?,?,?,?,?)",
                rows,
            )
        self.conn.commit()

    def get_exam(self, exam_id: str) -> Optional[StoredExam]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT exam_id, user_id, title, mode, created_at, updated_at, info FROM exams WHERE exam_id=?",
            (exam_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        info_raw = row[6] or "{}"
        try:
            info = json.loads(info_raw) if isinstance(info_raw, str) else {}
        except Exception:
            info = {}
        return StoredExam(
            exam_id=row[0],
            user_id=row[1],
            title=row[2],
            mode=row[3],
            created_at=row[4],
            updated_at=row[5],
            info=info,
        )

    def list_exams(self, *, user_id: str, limit: int = 50) -> List[StoredExam]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT exam_id, user_id, title, mode, created_at, updated_at, info "
            "FROM exams WHERE user_id=? ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        )
        rows = cur.fetchall()
        out: List[StoredExam] = []
        for row in rows:
            info_raw = row[6] or "{}"
            try:
                info = json.loads(info_raw) if isinstance(info_raw, str) else {}
            except Exception:
                info = {}
            out.append(
                StoredExam(
                    exam_id=row[0],
                    user_id=row[1],
                    title=row[2],
                    mode=row[3],
                    created_at=row[4],
                    updated_at=row[5],
                    info=info,
                )
            )
        return out

    def attach_documents_to_exam(self, *, exam_id: str, doc_ids: Sequence[str]) -> None:
        ids = [d for d in doc_ids if d]
        if not ids:
            return
        rows = [(exam_id, doc_id) for doc_id in ids]
        self.conn.executemany(
            "INSERT OR IGNORE INTO exam_documents(exam_id, doc_id) VALUES(?,?)",
            rows,
        )
        self.conn.commit()
        self.update_exam(exam_id)

    def list_exam_documents(self, *, exam_id: str) -> List[str]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT doc_id FROM exam_documents WHERE exam_id=? ORDER BY added_at ASC",
            (exam_id,),
        )
        return [r[0] for r in cur.fetchall() if r and r[0]]

    def add_event(
        self,
        *,
        user_id: str,
        exam_id: str,
        type: str,
        payload: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
    ) -> str:
        """
        Append an immutable event to the exam history (ratings, next-card served, etc.).
        """
        self.ensure_user(user_id)
        eid = event_id or uuid.uuid4().hex[:20]
        payload_raw = json.dumps(payload or {}, ensure_ascii=False)
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO events(event_id, user_id, exam_id, type, payload, created_at) VALUES(?,?,?,?,?,?)",
            (eid, user_id, exam_id, type, payload_raw, now),
        )
        self.conn.commit()
        self.update_exam(exam_id)
        return eid

    # -----------------------------
    # Vector index mapping helpers
    # -----------------------------
    def add_vector_index_mapping(self, *, start_index: int, chunk_ids: Sequence[str]) -> None:
        """
        Persist mapping from vector index positions [start_index .. start_index+N-1] to chunk_ids,
        in the exact order vectors were added to the vector index.
        """
        ids = [c for c in chunk_ids if c]
        if not ids:
            return
        rows = [(start_index + i, cid) for i, cid in enumerate(ids)]
        self.conn.executemany(
            "INSERT OR REPLACE INTO vector_index_map(vector_index, chunk_id) VALUES(?,?)",
            rows,
        )
        self.conn.commit()

    def get_chunk_by_id(self, chunk_id: str) -> Optional[StoredChunk]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT chunk_id, doc_id, page, start, end, text FROM chunks WHERE chunk_id=?",
            (chunk_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return StoredChunk(*row)

    def get_chunk_by_vector_index(self, vector_index: int) -> Optional[StoredChunk]:
        """
        Preferred stable lookup: vector_index -> chunk_id -> chunk row.
        Falls back to legacy LIMIT/OFFSET if mapping isn't present (older stores).
        """
        cur = self.conn.cursor()
        cur.execute(
            "SELECT chunk_id FROM vector_index_map WHERE vector_index=?",
            (vector_index,),
        )
        row = cur.fetchone()
        if row and row[0]:
            ch = self.get_chunk_by_id(str(row[0]))
            if ch:
                return ch
        # Fallback for older DBs (pre-mapping)
        return self.get_chunk_by_index(vector_index)

    def get_vector_index_by_chunk_id(self, chunk_id: str) -> Optional[int]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT vector_index FROM vector_index_map WHERE chunk_id=?",
            (chunk_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        try:
            return int(row[0])
        except Exception:
            return None

    def list_vector_indices_by_chunk_ids(self, chunk_ids: Sequence[str]) -> Dict[str, int]:
        """
        Return mapping {chunk_id: vector_index} for any chunk_ids present in vector_index_map.
        """
        ids = [c for c in chunk_ids if c]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        cur = self.conn.cursor()
        cur.execute(
            f"SELECT chunk_id, vector_index FROM vector_index_map WHERE chunk_id IN ({placeholders})",
            ids,
        )
        out: Dict[str, int] = {}
        for ch_id, vec_idx in cur.fetchall():
            if not ch_id:
                continue
            try:
                out[str(ch_id)] = int(vec_idx)
            except Exception:
                continue
        return out

    def get_chunk_by_index(self, index: int) -> Optional[StoredChunk]:
        """
        Retrieve a chunk based on implicit rowid order (LIMIT 1 OFFSET index).
        Note: This is brittle and relies on rowid stability matching FAISS insertion order.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT chunk_id, doc_id, page, start, end, text FROM chunks LIMIT 1 OFFSET ?", (index,))
        row = cur.fetchone()
        if not row:
            return None
        return StoredChunk(*row)

    def list_chunks_by_doc(self, doc_id: str) -> List[StoredChunk]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT chunk_id, doc_id, page, start, end, text FROM chunks WHERE doc_id=? ORDER BY page, start",
            (doc_id,))
        rows = cur.fetchall()
        return [StoredChunk(*row) for row in rows]

    def get_cached_embeddings(self, hashes: List[str]) -> Dict[str, np.ndarray]:
        if not hashes:
            return {}
        placeholders = ",".join("?" for _ in hashes)
        cur = self.conn.cursor()
        cur.execute(
            f"SELECT hash, dim, vector FROM embedding_cache WHERE hash IN ({placeholders})",
            hashes,
        )
        cached: Dict[str, np.ndarray] = {}
        for hash_value, dim, blob in cur.fetchall():
            try:
                arr = np.frombuffer(blob, dtype="float32")
                if arr.size != dim:
                    continue
                cached[hash_value] = arr
            except Exception:
                continue
        return cached

    def add_cached_embeddings(self, mapping: Dict[str, np.ndarray]) -> None:
        if not mapping:
            return
        rows = []
        for key, vec in mapping.items():
            arr = vec.astype("float32", copy=False)
            rows.append((key, arr.size, arr.tobytes()))
        self.conn.executemany(
            "INSERT OR REPLACE INTO embedding_cache(hash, dim, vector) VALUES(?,?,?)",
            rows,
        )
        self.conn.commit()

