import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Iterable
from dataclasses import dataclass

@dataclass
class StoredChunk:
    chunk_id: str
    doc_id: str
    page: int
    start: int
    end: int
    text: str

class SQLiteDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
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

