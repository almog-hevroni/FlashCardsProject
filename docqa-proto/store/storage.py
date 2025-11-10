from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple
import json, sqlite3, numpy as np
try:
    import faiss  # type: ignore
except Exception as exc:
    raise ImportError(
        "FAISS is required but not installed. Install it with 'pip install faiss-cpu' "
        "on CPU-only environments (or 'faiss-gpu' if you have CUDA)."
    ) from exc

VEC_DIM = 1536  # OpenAI text-embedding-3-small

@dataclass
class StoredChunk:
    chunk_id: str
    doc_id: str
    page: int
    start: int
    end: int
    text: str

class VectorStore:
    def __init__(self, basepath: str = "store"):
        self.base = Path(basepath)
        self.base.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base / "meta.sqlite"
        self.index_path = self.base / "faiss.index"

        self.conn = sqlite3.connect(self.db_path)
        self._ensure_schema()

        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = faiss.IndexFlatIP(VEC_DIM)  # cosine via normalized dot

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
        """)
        self.conn.commit()

    # ------ write ------
    def add_document(self, doc_id: str, path: str, title: str, info: dict):
        self.conn.execute(
            "INSERT OR REPLACE INTO documents(doc_id, path, title, info) VALUES(?,?,?,?)",
            (doc_id, path, title, json.dumps(info, ensure_ascii=False)))
        self.conn.commit()

    def add_chunks(self, chunk_rows: Iterable[StoredChunk], vectors: np.ndarray):
        rows = [(c.chunk_id, c.doc_id, c.page, c.start, c.end, c.text) for c in chunk_rows]
        self.conn.executemany(
            "INSERT OR REPLACE INTO chunks(chunk_id, doc_id, page, start, end, text) VALUES(?,?,?,?,?,?)", rows)
        self.conn.commit()

        # normalize for cosine similarity
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        faiss.write_index(self.index, str(self.index_path))

    # ------ read/search ------
    def topk(self, query_vec: np.ndarray, k: int = 5) -> List[Tuple[StoredChunk, float]]:
        faiss.normalize_L2(query_vec)
        D, I = self.index.search(query_vec, k)
        # FAISS doesn't store metadata; we align by row order using rowid from chunks table.
        # Easiest is to store an external table mapping row order; here we rely on implicit order.
        # So instead, we’ll rebuild I -> chunk by rowid using LIMIT/OFFSET (works for proto).
        out: List[Tuple[StoredChunk, float]] = []
        cur = self.conn.cursor()
        for idx, score in zip(I[0].tolist(), D[0].tolist()):
            if idx < 0: continue
            cur.execute("SELECT chunk_id, doc_id, page, start, end, text FROM chunks LIMIT 1 OFFSET ?", (idx,))
            row = cur.fetchone()
            if not row: continue
            c = StoredChunk(*row)
            out.append((c, float(score)))
        return out
