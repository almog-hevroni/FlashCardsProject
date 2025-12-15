from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import numpy as np
from app.data.db import SQLiteDB, StoredChunk

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False

VEC_DIM = 3072  # OpenAI text-embedding-3-small

def _normalize_L2(x: np.ndarray) -> None:
    # In-place L2 normalization along rows, similar to faiss.normalize_L2
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    x[:] = x / norms

class _NumpyIPIndex:
    def __init__(self, dim: int, path: Path):
        self.dim = dim
        self.path = path
        if self.path.exists():
            arr = np.load(self.path)
            self.vectors = arr.astype("float32", copy=False)
        else:
            self.vectors = np.zeros((0, dim), dtype="float32")

    def add(self, vectors: np.ndarray) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype("float32")
        _normalize_L2(vectors)
        if self.vectors.size == 0:
            self.vectors = vectors.copy()
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.save()

    def search(self, query_vec: np.ndarray, k: int):
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype("float32")
        _normalize_L2(query_vec)
        n = self.vectors.shape[0]
        if n == 0:
            D = np.zeros((1, k), dtype="float32")
            I = -np.ones((1, k), dtype="int64")
            return D, I
        # query_vec is (1, dim); compute dot-product scores
        scores = self.vectors @ query_vec[0]
        k_eff = min(k, n)
        idx = np.argpartition(-scores, kth=k_eff-1)[:k_eff]
        idx_sorted = idx[np.argsort(-scores[idx])]
        D = scores[idx_sorted].astype("float32")
        I = idx_sorted.astype("int64")
        # pad to k
        if k_eff < k:
            pad_d = np.zeros(k - k_eff, dtype="float32")
            pad_i = -np.ones(k - k_eff, dtype="int64")
            D = np.concatenate([D, pad_d], axis=0)
            I = np.concatenate([I, pad_i], axis=0)
        return D[None, :k], I[None, :k]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.path, self.vectors)


class VectorStore:
    def __init__(self, basepath: str = "store"):
        self.base = Path(basepath)
        self.base.mkdir(parents=True, exist_ok=True)
        
        # Initialize the Metadata Store (SQLite)
        self.db = SQLiteDB(self.base / "meta.sqlite")
        
        # Initialize the Vector Index (FAISS or Numpy)
        self.index_path = self.base / "faiss.index"
        self.vectors_path = self.base / "vectors.npy"

        if _FAISS_AVAILABLE:
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))  # type: ignore
            else:
                self.index = faiss.IndexFlatIP(VEC_DIM)  # type: ignore  # cosine via normalized dot
        else:
            self.index = _NumpyIPIndex(VEC_DIM, self.vectors_path)
        self.vector_dimension = VEC_DIM

    # ------ write ------
    def add_document(self, doc_id: str, path: str, title: str, info: dict):
        self.db.add_document(doc_id, path, title, info)

    def add_chunks(self, chunk_rows: Iterable[StoredChunk], vectors: np.ndarray):
        # 1. Save metadata to SQL
        self.db.add_chunks(chunk_rows)

        # 2. Add vectors to Index
        if vectors.dtype != np.float32:
            vectors = vectors.astype("float32")
        
        if _FAISS_AVAILABLE:
            faiss.normalize_L2(vectors)
            self.index.add(vectors)
            faiss.write_index(self.index, str(self.index_path))
        else:
            self.index.add(vectors)

    # ------ read/search ------
    def topk(self, query_vec: np.ndarray, k: int = 5) -> List[Tuple[StoredChunk, float]]:
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype("float32")
            
        if _FAISS_AVAILABLE:
            faiss.normalize_L2(query_vec)  # type: ignore
            D, I = self.index.search(query_vec, k)  # type: ignore
        else:
            D, I = self.index.search(query_vec, k)
            
        # Reconstruct chunks from SQL using the index from FAISS
        out: List[Tuple[StoredChunk, float]] = []
        for idx, score in zip(I[0].tolist(), D[0].tolist()):
            if idx < 0: continue
            
            chunk = self.db.get_chunk_by_index(idx)
            if not chunk: continue
            
            out.append((chunk, float(score)))
        return out

    # ------ doc helpers ------
    def list_chunks_by_doc(self, doc_id: str) -> List[StoredChunk]:
        return self.db.list_chunks_by_doc(doc_id)

    def sample_chunks_by_doc(self, doc_id: str, n: int = 20) -> List[StoredChunk]:
        import random
        chunks = self.list_chunks_by_doc(doc_id)
        if len(chunks) <= n:
            return chunks
        idxs = list(range(len(chunks)))
        random.shuffle(idxs)
        return [chunks[i] for i in idxs[:n]]

    # ------ cache helpers ------
    def get_cached_embeddings(self, hashes: List[str]) -> Dict[str, np.ndarray]:
        return self.db.get_cached_embeddings(hashes)

    def add_cached_embeddings(self, mapping: Dict[str, np.ndarray]) -> None:
        return self.db.add_cached_embeddings(mapping)
