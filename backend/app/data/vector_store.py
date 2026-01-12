from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Sequence, Optional
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

    def _index_size(self) -> int:
        if _FAISS_AVAILABLE:
            return int(getattr(self.index, "ntotal", 0))
        # numpy fallback index
        return int(getattr(self.index, "vectors", np.zeros((0, VEC_DIM), dtype="float32")).shape[0])

    # ------ write ------
    def add_document(self, doc_id: str, path: str, title: str, info: dict):
        self.db.add_document(doc_id, path, title, info)

    def add_chunks(self, chunk_rows: Iterable[StoredChunk], vectors: np.ndarray):
        # Materialize to preserve insertion order for vector_index_map.
        chunk_list = list(chunk_rows)

        # 1) Save metadata to SQL
        self.db.add_chunks(chunk_list)

        # 2) Add vectors to Index + persist stable mapping
        if vectors.dtype != np.float32:
            vectors = vectors.astype("float32")

        start_index = self._index_size()
        
        if _FAISS_AVAILABLE:
            faiss.normalize_L2(vectors)
            self.index.add(vectors)
            faiss.write_index(self.index, str(self.index_path))
        else:
            self.index.add(vectors)

        # Persist mapping from vector positions -> chunk_id for stable retrieval
        self.db.add_vector_index_mapping(
            start_index=start_index,
            chunk_ids=[c.chunk_id for c in chunk_list],
        )

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
            
            chunk = self.db.get_chunk_by_vector_index(idx)
            if not chunk: continue
            
            out.append((chunk, float(score)))
        return out

    def get_vectors_for_chunk_ids(self, chunk_ids: Sequence[str]) -> Tuple[List[str], np.ndarray]:
        """
        Fetch vectors (in embedding space) for the given chunk_ids.
        Returns (resolved_chunk_ids_in_order, vectors[N, dim]).

        Notes:
        - Requires vector_index_map entries; for older stores, missing chunk_ids are skipped.
        """
        ids = [c for c in chunk_ids if c]
        if not ids:
            return [], np.zeros((0, self.vector_dimension), dtype="float32")
        mapping = self.db.list_vector_indices_by_chunk_ids(ids)
        resolved: List[Tuple[str, int]] = [(cid, mapping[cid]) for cid in ids if cid in mapping]
        if not resolved:
            return [], np.zeros((0, self.vector_dimension), dtype="float32")
        resolved_chunk_ids = [cid for cid, _ in resolved]
        indices = [idx for _, idx in resolved]

        if _FAISS_AVAILABLE:
            # Prefer batch reconstruction when available.
            try:
                if hasattr(self.index, "reconstruct_batch"):
                    vecs = self.index.reconstruct_batch(np.array(indices, dtype="int64"))  # type: ignore
                    return resolved_chunk_ids, np.array(vecs, dtype="float32", copy=False)
            except Exception:
                pass
            # Fallback: reconstruct one-by-one.
            out = np.zeros((len(indices), self.vector_dimension), dtype="float32")
            for i, idx in enumerate(indices):
                out[i] = np.array(self.index.reconstruct(int(idx)), dtype="float32")  # type: ignore
            return resolved_chunk_ids, out

        # Numpy fallback index stores vectors directly
        vec_mat = getattr(self.index, "vectors", None)
        if vec_mat is None:
            return resolved_chunk_ids, np.zeros((0, self.vector_dimension), dtype="float32")
        # Guard against out-of-range indices (shouldn't happen if mapping is correct).
        max_n = int(vec_mat.shape[0])
        safe_pairs: List[Tuple[str, int]] = [(cid, idx) for cid, idx in resolved if 0 <= idx < max_n]
        if not safe_pairs:
            return [], np.zeros((0, self.vector_dimension), dtype="float32")
        resolved_chunk_ids = [cid for cid, _ in safe_pairs]
        safe_indices = [idx for _, idx in safe_pairs]
        return resolved_chunk_ids, np.array(vec_mat[safe_indices], dtype="float32", copy=False)

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


class QuestionIndex:
    """
    Manages question embeddings for efficient similarity-based deduplication.
    
    Separate from the chunk index - stores only question vectors.
    Uses FAISS when available, falls back to numpy otherwise.
    
    The mapping from vector_idx -> card_id is stored in cards.info JSON field,
    not in a separate table.
    """
    
    def __init__(self, basepath: str = "store"):
        self.base = Path(basepath)
        self.base.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.base / "questions.index"
        self.vectors_path = self.base / "questions.npy"
        
        if _FAISS_AVAILABLE:
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))  # type: ignore
            else:
                self.index = faiss.IndexFlatIP(VEC_DIM)  # type: ignore
        else:
            self.index = _NumpyIPIndex(VEC_DIM, self.vectors_path)
        
        self.vector_dimension = VEC_DIM
    
    def size(self) -> int:
        """Return the number of vectors in the index."""
        if _FAISS_AVAILABLE:
            return int(getattr(self.index, "ntotal", 0))
        return int(getattr(self.index, "vectors", np.zeros((0, VEC_DIM))).shape[0])
    
    def search(self, query_vec: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar questions.
        
        Args:
            query_vec: Query embedding, shape (dim,) or (1, dim)
            k: Number of nearest neighbors to return
            
        Returns:
            (similarities, indices) - both shape (1, k)
            similarities are cosine similarities (higher = more similar)
            indices are vector positions (-1 for padding if fewer than k exist)
        """
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype("float32")
        
        if _FAISS_AVAILABLE:
            faiss.normalize_L2(query_vec)  # type: ignore
            D, I = self.index.search(query_vec, k)  # type: ignore
        else:
            D, I = self.index.search(query_vec, k)
        
        return D, I
    
    def add(self, embedding: np.ndarray) -> int:
        """
        Add a question embedding to the index.
        
        Args:
            embedding: Question embedding, shape (dim,) or (1, dim)
            
        Returns:
            The vector index where it was stored (to save in cards.info)
        """
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        if embedding.dtype != np.float32:
            embedding = embedding.astype("float32")
        
        vector_idx = self.size()
        
        if _FAISS_AVAILABLE:
            faiss.normalize_L2(embedding)  # type: ignore
            self.index.add(embedding)  # type: ignore
            faiss.write_index(self.index, str(self.index_path))  # type: ignore
        else:
            self.index.add(embedding)
        
        return vector_idx
