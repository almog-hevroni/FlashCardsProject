from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Sequence, Optional
import os
import numpy as np
from app.data.db_repository import DBRepository, StoredChunk
from app.data.pinecone_backend import PineconeClient

VEC_DIM = 3072  # OpenAI text-embedding-3-large

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
        self.db = DBRepository(self.base / "meta.sqlite")

        # Vector backend selection (default keeps current behavior)
        self.vector_backend: str = os.getenv("VECTOR_BACKEND", "pinecone").strip().lower()
        # Pinecone namespace must be set by exam-scoped flows.
        self.namespace: Optional[str] = None
        self._pinecone: Optional[PineconeClient] = None
        
        # Local fallback index (numpy-only). Pinecone is the primary backend when configured.
        self.vectors_path = self.base / "vectors.npy"
        self.index = _NumpyIPIndex(VEC_DIM, self.vectors_path)
        self.vector_dimension = VEC_DIM

    def set_namespace(self, namespace: Optional[str]) -> None:
        self.namespace = namespace

    def _pinecone_client(self) -> PineconeClient:
        if self._pinecone is None:
            self._pinecone = PineconeClient()
        return self._pinecone

    def _index_size(self) -> int:
        return int(getattr(self.index, "vectors", np.zeros((0, VEC_DIM), dtype="float32")).shape[0])

    # ------ write ------
    def add_document(self, doc_id: str, path: str, title: str, info: dict):
        self.db.add_document(doc_id, path, title, info)

    def add_chunks(self, chunk_rows: Iterable[StoredChunk], vectors: np.ndarray):
        # Materialize to preserve insertion order for vector_index_map.
        chunk_list = list(chunk_rows)

        # 1) Save metadata to SQL
        self.db.add_chunks(chunk_list)

        # 2a) Pinecone backend: upsert vectors by chunk_id into current namespace.
        if self.vector_backend == "pinecone":
            if not self.namespace:
                raise RuntimeError(
                    "VectorStore.namespace not set. For Pinecone backend you must call "
                    "store.set_namespace('u:{user_id}|e:{exam_id}') before add_chunks()."
                )
            if vectors.dtype != np.float32:
                vectors = vectors.astype("float32")
            pc = self._pinecone_client()
            # Attach minimal metadata for debugging/filters.
            meta_by_id: Dict[str, Dict[str, object]] = {}
            for ch in chunk_list:
                meta_by_id[ch.chunk_id] = {
                    "doc_id": ch.doc_id,
                    "page": int(ch.page),
                }
            pc.upsert(
                index=pc.chunks,
                namespace=self.namespace,
                vectors=[(ch.chunk_id, vec) for ch, vec in zip(chunk_list, vectors)],
                metadata_by_id=meta_by_id,
                batch_size=100,
            )
            return

        # 2) Local fallback: add vectors to numpy index + persist stable mapping
        if vectors.dtype != np.float32:
            vectors = vectors.astype("float32")

        start_index = self._index_size()
        self.index.add(vectors)

        # Persist mapping from vector positions -> chunk_id for stable retrieval
        self.db.add_vector_index_mapping(
            start_index=start_index,
            chunk_ids=[c.chunk_id for c in chunk_list],
        )

    # ------ read/search ------
    def topk(self, query_vec: np.ndarray, k: int = 5) -> List[Tuple[StoredChunk, float]]:
        # Pinecone path: return (chunk, score) by querying chunk_id vectors.
        if self.vector_backend == "pinecone":
            if not self.namespace:
                # Fail fast: Pinecone is exam-scoped by namespace.
                raise RuntimeError(
                    "VectorStore.namespace not set. For Pinecone backend you must call "
                    "store.set_namespace('u:{user_id}|e:{exam_id}') before retrieval."
                )
            pc = self._pinecone_client()
            matches = pc.query(
                index=pc.chunks,
                namespace=self.namespace,
                query_vec=query_vec,
                top_k=int(k),
                filter=None,
            )
            if not matches:
                return []
            chunk_ids = [cid for cid, _ in matches]
            by_id = self.db.get_chunks_by_ids(chunk_ids)
            out: List[Tuple[StoredChunk, float]] = []
            for cid, score in matches:
                ch = by_id.get(cid)
                if ch is None:
                    continue
                out.append((ch, float(score)))
            return out

        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype("float32")
            
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
        # Pinecone path: fetch vectors by chunk_id from the current namespace.
        if self.vector_backend == "pinecone":
            if not self.namespace:
                raise RuntimeError(
                    "VectorStore.namespace not set. For Pinecone backend you must call "
                    "store.set_namespace('u:{user_id}|e:{exam_id}') before get_vectors_for_chunk_ids()."
                )
            ids = [c for c in chunk_ids if c]
            if not ids:
                return [], np.zeros((0, self.vector_dimension), dtype="float32")
            pc = self._pinecone_client()
            fetched = pc.fetch_vectors(index=pc.chunks, namespace=self.namespace, ids=ids)
            resolved_ids = [cid for cid in ids if cid in fetched]
            if not resolved_ids:
                return [], np.zeros((0, self.vector_dimension), dtype="float32")
            X = np.stack([fetched[cid] for cid in resolved_ids]).astype("float32", copy=False)
            return resolved_ids, X

        ids = [c for c in chunk_ids if c]
        if not ids:
            return [], np.zeros((0, self.vector_dimension), dtype="float32")
        mapping = self.db.list_vector_indices_by_chunk_ids(ids)
        resolved: List[Tuple[str, int]] = [(cid, mapping[cid]) for cid in ids if cid in mapping]
        if not resolved:
            return [], np.zeros((0, self.vector_dimension), dtype="float32")
        resolved_chunk_ids = [cid for cid, _ in resolved]
        indices = [idx for _, idx in resolved]

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
