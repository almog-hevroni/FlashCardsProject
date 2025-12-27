from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from app.data.db import StoredChunk, StoredTopic
from app.data.vector_store import VectorStore
from app.services.exams import load_exam
from app.services.llm import CHAT_MODEL_FAST, chat_completions_create


_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "as", "by",
    "is", "are", "was", "were", "be", "been", "being",
    "that", "this", "these", "those",
    "it", "its", "they", "their", "we", "our", "you", "your",
    "from", "at", "into", "over", "under", "between", "within",
    "not", "no", "yes",
}


@dataclass
class BuiltTopic:
    topic_id: str
    label: str
    chunk_ids: List[str]
    evidence: List[Dict[str, Any]]
    info: Dict[str, Any]


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (x / norms).astype("float32", copy=False)


def _tokenize(text: str) -> List[str]:
    # Simple tokenizer: keep alphanumerics, greek letters commonly appear as unicode words too.
    tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", text.lower())
    return [t for t in tokens if t and t not in _STOPWORDS and len(t) >= 3]


def _ngram_phrases(tokens: List[str], n: int) -> Iterable[str]:
    if n <= 0:
        return []
    for i in range(0, len(tokens) - n + 1):
        yield " ".join(tokens[i : i + n])


def _choose_grounded_label(texts: Sequence[str], max_words: int = 3) -> str:
    """
    Pick a topic label that appears in the cluster text (grounded) by scoring frequent n-grams.
    This is intentionally non-LLM for determinism and to avoid hallucinated labels.
    """
    joined = "\n".join(t for t in texts if t)
    tokens = _tokenize(joined)
    if not tokens:
        return "Topic"

    scores: Dict[str, float] = {}
    # Prefer longer phrases slightly.
    for n in range(min(max_words, 3), 0, -1):
        for phrase in _ngram_phrases(tokens, n):
            # Downweight overly generic endings.
            if phrase.split()[-1] in _STOPWORDS:
                continue
            scores[phrase] = scores.get(phrase, 0.0) + (1.0 + 0.2 * (n - 1))

    if not scores:
        return "Topic"

    # Choose best scoring phrase that is actually present as a substring in cluster text (case-insensitive).
    joined_lower = joined.lower()
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    for phrase, _ in ranked[:200]:
        if phrase in joined_lower:
            # Title-case lightly for presentation, but keep original words.
            return " ".join(w.capitalize() if len(w) > 3 else w for w in phrase.split())

    # Fallback: top token
    return tokens[0].capitalize()


def _extract_evidence_for_label(chunks: Sequence[StoredChunk], label: str, max_items: int = 3) -> List[Dict[str, Any]]:
    """
    Find up to max_items evidence spans where the label appears in the chunk text.
    Evidence is grounded because it's literally taken from document text.
    """
    if not label:
        return []
    label_l = label.lower()
    evidence: List[Dict[str, Any]] = []
    for ch in chunks:
        hay = (ch.text or "")
        idx = hay.lower().find(label_l)
        if idx < 0:
            continue
        # Capture a small window around the match for inspection.
        window = 220
        s = max(0, idx - window // 2)
        e = min(len(hay), idx + len(label) + window // 2)
        snippet = hay[s:e].strip()
        evidence.append(
            {
                "evidence_id": uuid.uuid4().hex[:20],
                "doc_id": ch.doc_id,
                "page": ch.page,
                "start": int(ch.start + idx),
                "end": int(ch.start + idx + len(label)),
                "text": snippet,
            }
        )
        if len(evidence) >= max_items:
            break
    return evidence


def _extract_evidence_for_phrases(
    chunks: Sequence[StoredChunk],
    phrases: Sequence[str],
    *,
    max_items: int = 3,
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Try multiple candidate phrases and return (evidence, matched_phrase).
    """
    cleaned = [p.strip() for p in phrases if isinstance(p, str) and p.strip()]
    if not cleaned:
        return [], None
    for phrase in cleaned:
        ev = _extract_evidence_for_label(chunks, phrase, max_items=max_items)
        if ev:
            return ev, phrase
    return [], None


def _build_cluster_context_pack(
    *,
    cids: Sequence[str],
    chunk_by_id: Dict[str, StoredChunk],
    X: np.ndarray,
    id_to_row: Dict[str, int],
    max_chunks: int = 8,
    max_chars_per_chunk: int = 650,
    max_total_chars: int = 6000,
) -> str:
    """
    Build a compact context pack from the most representative chunks in a cluster.
    Representative = highest cosine similarity to cluster centroid (on normalized embeddings).
    """
    rows: List[int] = []
    kept_ids: List[str] = []
    for cid in cids:
        r = id_to_row.get(cid)
        if r is None:
            continue
        rows.append(int(r))
        kept_ids.append(cid)
    if not rows:
        return ""

    Xn = _l2_normalize(X.astype("float32", copy=False))
    V = Xn[rows]
    centroid = np.mean(V, axis=0, keepdims=True).astype("float32", copy=False)
    centroid = _l2_normalize(centroid)[0]
    sims = (V @ centroid).tolist()
    ranked = sorted(zip(kept_ids, sims), key=lambda t: t[1], reverse=True)

    buf: List[str] = []
    used = 0
    picked = 0
    for cid, sim in ranked:
        ch = chunk_by_id.get(cid)
        if not ch or not ch.text:
            continue
        text = ch.text.strip().replace("\n", " ")
        if len(text) > max_chars_per_chunk:
            text = text[: max_chars_per_chunk - 3].rstrip() + "..."
        block = f"[doc={ch.doc_id} page={ch.page} sim={sim:.2f}] {text}\n"
        if used + len(block) > max_total_chars:
            break
        buf.append(block)
        used += len(block)
        picked += 1
        if picked >= max_chunks:
            break
    return "".join(buf).strip()


def _llm_label_cluster(
    *,
    context_pack: str,
    model: str = CHAT_MODEL_FAST,
) -> Dict[str, Any]:
    """
    Ask a fast LLM to propose a conceptual topic label and evidence phrases.
    Returns dict with keys: label (str), evidence_phrases (list[str]), key_terms (list[str]).
    """
    sys_prompt = (
        "You extract study topics from document excerpts.\n"
        "Propose conceptual, specific topic labels.\n"
        "Return JSON only."
    )
    user_prompt = (
        "Given the following excerpts that belong to ONE semantic cluster, propose a single topic.\n\n"
        "EXCERPTS:\n"
        f"{context_pack}\n\n"
        "Return JSON with this schema:\n"
        "{\n"
        '  \"label\": \"<short topic label>\",\n'
        '  \"key_terms\": [\"<term>\", ...],\n'
        '  \"evidence_phrases\": [\"<short phrase that should appear verbatim in the excerpts>\", ...]\n'
        "}\n\n"
        "Rules:\n"
        "- label should be 3–10 words, specific (not generic like 'Students' or 'Questions').\n"
        "- Prefer academic phrasing if appropriate (e.g., 'Hypothesis-based evaluation (H1–H3)').\n"
        "- evidence_phrases must be phrases you believe appear verbatim in the excerpts; include abbreviations too.\n"
        "- Keep evidence_phrases short (2–6 words) and include 3–8 items.\n"
        "- Return valid JSON only."
    )
    resp = chat_completions_create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=220,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}

    label = str(data.get("label", "")).strip()
    key_terms = data.get("key_terms") if isinstance(data.get("key_terms"), list) else []
    evidence_phrases = data.get("evidence_phrases") if isinstance(data.get("evidence_phrases"), list) else []
    return {
        "label": label,
        "key_terms": [str(x).strip() for x in key_terms if str(x).strip()],
        "evidence_phrases": [str(x).strip() for x in evidence_phrases if str(x).strip()],
    }

def _kmeans_cluster(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """
    Returns assignments array of shape (n,) with values in [0..k-1].
    Uses FAISS KMeans when available; otherwise a small numpy kmeans fallback.
    """
    n, d = X.shape
    if n == 0:
        return np.zeros((0,), dtype="int64")
    k = max(1, min(int(k), n))

    # Try FAISS kmeans if available.
    try:
        import faiss  # type: ignore

        # FAISS KMeans supports cosine-ish via normalized vectors + IndexFlatIP assignment.
        Xn = _l2_normalize(X.astype("float32", copy=False))
        km = faiss.Kmeans(d, k, niter=25, verbose=False, seed=seed)  # type: ignore
        km.train(Xn)  # type: ignore
        centroids = km.centroids  # type: ignore
        centroids = _l2_normalize(np.array(centroids, dtype="float32", copy=False))
        sims = Xn @ centroids.T
        return np.argmax(sims, axis=1).astype("int64")
    except Exception:
        pass

    # Numpy fallback (small datasets): classic kmeans on normalized vectors.
    rng = np.random.RandomState(seed)
    Xn = _l2_normalize(X.astype("float32", copy=False))
    # init: pick k random points
    centroids = Xn[rng.choice(n, size=k, replace=False)].copy()
    for _ in range(20):
        sims = Xn @ centroids.T
        assign = np.argmax(sims, axis=1)
        new_centroids = centroids.copy()
        for j in range(k):
            mask = assign == j
            if not np.any(mask):
                new_centroids[j] = Xn[rng.randint(0, n)]
                continue
            new_centroids[j] = _l2_normalize(np.mean(Xn[mask], axis=0, keepdims=True))[0]
        if np.allclose(new_centroids, centroids, atol=1e-4):
            break
        centroids = new_centroids
    sims = Xn @ centroids.T
    return np.argmax(sims, axis=1).astype("int64")


def _pick_k(num_chunks: int) -> int:
    """
    Heuristic for number of topics. Conservative to avoid over-fragmentation.
    """
    if num_chunks <= 8:
        return 2
    k = int(round(np.sqrt(num_chunks)))
    return max(2, min(12, k))


def _merge_clusters_by_centroid(
    *,
    clusters: List[tuple[int, List[str]]],
    X: np.ndarray,
    id_to_row: Dict[str, int],
    threshold: float = 0.90,
) -> List[Dict[str, Any]]:
    """
    Post-process clusters to reduce redundancy:
    - compute centroid per cluster (mean of normalized chunk vectors)
    - greedily merge clusters whose centroids have cosine similarity >= threshold

    Returns list of merged cluster dicts:
    { "cluster_ids": [..], "chunk_ids": [..], "centroid": np.ndarray, "merged": bool }

    This is deliberately LLM-free and cheap (matrix ops only).
    """
    if not clusters:
        return []
    # Normalize vectors once for cosine similarity.
    Xn = _l2_normalize(X.astype("float32", copy=False))

    # Build centroid per cluster (on Xn).
    items: List[Dict[str, Any]] = []
    for cluster_id, cids in clusters:
        rows = [id_to_row[cid] for cid in cids if cid in id_to_row]
        if not rows:
            continue
        V = Xn[rows]
        centroid = _l2_normalize(np.mean(V, axis=0, keepdims=True))[0]
        items.append(
            {
                "cluster_ids": [int(cluster_id)],
                "chunk_ids": list(cids),
                "centroid": centroid,
                "merged": False,
                "merge_sims": [],  # list of floats for audit
            }
        )

    # Sort by size desc for stable greedy merges (keep biggest themes).
    items.sort(key=lambda it: len(it["chunk_ids"]), reverse=True)

    kept: List[Dict[str, Any]] = []
    for it in items:
        c = it["centroid"]
        best_j = None
        best_sim = -1.0
        for j, k_it in enumerate(kept):
            sim = float(np.dot(c, k_it["centroid"]))
            if sim > best_sim:
                best_sim = sim
                best_j = j
        if best_j is not None and best_sim >= float(threshold):
            # Merge into best existing cluster
            target = kept[best_j]
            target["cluster_ids"].extend(it["cluster_ids"])
            target["chunk_ids"].extend(it["chunk_ids"])
            target["merged"] = True
            target["merge_sims"].append(best_sim)

            # Recompute centroid using mean of member vectors (approx by averaging centroids weighted by sizes)
            # For better accuracy without heavy computation, recompute from underlying rows:
            rows = [id_to_row[cid] for cid in target["chunk_ids"] if cid in id_to_row]
            if rows:
                V = Xn[rows]
                target["centroid"] = _l2_normalize(np.mean(V, axis=0, keepdims=True))[0]
        else:
            kept.append(it)

    # De-duplicate chunk_ids within each merged cluster, preserve first-seen order.
    for it in kept:
        seen = set()
        deduped = []
        for cid in it["chunk_ids"]:
            if cid in seen:
                continue
            seen.add(cid)
            deduped.append(cid)
        it["chunk_ids"] = deduped
    return kept


def build_topics_for_exam(
    *,
    exam_id: str,
    store: Optional[VectorStore] = None,
    overwrite: bool = True,
    seed: int = 0,
    use_llm_labels: bool = True,
    llm_model: str = CHAT_MODEL_FAST,
    merge_threshold: float = 0.88,
) -> List[BuiltTopic]:
    """
    Build semantic topics for an exam by clustering chunk embeddings and producing grounded labels + evidence.
    Persists topics + topic_chunks + topic_evidence into SQLite.
    """
    store = store or VectorStore()
    ws = load_exam(store=store, exam_id=exam_id)
    doc_ids = ws.doc_ids
    if not doc_ids:
        return []

    # 1) collect chunks
    chunks: List[StoredChunk] = []
    for doc_id in doc_ids:
        chunks.extend(store.list_chunks_by_doc(doc_id))
    if not chunks:
        return []

    chunk_ids = [c.chunk_id for c in chunks]
    chunk_by_id = {c.chunk_id: c for c in chunks}

    # 2) fetch vectors aligned to chunk_ids (skips any missing)
    resolved_ids, X = store.get_vectors_for_chunk_ids(chunk_ids)
    if X.size == 0 or not resolved_ids:
        return []
    id_to_row = {cid: i for i, cid in enumerate(resolved_ids)}

    # 3) cluster
    k = min(_pick_k(len(resolved_ids)), len(resolved_ids))
    assign = _kmeans_cluster(X, k=k, seed=seed)
    # Drop empty clusters and re-index to contiguous ids
    groups: Dict[int, List[str]] = {}
    for cid, a in zip(resolved_ids, assign.tolist()):
        groups.setdefault(int(a), []).append(cid)
    # Sort clusters by size descending for stable output.
    clusters = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)

    # 3.5) merge near-duplicate clusters by centroid similarity (cheap redundancy reduction)
    merged_clusters = _merge_clusters_by_centroid(
        clusters=clusters,
        X=X,
        id_to_row=id_to_row,
        threshold=merge_threshold,
    )

    built: List[BuiltTopic] = []
    for merged in merged_clusters:
        cids = list(merged["chunk_ids"])
        c_chunks = [chunk_by_id[cid] for cid in cids if cid in chunk_by_id]
        texts = [c.text for c in c_chunks if c and c.text]
        label = ""
        evidence: List[Dict[str, Any]] = []
        label_source = "ngram"

        if use_llm_labels:
            context_pack = _build_cluster_context_pack(
                cids=cids,
                chunk_by_id=chunk_by_id,
                X=X,
                id_to_row=id_to_row,
            )
            if context_pack:
                llm_out = _llm_label_cluster(context_pack=context_pack, model=llm_model)
                proposed_label = str(llm_out.get("label") or "").strip()
                phrases = list(llm_out.get("evidence_phrases") or [])
                ev, _matched = _extract_evidence_for_phrases(c_chunks, [proposed_label] + phrases)
                if ev and proposed_label:
                    label = proposed_label
                    evidence = ev
                    label_source = "llm"

        if not label:
            # Fallback: deterministic grounded n-gram label.
            label = _choose_grounded_label(texts)
            evidence = _extract_evidence_for_label(c_chunks, label)
            if not evidence:
                # Pick a label from the most frequent token (still grounded because it appears in text)
                label = _choose_grounded_label(texts, max_words=1)
                evidence = _extract_evidence_for_label(c_chunks, label)
        topic_id = uuid.uuid4().hex[:16]
        info = {
            "n_chunks": len(cids),
            "method": "kmeans_embeddings",
            "k": int(k),
            "seed": int(seed),
            "label_source": label_source,
            "merged_from_clusters": merged.get("cluster_ids", []),
            "merge_sims": merged.get("merge_sims", []),
            "merge_threshold": float(merge_threshold),
        }
        built.append(
            BuiltTopic(
                topic_id=topic_id,
                label=label,
                chunk_ids=cids,
                evidence=evidence,
                info=info,
            )
        )

    # 4) persist
    if overwrite:
        store.db.delete_topics_for_exam(exam_id=exam_id)
    for t in built:
        store.db.upsert_topic(topic_id=t.topic_id, exam_id=exam_id, label=t.label, info=t.info)
        store.db.replace_topic_chunks(topic_id=t.topic_id, chunk_ids=t.chunk_ids)
        store.db.replace_topic_evidence(topic_id=t.topic_id, evidence=t.evidence)

    return built


def list_topics_for_exam(*, exam_id: str, store: Optional[VectorStore] = None) -> List[StoredTopic]:
    store = store or VectorStore()
    return store.db.list_topics(exam_id=exam_id)


