from typing import Any, Dict, List, Optional, Sequence, TypedDict
import hashlib
import json
import numpy as np
from langgraph.graph import StateGraph, END
from app.cache import DocContextSummaryCache
from app.llm import client, CHAT_MODEL, embed_texts
from app.answer import AnswerWithCitations
from store.storage import VectorStore
from app.models import ProofSpan
from app.api import retrieve_with_proofs

# ------------- State -------------

class QAItem(TypedDict, total=False):
    question: str
    answer: str
    proofs: List[Dict[str, Any]]
    question_score: float
    answer_score: float
    forced: bool
    critique: str

class State(TypedDict):
    doc_ids: List[str]
    store_basepath: str
    k: int
    min_score: float
    target_n: int
    attempts: int
    max_attempts: int
    questions: List[str]
    question_scores: List[float]
    current_index: int
    qa: List[QAItem]
    working_answer: Optional[str]
    working_proofs: Optional[List[ProofSpan]]
    last_critique: Optional[str]
    summary: str
    retrieval_cache: Dict[int, Dict[str, Any]]

# ------------- Helpers -------------

def _openai():
    return client()

def _sample_doc_context(
    store: VectorStore,
    doc_ids: Sequence[str],
    n: int = 20,
    seed: Optional[int] = None,
) -> str:
    if not doc_ids:
        return ""
    import random
    rng = random.Random(seed) if seed is not None else random
    pooled = []
    for doc_id in doc_ids:
        pooled.extend(store.sample_chunks_by_doc(doc_id, n=n))
    if not pooled:
        return ""
    rng.shuffle(pooled)
    selected = pooled[:n]
    buf = []
    for c in selected:
        buf.append(f"Page {c.page}\n{c.text.strip()}\n")
    return "\n\n".join(buf)[:12000]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embed_with_store_cache(texts: List[str], store: VectorStore) -> np.ndarray:
    if not texts:
        return np.zeros((0, store.vector_dimension), dtype="float32")
    hashes = [_hash_text(t) for t in texts]
    cached = store.get_cached_embeddings(hashes)
    missing_idx = [i for i, h in enumerate(hashes) if h not in cached]
    if missing_idx:
        missing_texts = [texts[i] for i in missing_idx]
        new_vectors = embed_texts(missing_texts)
        for idx, vec in zip(missing_idx, new_vectors):
            cached[hashes[idx]] = vec.astype("float32", copy=False)
        store.add_cached_embeddings({hashes[idx]: cached[hashes[idx]] for idx in missing_idx})
    vectors = np.stack([cached[h] for h in hashes]).astype("float32", copy=False)
    return vectors


def _get_context_and_summary(store: VectorStore, doc_ids: Sequence[str], basepath: str) -> tuple[str, str]:
    cache = DocContextSummaryCache(basepath)
    contexts: List[str] = []
    summaries: List[str] = []
    for doc_id in doc_ids:
        cached = cache.get(doc_id)
        if cached:
            context = cached.get("context", "")
            summary = cached.get("summary", "")
        else:
            seed = int(_hash_text(doc_id), 16) % (2**32)
            context = _sample_doc_context(store, [doc_id], n=24, seed=seed)
            summary = _summarize_context(context)
            cache.set(doc_id, context, summary)
        if context:
            contexts.append(context)
        if summary:
            summaries.append(summary)
    combined_context = "\n\n".join(contexts)[:12000]
    combined_summary = "\n\n".join(summaries) or "Summary unavailable"
    return combined_context, combined_summary


def _get_or_build_retrieval_pool(
    question: str,
    doc_ids: Sequence[str],
    store: VectorStore,
    desired_pool: int,
    cache_entry: Optional[Dict[str, Any]],
) -> tuple[List[ProofSpan], Dict[str, Any]]:
    allowed = set(doc_ids or [])
    cached_pool: Optional[List[ProofSpan]] = None
    if cache_entry:
        pool = cache_entry.get("pool")
        pool_size = cache_entry.get("pool_size", len(pool) if isinstance(pool, list) else 0)
        if isinstance(pool, list) and pool_size >= desired_pool:
            cached_pool = pool
    if cached_pool is not None:
        return cached_pool, dict(cache_entry or {"pool": cached_pool, "pool_size": desired_pool})

    hits = retrieve_with_proofs(question, k=desired_pool, store=store)
    if allowed:
        hits = [h for h in hits if h.doc_id in allowed]
    new_entry = {"pool": hits, "pool_size": desired_pool}
    return hits, new_entry


def _summarize_context(context: str) -> str:
    prompt = (
        "Read the document excerpt below and summarize the primary topics, goals, and insights "
        "that a learner should focus on. Ignore side anecdotes or illustrative examples that are not central.\n\n"
        f"CONTEXT:\n{context}"
    )
    resp = _openai().chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You write concise study guides."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=250,
    )
    text = resp.choices[0].message.content or ""
    return text.strip()

def _propose_questions_from_context(context: str, summary: str, n: int = 6) -> List[str]:
    prompt = (
        "You are designing study questions strictly about the document described below.\n\n"
        f"DOCUMENT SUMMARY:\n{summary}\n\n"
        "DOCUMENT EXCERPT:\n"
        f"{context}\n\n"
        f"Produce {n} distinct, high-impact questions that help a learner internalize the main concepts, findings, "
        "methods, and implications. Follow the rules:\n"
        "- Focus on the primary themes from the summary; ignore tangential anecdotes or passing examples.\n"
        "- Prefer questions that require understanding relationships, reasoning, or key arguments over rote facts.\n"
        "- Avoid redundant or near-duplicate questions.\n"
        "- Avoid questions that could be answered without reading the document, or that reference minor illustrations.\n"
        "Return exactly one question per line, with no numbering or bullet characters."
    )
    resp = _openai().chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You write excellent study questions."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=350,
    )
    text = resp.choices[0].message.content or ""
    qs = [q.strip() for q in text.splitlines() if q.strip()]
    if len(qs) > n:
        qs = qs[:n]
    return qs

def _dedupe_questions(
    questions: List[str],
    threshold: float = 0.85,
    store: Optional[VectorStore] = None,
) -> List[str]:
    if len(questions) <= 1:
        return questions
    try:
        if store:
            vecs = _embed_with_store_cache(questions, store)
        else:
            vecs = embed_texts(questions)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.clip(norms, 1e-12, None)
        kept: List[str] = []
        kept_indices: List[int] = []
        for i, q in enumerate(questions):
            if not kept_indices:
                kept.append(q)
                kept_indices.append(i)
                continue
            sims = vecs[i][kept_indices]
            if sims.size > 0 and np.max(sims) >= threshold:
                continue
            kept.append(q)
            kept_indices.append(i)
        return kept
    except Exception:
        seen = set()
        unique = []
        for q in questions:
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(q)
        return unique

def _build_question_pool(
    raw_questions: List[str],
    target_count: int,
    store: Optional[VectorStore] = None,
) -> List[str]:
    """
    Ensure we have at least target_count question candidates.
    Start with aggressive deduplication and gradually relax if we fall short.
    """
    cleaned = [q.strip() for q in raw_questions if q.strip()]
    if target_count <= 0:
        return []
    if not cleaned:
        return []
    thresholds = [0.85, 0.9, 0.93, 0.97, 0.99]
    for thresh in thresholds:
        deduped = _dedupe_questions(cleaned, threshold=thresh, store=store)
        if len(deduped) >= target_count:
            return deduped
    deduped = _dedupe_questions(cleaned, threshold=0.99, store=store)
    seen_lower = {q.lower() for q in deduped}
    for q in cleaned:
        if len(deduped) >= target_count:
            break
        key = q.lower()
        if key in seen_lower:
            continue
        deduped.append(q)
        seen_lower.add(key)
    if len(deduped) < target_count:
        idx = 0
        while len(deduped) < target_count:
            deduped.append(cleaned[idx % len(cleaned)])
            idx += 1
    return deduped[:max(target_count, len(deduped))]

def _score_questions(context: str, summary: str, questions: List[str]) -> List[Dict[str, Any]]:
    if not questions:
        return []
    payload = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    prompt = (
        "Document summary:\n"
        f"{summary}\n\n"
        "Document excerpt:\n"
        f"{context}\n\n"
        "Questions to evaluate:\n"
        f"{payload}\n\n"
        "For each question, evaluate:\n"
        "1. Centrality: Does it target the main ideas, methods, or findings described in the summary?\n"
        "2. Educational value: Will answering it deepen understanding of the document's core content?\n"
        "3. Redundancy: Is it meaningfully different from the other questions listed?\n\n"
        "Return a JSON array with one object per question, in order, using this schema:\n"
        '[{"score": <float 0-1>, "central": <true/false>, "reason": "<short explanation>"}]\n'
        "Score near 0 if a question is primarily about tangential examples or trivia.\n"
        "Respond with JSON only."
    )
    resp = _openai().chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a rigorous evaluator."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=400,
    )
    raw = resp.choices[0].message.content or "[]"
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            data = []
    except Exception:
        data = []
    out: List[Dict[str, Any]] = []
    for i in range(len(questions)):
        item = {"score": 0.5, "central": True, "reason": ""}
        try:
            entry = data[i]
            if isinstance(entry, dict):
                if "score" in entry:
                    item["score"] = float(entry["score"])
                if "central" in entry:
                    item["central"] = bool(entry["central"])
                if "reason" in entry:
                    item["reason"] = str(entry["reason"])
        except Exception:
            pass
        out.append(item)
    return out

def _select_questions(questions: List[str], evaluations: List[Dict[str, Any]], top_n: int = 3,
                      min_score: float = 0.55) -> List[tuple[str, float]]:
    ranked: List[tuple[str, float, bool]] = []
    for q, ev in zip(questions, evaluations):
        score = float(ev.get("score", 0.5))
        central = bool(ev.get("central", False))
        ranked.append((q, score, central))
    # Prioritize central questions, then by score
    ranked.sort(key=lambda x: (x[2], x[1]), reverse=True)

    selected: List[tuple[str, float]] = []
    # First pass: central questions above threshold
    for q, score, central in ranked:
        if len(selected) >= top_n:
            break
        if central and score >= min_score:
            selected.append((q, score))
    # Second pass: remaining central questions even if slightly below threshold
    if len(selected) < top_n:
        for q, score, central in ranked:
            if len(selected) >= top_n:
                break
            if central and (q, score) not in selected:
                selected.append((q, score))
    # Final pass: allow highest scoring non-central questions only if still short
    if len(selected) < top_n:
        for q, score, central in ranked:
            if len(selected) >= top_n:
                break
            if not central and (q, score) not in selected:
                selected.append((q, score))
    return selected[:top_n]

def _answer_with_citations(
    question: str,
    k: int,
    min_score: float,
    doc_ids: List[str],
    store_basepath: str,
    prefetched_pool: Optional[List[ProofSpan]] = None,
    pool_k: Optional[int] = None,
    store: Optional[VectorStore] = None,
) -> AnswerWithCitations:
    from app.answer import generate_answer
    store = store or VectorStore(basepath=store_basepath)
    extra_kwargs = {
        "prefetched_pool": prefetched_pool,
        "pool_k": pool_k,
    }
    if len(doc_ids) == 1:
        return generate_answer(
            question=question,
            k=k,
            min_score=min_score,
            doc_id=doc_ids[0],
            store=store,
            **extra_kwargs,
        )
    return generate_answer(
        question=question,
        k=k,
        min_score=min_score,
        allowed_doc_ids=doc_ids,
        store=store,
        **extra_kwargs,
    )

def _validate_answer(question: str, answer: str, proofs: List[ProofSpan]) -> Dict[str, Any]:
    sources = []
    for i, p in enumerate(proofs, 1):
        sources.append(f"S{i}: doc={p.doc_id} page={p.page} score={p.score:.2f}\n{p.text}")
    src = "\n\n".join(sources)
    prompt = (
        "Evaluate the answer strictly for grounding and completeness using the sources.\n"
        "Return JSON with fields: score (0.0-1.0), critique (string). "
        "Penalize missing or incorrect citations.\n\n"
        f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nSOURCES:\n{src}"
    )
    try:
        resp = _openai().chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "system", "content": "You are a strict fact-checker."},
                      {"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
    except Exception:
        resp = _openai().chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "system", "content": "You are a strict fact-checker."},
                      {"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except Exception:
        data = {"score": 0.5, "critique": "Could not parse evaluation."}
    if "score" not in data:
        data["score"] = 0.5
    if "critique" not in data:
        data["critique"] = "No critique provided."
    return data

def _format_report(items: List[QAItem]) -> str:
    lines = []
    for i, it in enumerate(items, 1):
        lines.append(f"Q{i}: {it['question']}\n")
        lines.append(f"Answer:\n{it['answer']}\n")
        score = it.get("answer_score", 0.0)
        forced = it.get("forced", False)
        flag = " (forced accept)" if forced else ""
        lines.append(f"Answer score: {score:.2f}{flag}")
        critique = it.get("critique")
        if critique:
            lines.append(f"Critique: {critique}")
        lines.append("Proofs:")
        for j, p in enumerate(it["proofs"], 1):
            lines.append(f"- S{j} | doc={p['doc_id']} page={p['page']} score={p['score']:.2f}")
            text = " ".join((p.get("text") or "").split())
            if text:
                lines.append(f'  "{text}"')
            else:
                lines.append("  (no proof text)")
        lines.append("")
    return "\n".join(lines).strip()

# ------------- Nodes -------------

def node_propose_questions(state: State) -> State:
    store = VectorStore(basepath=state["store_basepath"])
    context, summary = _get_context_and_summary(store, state["doc_ids"], state["store_basepath"])
    candidate_count = max(state["target_n"] * 4, 12)
    raw_qs = _propose_questions_from_context(context, summary, n=candidate_count)
    deduped = _build_question_pool(raw_qs, state["target_n"], store=store)
    evaluations = _score_questions(context, summary, deduped)
    selected = _select_questions(deduped, evaluations, top_n=state["target_n"])
    questions = [q for q, _ in selected]
    question_scores = [score for _, score in selected]
    if not questions:
        fallback = deduped[:state["target_n"]]
        questions = fallback
        question_scores = [0.5] * len(fallback)
    return {
        **state,
        "questions": questions,
        "question_scores": question_scores,
        "current_index": 0,
        "qa": [],
        "attempts": 0,
        "summary": summary,
    }

def node_answer(state: State) -> State:
    idx = state["current_index"]
    q = state["questions"][idx]
    store = VectorStore(basepath=state["store_basepath"])
    retrieval_cache = state.get("retrieval_cache", {})
    cache_entry = retrieval_cache.get(idx)
    desired_pool = max(state["k"] + 6, 16)
    pool, updated_entry = _get_or_build_retrieval_pool(
        q,
        state["doc_ids"],
        store,
        desired_pool,
        cache_entry,
    )
    ans = _answer_with_citations(
        q,
        k=state["k"],
        min_score=state["min_score"],
        doc_ids=state["doc_ids"],
        store_basepath=state["store_basepath"],
        prefetched_pool=pool,
        pool_k=desired_pool,
        store=store,
    )
    retrieval_cache[idx] = updated_entry
    return {
        **state,
        "working_answer": ans.answer,
        "working_proofs": ans.proofs,
        "retrieval_cache": retrieval_cache,
    }

def node_judge(state: State) -> State:
    idx = state["current_index"]
    q = state["questions"][idx]
    answer = state["working_answer"] or ""
    proofs = state["working_proofs"] or []
    judge = _validate_answer(q, answer, proofs)
    score = float(judge.get("score", 0.5))
    critique = str(judge.get("critique", ""))
    accepted = score >= 0.7
    attempts_used = state["attempts"] + 1
    force_accept = not accepted and attempts_used >= state["max_attempts"]
    qa_list = state["qa"][:]
    if accepted or force_accept:
        qa_entry: QAItem = {
            "question": q,
            "answer": answer,
            "proofs": [p.model_dump() for p in proofs],
            "question_score": state["question_scores"][idx] if idx < len(state["question_scores"]) else 0.5,
            "answer_score": score,
        }
        if force_accept:
            qa_entry["forced"] = True
            if critique:
                qa_entry["critique"] = critique
        elif critique:
            qa_entry["critique"] = critique
        qa_list.append(qa_entry)
    return {
        **state,
        "last_critique": critique,
        "attempts": state["attempts"] + 1,
        "qa": qa_list,
    }

def should_retry(state: State) -> str:
    idx = state["current_index"]
    accepted_for_this_idx = len(state["qa"]) > idx
    if accepted_for_this_idx:
        return "accept"
    if state["attempts"] < state["max_attempts"]:
        return "retry"
    return "accept"

def node_strengthen(state: State) -> State:
    return {
        **state,
        "k": min(12, state["k"] + 2),
        "min_score": max(0.25, state["min_score"] - 0.05),
    }

def node_next_or_finish(state: State) -> State:
    if len(state["qa"]) >= state["target_n"]:
        return state
    if state["current_index"] + 1 < len(state["questions"]):
        return {
            **state,
            "current_index": state["current_index"] + 1,
            "attempts": 0,
            "working_answer": None,
            "working_proofs": None,
            "last_critique": None,
            "k": 10,
            "min_score": 0.4,
        }
    return {
        **state,
        "current_index": len(state["questions"]),
        "attempts": 0,
        "working_answer": None,
        "working_proofs": None,
        "last_critique": None,
        "k": 10,
        "min_score": 0.4,
    }

# ------------- Build graph -------------

def build_graph():
    g = StateGraph(State)
    g.add_node("propose_questions", node_propose_questions)
    g.add_node("answer", node_answer)
    g.add_node("judge", node_judge)
    g.add_node("strengthen", node_strengthen)
    g.add_node("next_or_finish", node_next_or_finish)

    g.set_entry_point("propose_questions")
    g.add_edge("propose_questions", "answer")
    g.add_edge("answer", "judge")
    g.add_conditional_edges("judge", should_retry, {
        "retry": "strengthen",
        "accept": "next_or_finish",
    })
    g.add_edge("strengthen", "answer")

    def done_or_continue(state: State) -> str:
        if len(state["qa"]) >= state["target_n"]:
            return END
        if state["current_index"] >= len(state["questions"]):
            return END
        return "answer"

    g.add_conditional_edges("next_or_finish", done_or_continue, {
        "answer": "answer",
        END: END,
    })
    return g.compile()

# ------------- Runner -------------

def run_generate_qa(doc_ids: Sequence[str], num_questions: int = 5, store_basepath: str = "store") -> Dict[str, Any]:
    graph = build_graph()
    doc_id_list = [d for d in doc_ids if d]
    if not doc_id_list:
        return {"doc_ids": [], "n": 0, "items": [], "report": ""}
    init: State = {
        "doc_ids": doc_id_list,
        "store_basepath": store_basepath,
        "k": 10,
        "min_score": 0.4,
        "target_n": num_questions,
        "attempts": 0,
        "max_attempts": 3,
        "questions": [],
        "question_scores": [],
        "current_index": 0,
        "qa": [],
        "working_answer": None,
        "working_proofs": None,
        "last_critique": None,
        "summary": "",
        "retrieval_cache": {},
    }
    final_state = graph.invoke(init)
    report = _format_report(final_state["qa"])
    return {
        "doc_ids": doc_id_list,
        "n": len(final_state["qa"]),
        "items": final_state["qa"],
        "report": report,
    }

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--doc_id", nargs="+", help="Existing document id(s) in the store")
    p.add_argument("--path", help="Path to a document to ingest first")
    p.add_argument("--n", type=int, default=5)
    args = p.parse_args()
    store = VectorStore()
    if args.path:
        from app.api import ingest_documents
        res = ingest_documents([args.path], store=store)[0]
        doc_ids = [res.doc_id]
        print(f"Ingested doc_id={doc_ids[0]}")
    else:
        if not args.doc_id:
            raise SystemExit("Either --path or --doc_id must be provided")
        doc_ids = args.doc_id
    out = run_generate_qa(doc_ids=doc_ids, num_questions=args.n, store_basepath=str(store.base))
    print("\n=== AUTO-QA REPORT ===\n")
    print(out["report"])

