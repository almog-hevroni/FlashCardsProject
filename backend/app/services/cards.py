from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.data.vector_store import VectorStore
from app.services.llm import CHAT_MODEL_FAST, chat_completions_create
from app.services.qa import generate_answer
from app.services.context_packs import build_representative_chunk_pack
from app.utils.vectors import l2_normalize
from app.services.graph import node_propose_questions


@dataclass
class GeneratedCard:
    card_id: str
    exam_id: str
    topic_id: str
    topic_label: str
    question: str
    answer: str
    difficulty: int
    proofs: List[Dict[str, Any]]


def _safe_json_load(s: str) -> Dict[str, Any]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def pick_starter_topics(
    *,
    exam_id: str,
    store: Optional[VectorStore] = None,
    n: int = 5,
) -> List[Tuple[str, str]]:
    """
    Pick N topics for starter flashcards.
    Current heuristic: highest n_chunks (stored in topics.info) first.
    Returns [(topic_id, label), ...].
    """
    store = store or VectorStore()
    topics = store.db.list_topics(exam_id=exam_id)
    scored: List[Tuple[int, str, str]] = []
    for t in topics:
        n_chunks = int(t.info.get("n_chunks") or 0)
        scored.append((n_chunks, t.topic_id, t.label))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(topic_id, label) for _, topic_id, label in scored[: max(0, int(n))]]


def generate_starter_question(
    *,
    topic_label: str,
    context_pack: str,
    model: str = CHAT_MODEL_FAST,
) -> str:
    """
    Generate one easy diagnostic flashcard question for a topic, grounded in the provided excerpts.
    """
    sys_prompt = (
        "You write high-quality flashcard questions.\n"
        "You must ground the question in the provided excerpts and avoid generic questions.\n"
        "Return JSON only."
    )
    user_prompt = (
        f"TOPIC LABEL:\n{topic_label}\n\n"
        "EXCERPTS (from the document, same topic):\n"
        f"{context_pack}\n\n"
        "Create ONE easy diagnostic flashcard question that tests basic understanding of this topic.\n"
        "Rules:\n"
        "- The question must be answerable using ONLY the excerpts.\n"
        "- Ask about a core definition, relationship, or key claim (not trivia).\n"
        "- Avoid referencing 'the excerpt' or 'the paper' in the question.\n"
        "- Return JSON: {\"question\": \"...\"}\n"
    )
    resp = chat_completions_create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=180,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    data = _safe_json_load(raw)
    q = str(data.get("question") or "").strip()
    return q


def generate_starter_cards(
    *,
    exam_id: str,
    user_id: str,
    store: Optional[VectorStore] = None,
    n: int = 5,
    difficulty: int = 1,
    model_fast: str = CHAT_MODEL_FAST,
    k: int = 8,
    min_score: float = 0.4,
) -> List[GeneratedCard]:
    """
    Generate N starter cards across distinct topics:
    - pick topics
    - generate one easy question per topic (LLM-fast, grounded in topic excerpts)
    - answer with topic-scoped Teacher (generate_answer with allowed_chunk_ids)
    - persist cards + proofs
    """
    store = store or VectorStore()
    picked = pick_starter_topics(exam_id=exam_id, store=store, n=n)

    # Avoid duplicating starter cards for the same topic if rerun.
    existing = store.db.list_cards_for_exam(exam_id=exam_id)
    existing_topic_ids = set()
    for c in existing:
        if c.topic_id and c.info.get("type") == "starter":
            existing_topic_ids.add(c.topic_id)

    out: List[GeneratedCard] = []

    # Fallback: if no topics exist, generate doc-level starter cards and attach them to a single
    # "Document Overview" topic that spans all chunks in the exam.
    if not picked:
        doc_ids = store.db.list_exam_documents(exam_id=exam_id)
        if not doc_ids:
            return []
        all_chunk_ids: List[str] = []
        for doc_id in doc_ids:
            for ch in store.list_chunks_by_doc(doc_id):
                if ch and ch.chunk_id:
                    all_chunk_ids.append(ch.chunk_id)
        if not all_chunk_ids:
            return []

        overview_topic_id = uuid.uuid4().hex[:16]
        overview_label = "Document Overview"
        store.db.upsert_topic(
            topic_id=overview_topic_id,
            exam_id=exam_id,
            label=overview_label,
            info={"type": "fallback_overview", "n_chunks": len(all_chunk_ids)},
        )
        store.db.replace_topic_chunks(topic_id=overview_topic_id, chunk_ids=all_chunk_ids)

        # Persist a centroid vector for the overview topic (helps future routing/packs)
        resolved, X = store.get_vectors_for_chunk_ids(all_chunk_ids)
        if X.size and resolved:
            centroid = l2_normalize(l2_normalize(X).mean(axis=0, keepdims=True))[0]
            store.db.upsert_topic_vector(topic_id=overview_topic_id, vector=centroid)

        # Use the existing doc-level proposer as the fallback generator (already deduped/scored).
        init_state = {
            "doc_ids": doc_ids,
            "store_basepath": str(store.base),
            "k": 10,
            "min_score": 0.4,
            "target_n": int(n),
            "attempts": 0,
            "max_attempts": 1,
            "questions": [],
            "question_scores": [],
            "current_index": 0,
            "qa": [],
            "working_answer": None,
            "working_proofs": None,
            "working_score": None,
            "working_answer_critique": None,
            "last_critique": None,
            "summary": "",
            "retrieval_cache": {},
        }
        q_state = node_propose_questions(init_state)  # type: ignore
        questions = list(q_state.get("questions") or [])[: int(n)]
        if not questions:
            return []

        for q in questions:
            ans = generate_answer(
                question=q,
                k=k,
                min_score=min_score,
                store=store,
                allowed_chunk_ids=all_chunk_ids,
            )
            card_id = uuid.uuid4().hex[:16]
            store.db.upsert_card(
                card_id=card_id,
                exam_id=exam_id,
                topic_id=overview_topic_id,
                question=q,
                answer=ans.answer,
                difficulty=int(difficulty),
                status="active",
                info={"type": "starter", "topic_label": overview_label, "user_id": user_id, "fallback": True},
            )
            store.db.replace_card_proofs(
                card_id=card_id,
                proofs=[
                    {
                        "doc_id": p.doc_id,
                        "page": p.page,
                        "start": p.start,
                        "end": p.end,
                        "text": (p.text or ""),
                        "score": float(p.score or 0.0),
                    }
                    for p in (ans.proofs or [])
                ],
            )
            out.append(
                GeneratedCard(
                    card_id=card_id,
                    exam_id=exam_id,
                    topic_id=overview_topic_id,
                    topic_label=overview_label,
                    question=q,
                    answer=ans.answer,
                    difficulty=int(difficulty),
                    proofs=[p.model_dump() for p in (ans.proofs or [])],
                )
            )
        return out[: int(n)]

    for topic_id, topic_label in picked:
        if topic_id in existing_topic_ids:
            continue
        allowed_chunk_ids = store.db.list_chunk_ids_for_topic(topic_id=topic_id)
        if not allowed_chunk_ids:
            continue
        centroid = store.db.get_topic_vector(topic_id=topic_id)
        context_pack = build_representative_chunk_pack(
            store=store,
            chunk_ids=allowed_chunk_ids,
            centroid=centroid,
        )
        if not context_pack:
            continue
        question = generate_starter_question(topic_label=topic_label, context_pack=context_pack, model=model_fast)
        if not question:
            continue

        ans = generate_answer(
            question=question,
            k=k,
            min_score=min_score,
            store=store,
            allowed_chunk_ids=allowed_chunk_ids,
        )

        card_id = uuid.uuid4().hex[:16]
        store.db.upsert_card(
            card_id=card_id,
            exam_id=exam_id,
            topic_id=topic_id,
            question=question,
            answer=ans.answer,
            difficulty=int(difficulty),
            status="active",
            info={"type": "starter", "topic_label": topic_label, "user_id": user_id},
        )
        store.db.replace_card_proofs(
            card_id=card_id,
            proofs=[
                {
                    "doc_id": p.doc_id,
                    "page": p.page,
                    "start": p.start,
                    "end": p.end,
                    "text": (p.text or ""),
                    "score": float(p.score or 0.0),
                }
                for p in (ans.proofs or [])
            ],
        )
        out.append(
            GeneratedCard(
                card_id=card_id,
                exam_id=exam_id,
                topic_id=topic_id,
                topic_label=topic_label,
                question=question,
                answer=ans.answer,
                difficulty=int(difficulty),
                proofs=[p.model_dump() for p in (ans.proofs or [])],
            )
        )
        if len(out) >= int(n):
            break
    return out


