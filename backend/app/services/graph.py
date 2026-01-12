"""
Card Generation LangGraph Flow

Topic-scoped, difficulty-aware card generation with:
- FAISS-based question deduplication (exam-scoped)
- Bloom's taxonomy difficulty levels
- Robust retry logic (full restart on persistent failures)
"""

from typing import Any, Dict, List, Optional, Sequence, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
import uuid
import numpy as np
from langgraph.graph import StateGraph, END

from app.data.vector_store import VectorStore, QuestionIndex
from app.services.llm import (
    chat_completions_create,
    CHAT_MODEL,
    CHAT_MODEL_FAST,
    embed_texts,
)
from app.services.qa import generate_answer
from app.services.cards import (
    GeneratedCard,
    generate_question_at_difficulty,
    pick_starter_topics,
    DIFFICULTY_LEVELS,
)
from app.services.context_packs import build_diverse_chunk_pack
from app.api.schemas import ProofSpan

logger = logging.getLogger(__name__)

# ------------- Configuration -------------

DEFAULT_CONFIG = {
    "max_question_attempts": 3,
    "max_answer_attempts": 3,
    "max_full_restarts": 5,
    "uniqueness_threshold": 0.85,
    "validation_threshold": 0.7,
    "initial_k": 8,
    "initial_min_score": 0.4,
    "strengthen_k_delta": 2,
    "strengthen_min_score_delta": 0.05,
}

# ------------- Helpers (kept from old implementation) -------------


def _hash_text(text: str) -> str:
    """Hash text for embedding cache key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embed_with_store_cache(texts: List[str], store: VectorStore) -> np.ndarray:
    """Embed texts using the store's embedding cache."""
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


def _validate_answer(question: str, answer: str, proofs: List[ProofSpan]) -> Dict[str, Any]:
    """
    Validate answer groundedness against source proofs.
    Returns dict with 'score' (0.0-1.0) and 'critique' (string).
    """
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
        resp = chat_completions_create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict fact-checker."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
    except Exception:
        resp = chat_completions_create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict fact-checker."},
                {"role": "user", "content": prompt},
            ],
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


# ------------- State -------------


class CardGenState(TypedDict):
    """State for single card generation flow."""
    
    # Input
    exam_id: str
    user_id: str
    store_basepath: str
    
    # Current topic
    topic_id: str
    topic_label: str
    difficulty: int
    allowed_chunk_ids: List[str]
    context_pack: str
    
    # Question state
    question: Optional[str]
    question_embedding: Optional[np.ndarray]
    question_vector_idx: Optional[int]
    is_unique: bool
    
    # Answer state
    answer: Optional[str]
    proofs: Optional[List[ProofSpan]]
    validation_score: Optional[float]
    validation_critique: Optional[str]
    
    # Retry tracking
    question_attempts: int
    answer_attempts: int
    full_restart_count: int
    
    # Limits
    max_question_attempts: int
    max_answer_attempts: int
    max_full_restarts: int
    
    # Retrieval params (for strengthening)
    k: int
    min_score: float
    
    # Flow control
    stop_after_embedding: bool  # For parallel starter pack: stop after Phase 1
    
    # Output
    card: Optional[GeneratedCard]


# ------------- Nodes -------------


def node_generate_question(state: CardGenState) -> CardGenState:
    """Generate a question at the specified difficulty level."""
    question = generate_question_at_difficulty(
        topic_label=state["topic_label"],
        context_pack=state["context_pack"],
        difficulty=state["difficulty"],
    )
    return {
        **state,
        "question": question,
        "question_attempts": state["question_attempts"] + 1,
    }


def node_embed_question(state: CardGenState) -> CardGenState:
    """Embed the generated question."""
    store = VectorStore(basepath=state["store_basepath"])
    embedding = _embed_with_store_cache([state["question"]], store)[0]
    return {
        **state,
        "question_embedding": embedding,
    }


def node_check_uniqueness(state: CardGenState) -> CardGenState:
    """
    Check if question is semantically unique within this exam.
    Uses FAISS search for O(log n) similarity lookup.
    """
    question_index = QuestionIndex(basepath=state["store_basepath"])
    store = VectorStore(basepath=state["store_basepath"])
    
    # If index is empty, question is unique
    if question_index.size() == 0:
        return {**state, "is_unique": True}
    
    # Search for similar questions
    query_vec = state["question_embedding"]
    D, I = question_index.search(query_vec, k=20)
    
    # Get mapping of vector_idx -> card for this exam
    exam_cards = store.db.get_cards_with_question_vector_idx(exam_id=state["exam_id"])
    
    # Check if any result is from the same exam with similarity >= threshold
    threshold = DEFAULT_CONFIG["uniqueness_threshold"]
    for sim, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        # Only check cards from the same exam
        if int(idx) in exam_cards and sim >= threshold:
            logger.debug(
                "Question duplicate found: sim=%.3f, threshold=%.3f",
                sim, threshold
            )
            return {**state, "is_unique": False}
    
    return {**state, "is_unique": True}


def node_store_embedding(state: CardGenState) -> CardGenState:
    """Store the question embedding in the question index."""
    question_index = QuestionIndex(basepath=state["store_basepath"])
    vector_idx = question_index.add(state["question_embedding"])
    return {
        **state,
        "question_vector_idx": vector_idx,
    }


def node_generate_answer(state: CardGenState) -> CardGenState:
    """Generate answer using topic-scoped retrieval."""
    store = VectorStore(basepath=state["store_basepath"])
    
    result = generate_answer(
        question=state["question"],
        k=state["k"],
        min_score=state["min_score"],
        store=store,
        allowed_chunk_ids=state["allowed_chunk_ids"],
    )
    
    return {
        **state,
        "answer": result.answer,
        "proofs": result.proofs,
        "answer_attempts": state["answer_attempts"] + 1,
    }


def node_validate(state: CardGenState) -> CardGenState:
    """Validate the answer for groundedness."""
    validation = _validate_answer(
        question=state["question"],
        answer=state["answer"],
        proofs=state["proofs"] or [],
    )
    return {
        **state,
        "validation_score": float(validation.get("score", 0.5)),
        "validation_critique": str(validation.get("critique", "")),
    }


def node_strengthen(state: CardGenState) -> CardGenState:
    """Strengthen retrieval parameters for retry."""
    return {
        **state,
        "k": state["k"] + DEFAULT_CONFIG["strengthen_k_delta"],
        "min_score": max(0.2, state["min_score"] - DEFAULT_CONFIG["strengthen_min_score_delta"]),
    }


def node_store_card(state: CardGenState) -> CardGenState:
    """Store the validated card in the database."""
    store = VectorStore(basepath=state["store_basepath"])
    
    card_id = uuid.uuid4().hex[:16]
    
    # Store card with question_vector_idx in info
    store.db.upsert_card(
        card_id=card_id,
        exam_id=state["exam_id"],
        topic_id=state["topic_id"],
        question=state["question"],
        answer=state["answer"],
        difficulty=state["difficulty"],
        status="active",
        info={
            "question_vector_idx": state["question_vector_idx"],
            "topic_label": state["topic_label"],
            "user_id": state["user_id"],
            "validation_score": state["validation_score"],
        },
    )
    
    # Store proofs
    proofs_data = [
        {
            "doc_id": p.doc_id,
            "page": p.page,
            "start": p.start,
            "end": p.end,
            "text": p.text or "",
            "score": float(p.score or 0.0),
        }
        for p in (state["proofs"] or [])
    ]
    store.db.replace_card_proofs(card_id=card_id, proofs=proofs_data)
    
    # Create output card
    card = GeneratedCard(
        card_id=card_id,
        exam_id=state["exam_id"],
        topic_id=state["topic_id"],
        topic_label=state["topic_label"],
        question=state["question"],
        answer=state["answer"],
        difficulty=state["difficulty"],
        proofs=proofs_data,
    )
    
    return {**state, "card": card}


def node_full_restart(state: CardGenState) -> CardGenState:
    """Reset state for full restart with new question."""
    return {
        **state,
        "question": None,
        "question_embedding": None,
        "question_vector_idx": None,
        "is_unique": False,
        "answer": None,
        "proofs": None,
        "validation_score": None,
        "validation_critique": None,
        "question_attempts": 0,
        "answer_attempts": 0,
        "full_restart_count": state["full_restart_count"] + 1,
        "k": DEFAULT_CONFIG["initial_k"],
        "min_score": DEFAULT_CONFIG["initial_min_score"],
    }


# ------------- Conditional Edges -------------


def decide_after_uniqueness(state: CardGenState) -> str:
    """Decide next step after uniqueness check."""
    if state["is_unique"]:
        return "store_embedding"
    
    if state["question_attempts"] < state["max_question_attempts"]:
        return "regenerate_question"
    
    # Max question retries reached, do full restart
    if state["full_restart_count"] < state["max_full_restarts"]:
        return "full_restart"
    
    # Absolute max reached - give up
    return "end"


def decide_after_validation(state: CardGenState) -> str:
    """Decide next step after answer validation."""
    threshold = DEFAULT_CONFIG["validation_threshold"]
    
    if state["validation_score"] >= threshold:
        return "store_card"
    
    if state["answer_attempts"] < state["max_answer_attempts"]:
        return "strengthen"
    
    # Max answer retries reached, do full restart
    if state["full_restart_count"] < state["max_full_restarts"]:
        return "full_restart"
    
    # Absolute max reached - give up
    return "end"


def decide_after_restart(state: CardGenState) -> str:
    """Decide next step after full restart."""
    if state["full_restart_count"] < state["max_full_restarts"]:
        return "generate_question"
    return "end"


def decide_after_store_embedding(state: CardGenState) -> str:
    """Decide whether to continue to answering or stop (for parallel starter pack)."""
    if state.get("stop_after_embedding", False):
        return "end"
    return "generate_answer"


# ------------- Graph Builder -------------


def build_card_graph():
    """Build the LangGraph for single card generation."""
    g = StateGraph(CardGenState)
    
    # Add nodes
    g.add_node("generate_question", node_generate_question)
    g.add_node("embed_question", node_embed_question)
    g.add_node("check_uniqueness", node_check_uniqueness)
    g.add_node("store_embedding", node_store_embedding)
    g.add_node("generate_answer", node_generate_answer)
    g.add_node("validate", node_validate)
    g.add_node("strengthen", node_strengthen)
    g.add_node("store_card", node_store_card)
    g.add_node("full_restart", node_full_restart)
    
    # Set entry point
    g.set_entry_point("generate_question")
    
    # Linear edges
    g.add_edge("generate_question", "embed_question")
    g.add_edge("embed_question", "check_uniqueness")
    g.add_edge("generate_answer", "validate")
    g.add_edge("strengthen", "generate_answer")
    g.add_edge("store_card", END)
    
    # Conditional edge after store_embedding (for parallel starter pack)
    g.add_conditional_edges(
        "store_embedding",
        decide_after_store_embedding,
        {
            "generate_answer": "generate_answer",
            "end": END,
        },
    )
    
    # Conditional edges after uniqueness check
    g.add_conditional_edges(
        "check_uniqueness",
        decide_after_uniqueness,
        {
            "store_embedding": "store_embedding",
            "regenerate_question": "generate_question",
            "full_restart": "full_restart",
            "end": END,
        },
    )
    
    # Conditional edges after validation
    g.add_conditional_edges(
        "validate",
        decide_after_validation,
        {
            "store_card": "store_card",
            "strengthen": "strengthen",
            "full_restart": "full_restart",
            "end": END,
        },
    )
    
    # Conditional edges after full restart
    g.add_conditional_edges(
        "full_restart",
        decide_after_restart,
        {
            "generate_question": "generate_question",
            "end": END,
        },
    )
    
    return g.compile()


# ------------- Entry Points -------------


def _run_question_phase(
    *,
    exam_id: str,
    topic_id: str,
    topic_label: str,
    allowed_chunk_ids: List[str],
    context_pack: str,
    difficulty: int,
    user_id: str,
    store: VectorStore,
) -> Optional[CardGenState]:
    """
    Phase 1: Generate and dedupe a unique question.
    
    Returns the state with question + embedding stored, or None if failed.
    """
    initial_state: CardGenState = {
        "exam_id": exam_id,
        "user_id": user_id,
        "store_basepath": str(store.base),
        "topic_id": topic_id,
        "topic_label": topic_label,
        "difficulty": difficulty,
        "allowed_chunk_ids": allowed_chunk_ids,
        "context_pack": context_pack,
        "question": None,
        "question_embedding": None,
        "question_vector_idx": None,
        "is_unique": False,
        "answer": None,
        "proofs": None,
        "validation_score": None,
        "validation_critique": None,
        "question_attempts": 0,
        "answer_attempts": 0,
        "full_restart_count": 0,
        "max_question_attempts": DEFAULT_CONFIG["max_question_attempts"],
        "max_answer_attempts": DEFAULT_CONFIG["max_answer_attempts"],
        "max_full_restarts": DEFAULT_CONFIG["max_full_restarts"],
        "k": DEFAULT_CONFIG["initial_k"],
        "min_score": DEFAULT_CONFIG["initial_min_score"],
        "stop_after_embedding": True,
        "card": None,
    }
    
    graph = build_card_graph()
    final_state = graph.invoke(initial_state)
    
    # Check if we got a valid question + embedding
    if final_state.get("question") and final_state.get("question_vector_idx") is not None:
        return final_state
    return None


def _run_answer_phase(state: CardGenState) -> Optional[GeneratedCard]:
    """
    Phase 2: Generate answer, validate, and store card.
    
    Includes retry logic for validation failures.
    Returns GeneratedCard on success, None on failure.
    """
    # Continue from where Phase 1 left off
    state = {**state, "stop_after_embedding": False}
    
    max_answer_attempts = state["max_answer_attempts"]
    max_full_restarts = state["max_full_restarts"]
    
    for restart in range(max_full_restarts):
        for attempt in range(max_answer_attempts):
            # Generate answer
            state = node_generate_answer(state)
            
            # Validate
            state = node_validate(state)
            
            # Check validation
            if state["validation_score"] >= DEFAULT_CONFIG["validation_threshold"]:
                # Success! Store the card
                state = node_store_card(state)
                return state.get("card")
            
            # Failed validation - strengthen and retry
            if attempt < max_answer_attempts - 1:
                state = node_strengthen(state)
                logger.debug(
                    "Answer validation failed (%.2f), strengthening retrieval (attempt %d)",
                    state["validation_score"], attempt + 1
                )
        
        # All answer attempts exhausted for this question
        # For starter pack, we don't do full restart (question is already unique)
        # Just give up on this card
        logger.warning(
            "Answer validation failed after %d attempts for question: %s",
            max_answer_attempts, state["question"][:50]
        )
        break
    
    return None


def generate_single_card(
    *,
    exam_id: str,
    topic_id: str,
    topic_label: str,
    allowed_chunk_ids: List[str],
    context_pack: str,
    difficulty: int = 1,
    user_id: str = "system",
    store: Optional[VectorStore] = None,
    stop_after_embedding: bool = False,
) -> Optional[GeneratedCard]:
    """
    Generate a single card using the LangGraph flow.
    
    Args:
        stop_after_embedding: If True, stop after storing embedding (Phase 1 only).
                              Used for parallel starter pack generation.
    
    Retries until success or max restarts reached.
    Returns None only if all retries exhausted (rare edge case).
    """
    store = store or VectorStore()
    
    initial_state: CardGenState = {
        "exam_id": exam_id,
        "user_id": user_id,
        "store_basepath": str(store.base),
        "topic_id": topic_id,
        "topic_label": topic_label,
        "difficulty": difficulty,
        "allowed_chunk_ids": allowed_chunk_ids,
        "context_pack": context_pack,
        "question": None,
        "question_embedding": None,
        "question_vector_idx": None,
        "is_unique": False,
        "answer": None,
        "proofs": None,
        "validation_score": None,
        "validation_critique": None,
        "question_attempts": 0,
        "answer_attempts": 0,
        "full_restart_count": 0,
        "max_question_attempts": DEFAULT_CONFIG["max_question_attempts"],
        "max_answer_attempts": DEFAULT_CONFIG["max_answer_attempts"],
        "max_full_restarts": DEFAULT_CONFIG["max_full_restarts"],
        "k": DEFAULT_CONFIG["initial_k"],
        "min_score": DEFAULT_CONFIG["initial_min_score"],
        "stop_after_embedding": stop_after_embedding,
        "card": None,
    }
    
    graph = build_card_graph()
    final_state = graph.invoke(initial_state)
    
    return final_state.get("card")


def generate_starter_cards_v2(
    *,
    exam_id: str,
    user_id: str,
    n: int = 5,
    difficulty: int = 1,
    store: Optional[VectorStore] = None,
    max_workers: int = 5,
) -> List[GeneratedCard]:
    """
    Generate N starter cards across distinct topics.
    
    Phase 1 (Sequential): Generate + dedupe unique questions for N topics
    Phase 2 (Parallel): Answer + validate all questions simultaneously
    
    Returns list of generated cards (may be < N if topics insufficient).
    """
    store = store or VectorStore()
    
    # Pick top N topics
    picked = pick_starter_topics(exam_id=exam_id, store=store, n=n)
    if not picked:
        logger.warning("No topics found for exam %s", exam_id)
        return []
    
    # Prepare context packs for each topic
    topic_contexts: Dict[str, Dict[str, Any]] = {}
    for topic_id, topic_label in picked:
        allowed_chunk_ids = store.db.list_chunk_ids_for_topic(topic_id=topic_id)
        if not allowed_chunk_ids:
            continue
        centroid = store.db.get_topic_vector(topic_id=topic_id)
        context_pack = build_diverse_chunk_pack(
            store=store,
            chunk_ids=allowed_chunk_ids,
            centroid=centroid,
        )
        if not context_pack:
            continue
        topic_contexts[topic_id] = {
            "topic_label": topic_label,
            "allowed_chunk_ids": allowed_chunk_ids,
            "context_pack": context_pack,
        }
    
    if not topic_contexts:
        logger.warning("No valid topic contexts for exam %s", exam_id)
        return []
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1: Sequential question generation + deduplication
    # Must be sequential so each uniqueness check sees all previous questions
    # ═══════════════════════════════════════════════════════════════════════
    
    question_states: List[CardGenState] = []
    
    for topic_id, ctx in topic_contexts.items():
        if len(question_states) >= n:
            break
        
        state = _run_question_phase(
            exam_id=exam_id,
            topic_id=topic_id,
            topic_label=ctx["topic_label"],
            allowed_chunk_ids=ctx["allowed_chunk_ids"],
            context_pack=ctx["context_pack"],
            difficulty=difficulty,
            user_id=user_id,
            store=store,
        )
        
        if state:
            question_states.append(state)
            logger.info(
                "Phase 1: Generated unique question for topic '%s': %s",
                ctx["topic_label"], state["question"][:50]
            )
    
    if not question_states:
        logger.warning("Phase 1: No unique questions generated for exam %s", exam_id)
        return []
    
    logger.info(
        "Phase 1 complete: %d unique questions ready for answering",
        len(question_states)
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: Parallel answer generation + validation
    # Each answer is independent, so we can parallelize
    # ═══════════════════════════════════════════════════════════════════════
    
    cards: List[GeneratedCard] = []
    
    with ThreadPoolExecutor(max_workers=min(max_workers, len(question_states))) as executor:
        future_to_state = {
            executor.submit(_run_answer_phase, state): state
            for state in question_states
        }
        
        for future in as_completed(future_to_state):
            state = future_to_state[future]
            try:
                card = future.result()
                if card:
                    cards.append(card)
                    logger.info(
                        "Phase 2: Generated card %s for topic '%s'",
                        card.card_id, card.topic_label
                    )
                else:
                    logger.warning(
                        "Phase 2: Failed to generate answer for question: %s",
                        state["question"][:50]
                    )
            except Exception as e:
                logger.error(
                    "Phase 2: Exception generating answer for '%s': %s",
                    state["topic_label"], str(e)
                )
    
    logger.info(
        "Phase 2 complete: %d cards generated out of %d questions",
        len(cards), len(question_states)
    )
    
    return cards


# For backward compatibility - alias to old name pattern
def node_propose_questions(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backward compatibility stub.
    The old doc-level question proposer is deprecated.
    Use generate_starter_cards_v2() instead.
    """
    logger.warning(
        "node_propose_questions is deprecated. Use generate_starter_cards_v2() instead."
    )
    return state


