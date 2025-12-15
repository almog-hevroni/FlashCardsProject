from typing import List, Tuple, Optional
import logging
import numpy as np
from app.utils import getenv

_client = None
logger = logging.getLogger(__name__)

def client():
    # Lazy import to avoid hard dependency during unrelated code paths.
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        raise ImportError(
            "OpenAI SDK is required but not installed. Install it with 'pip install openai'."
        ) from exc
    api_key = getenv("OPENAI_API_KEY")
    if api_key.lower().startswith("sk-your"):
        raise RuntimeError(
            "OPENAI_API_KEY is still set to the placeholder value. "
            "Update /Users/lioka/Desktop/flashcards/docqa-proto/.env with your real key."
        )
    global _client
    if _client is None:
        logger.debug("Initializing OpenAI client (key prefix: %s****)", api_key[:8])
        _client = OpenAI(api_key=api_key)
    return _client

EMBED_MODEL = getenv("EMBED_MODEL", "text-embedding-3-large")
CHAT_MODEL = getenv("CHAT_MODEL", "gpt-4o")
CHAT_MODEL_FAST = getenv("CHAT_MODEL_FAST", "gpt-4o-mini")

def embed_texts(texts: List[str]) -> np.ndarray:
    # OpenAI returns list of vectors
    # Create embeddings for the texts
    resp = client().embeddings.create(model=EMBED_MODEL, input=texts)
    # Extract the embeddings from the response to an array of arrays
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype="float32")

def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])

def generate_alternate_queries(
    question: str,
    num_variations: int = 3,
    *,
    model: Optional[str] = None,
) -> List[str]:
    """
    Use the chat model to produce semantically diverse reformulations of the query
    to improve recall (multi-query retrieval).
    """
    prompt = (
        "Rewrite the user's question into diverse, concise search queries that capture different phrasings.\n"
        f"Original question: {question}\n"
        f"Return exactly {num_variations} queries, one per line, no numbering."
    )
    resp = client().chat.completions.create(
        model=model or CHAT_MODEL_FAST,
        messages=[
            {"role": "system", "content": "You are an expert search assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=120,
    )
    text = resp.choices[0].message.content or ""
    queries = [q.strip() for q in text.splitlines() if q.strip()]
    if len(queries) < num_variations:
        # pad with original question if the model returned fewer
        queries += [question] * (num_variations - len(queries))
    return queries[:num_variations]

