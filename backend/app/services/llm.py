from typing import List, Tuple, Optional
import logging
import numpy as np
from app.utils.common import getenv

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
