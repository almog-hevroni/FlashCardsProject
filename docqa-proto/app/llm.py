from typing import List
import numpy as np
from openai import OpenAI
from app.utils import getenv

_client = None
def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=getenv("OPENAI_API_KEY"))
    return _client

EMBED_MODEL = getenv("EMBED_MODEL", "text-embedding-3-small")

def embed_texts(texts: List[str]) -> np.ndarray:
    # OpenAI returns list of vectors
    resp = client().embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype="float32")

def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])
