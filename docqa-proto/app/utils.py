import os, tiktoken
from typing import List

def getenv(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing env var: {name}")
    return v

_enc_cache: dict[str, tiktoken.Encoding] = {}

def token_len(text: str, model: str = "gpt-4o-mini") -> int:
    if model not in _enc_cache:
        _enc_cache[model] = tiktoken.get_encoding("cl100k_base")
    return len(_enc_cache[model].encode(text))

def sliding_window(chunks: List[str], overlap: int = 0) -> List[str]:
    if overlap <= 0: return chunks
    out = []
    for i, ch in enumerate(chunks):
        prev = chunks[i-1] if i > 0 else ""
        out.append((prev[-overlap:] + " " + ch).strip())
    return out
