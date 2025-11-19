import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Always load the project-level .env so CLI invocations pick up secrets.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=True)
load_dotenv(override=False)

def getenv(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing env var: {name}")
    return v

_enc_cache: Dict[str, Any] = {}

def token_len(text: str, model: str = "gpt-4o-mini") -> int:
    # Import lazily to avoid hard dependency unless needed.
    if model not in _enc_cache:
        try:
            import tiktoken  # type: ignore
        except Exception as exc:
            raise ImportError(
                "tiktoken is required for token_len(). Install it with 'pip install tiktoken'."
            ) from exc
        _enc_cache[model] = tiktoken.get_encoding("cl100k_base")
    return len(_enc_cache[model].encode(text))

def sliding_window(chunks: List[str], overlap: int = 0) -> List[str]:
    if overlap <= 0: return chunks
    out = []
    for i, ch in enumerate(chunks):
        prev = chunks[i-1] if i > 0 else ""
        out.append((prev[-overlap:] + " " + ch).strip())
    return out
