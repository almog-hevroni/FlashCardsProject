from pathlib import Path
from typing import List, TypedDict

class Page(TypedDict):
    page: int
    text: str

def load_txt(path: str | Path) -> List[Page]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return [{"page": 1, "text": text}]
