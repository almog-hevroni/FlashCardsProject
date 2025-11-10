from pathlib import Path
from typing import List, TypedDict
from pypdf import PdfReader

class Page(TypedDict):
    page: int
    text: str

def load_pdf(path: str | Path) -> List[Page]:
    reader = PdfReader(str(path))
    pages: List[Page] = []
    for i, p in enumerate(reader.pages, start=1):
        pages.append({"page": i, "text": p.extract_text() or ""})
    return pages
