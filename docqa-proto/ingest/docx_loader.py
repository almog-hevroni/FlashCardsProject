from typing import List, TypedDict
from docx import Document

class Page(TypedDict):
    page: int
    text: str

def load_docx(path: str) -> List[Page]:
    # DOCX has no pages; treat as single "page 1"
    doc = Document(path)
    text = "\n".join(p.text for p in doc.paragraphs)
    return [{"page": 1, "text": text}]
