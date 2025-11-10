from typing import List, TypedDict
try:
    from docx import Document  # type: ignore
except Exception as exc:
    raise ImportError(
        "python-docx is required but not installed. Install it with 'pip install python-docx'."
    ) from exc

class Page(TypedDict):
    page: int
    text: str

def load_docx(path: str) -> List[Page]:
    # DOCX has no pages; treat as single "page 1"
    doc = Document(path)
    text = "\n".join(p.text for p in doc.paragraphs)
    return [{"page": 1, "text": text}]
