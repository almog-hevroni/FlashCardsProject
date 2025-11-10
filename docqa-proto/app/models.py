from pydantic import BaseModel

class IngestResult(BaseModel):
    doc_id: str
    num_chunks: int

class ProofSpan(BaseModel):
    doc_id: str
    page: int
    start: int
    end: int
    text: str
    score: float
