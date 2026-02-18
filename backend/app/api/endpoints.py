try:
    from fastapi import FastAPI, UploadFile, File, Form  # type: ignore
    from fastapi.responses import JSONResponse  # type: ignore
except Exception as exc:
    raise ImportError(
        "FastAPI is required but not installed. Install it with 'pip install fastapi uvicorn'."
    ) from exc
from pathlib import Path
from typing import Any, Dict, List, Optional
import shutil

from app.services.ingestion import ingest_documents
from app.data.vector_store import VectorStore
from app.services.qa import generate_answer
from app.services.exams import attach_documents, ImmutableExamError

app = FastAPI(title="DocQA Proto")


def _immutable_exam_error_payload(*, exam_id: Optional[str] = None, message: str) -> Dict[str, Any]:
    return {
        "error": {
            "code": "immutable_exam",
            "message": message,
            "exam_id": exam_id,
        }
    }


@app.post("/ingest")
async def ingest_endpoint(
    files: List[UploadFile] = File(...),
    user_id: Optional[str] = Form(default=None),
    exam_id: Optional[str] = Form(default=None),
):
    # Save uploaded files to temp paths inside ./uploads
    uploads = Path("uploads")
    uploads.mkdir(exist_ok=True)
    temp_paths: List[Path] = []
    filenames: List[str] = []
    for file in files:
        temp_path = uploads / file.filename
        with temp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        temp_paths.append(temp_path)
        filenames.append(file.filename)

    store = VectorStore()
    exam_scoped = bool(user_id and exam_id)

    if store.vector_backend == "pinecone" and not exam_scoped:
        return JSONResponse(
            {
                "error": "Pinecone backend requires exam-scoped ingestion. "
                "This /ingest proto endpoint is deprecated.",
            },
            status_code=501,
        )
    try:
        results = ingest_documents(
            [str(p) for p in temp_paths],
            store=store,
            user_id=user_id,
            exam_id=exam_id,
        )
        if exam_scoped and exam_id is not None:
            doc_ids = [res.doc_id for res in results]
            attach_documents(store=store, exam_id=exam_id, doc_ids=doc_ids)
    except ImmutableExamError as exc:
        return JSONResponse(
            _immutable_exam_error_payload(exam_id=exc.exam_id, message=str(exc)),
            status_code=409,
        )
    except ValueError as exc:
        if exam_scoped and exam_id is not None and str(exc) == f"Exam not found: {exam_id}":
            return JSONResponse({"error": str(exc)}, status_code=404)
        raise

    docs = [
        {"doc_id": res.doc_id, "num_chunks": res.num_chunks, "filename": name}
        for res, name in zip(results, filenames)
    ]
    return {"documents": docs}

@app.post("/ask")
async def ask_endpoint(question: str = Form(...), k: int = Form(8), min_score: float = Form(0.4)):
    store = VectorStore()
    if store.vector_backend == "pinecone":
        return JSONResponse(
            {
                "error": "Pinecone backend requires exam-scoped retrieval. "
                "This /ask proto endpoint is deprecated.",
            },
            status_code=501,
        )
    ans = generate_answer(question=question, k=k, min_score=min_score, store=store)
    # Format proofs for response (short text)
    proofs = [{
        "doc_id": p.doc_id, "page": p.page, "score": p.score,
        "start": p.start, "end": p.end, "text": p.text.strip()
    } for p in ans.proofs]
    return JSONResponse({"answer": ans.answer, "proofs": proofs})
