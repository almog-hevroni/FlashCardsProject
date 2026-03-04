from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.api.schemas import ProofSpan
from app.data.vector_store import VectorStore
from app.services.qa import generate_answer


@dataclass
class TeacherModelResult:
    answer: str
    proofs: List[ProofSpan]


@dataclass
class TeacherModelService:
    """Grounded teacher-side answer generation."""

    def generate_answer(
        self,
        *,
        question: str,
        k: int,
        min_score: float,
        store: VectorStore,
        allowed_chunk_ids: list[str],
    ) -> TeacherModelResult:
        result = generate_answer(
            question=question,
            k=k,
            min_score=min_score,
            store=store,
            allowed_chunk_ids=allowed_chunk_ids,
        )
        return TeacherModelResult(answer=result.answer, proofs=result.proofs)
