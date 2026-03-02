from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.data.db_repository import DBRepository
from app.services.diagnostic_review_reducer import DiagnosticReviewReducer
from app.services.review_state_reducer import ReviewStateReducer


@dataclass
class ReviewServiceResult:
    review_id: str
    card_id: str
    rating: str
    exam_state: str
    topic_proficiency: float
    due_at: Optional[str] = None
    interval_days: Optional[float] = None
    ease: Optional[float] = None
    diagnostic_answered: Optional[int] = None
    diagnostic_total: Optional[int] = None
    idempotent_replay: bool = False


class ReviewService:
    """
    Thin orchestration adapter that routes review handling by exam state.
    """

    def __init__(
        self,
        *,
        repo: Optional[DBRepository] = None,
        diagnostic_reducer: Optional[DiagnosticReviewReducer] = None,
        active_reducer: Optional[ReviewStateReducer] = None,
    ) -> None:
        self.repo = repo or DBRepository(Path("store/meta.sqlite"))
        self.diagnostic_reducer = diagnostic_reducer or DiagnosticReviewReducer(repo=self.repo)
        self.active_reducer = active_reducer or ReviewStateReducer(repo=self.repo)

    def apply_review(
        self,
        *,
        user_id: str,
        exam_id: str,
        card_id: str,
        rating: str,
        idempotency_key: str,
    ) -> ReviewServiceResult:
        exam = self.repo.get_exam(exam_id)
        if exam is None:
            raise ValueError(f"Exam not found: {exam_id}")
        if exam.user_id != user_id:
            raise ValueError("Exam does not belong to user")

        if exam.state == "diagnostic":
            d = self.diagnostic_reducer.apply_review(
                user_id=user_id,
                exam_id=exam_id,
                card_id=card_id,
                rating=rating,
                idempotency_key=idempotency_key,
            )
            return ReviewServiceResult(
                review_id=d.review_id,
                card_id=d.card_id,
                rating=d.rating,
                exam_state=d.exam_state,
                topic_proficiency=d.topic_proficiency,
                diagnostic_answered=d.diagnostic_answered,
                diagnostic_total=d.diagnostic_total,
                idempotent_replay=d.idempotent_replay,
            )

        a = self.active_reducer.apply_review(
            user_id=user_id,
            exam_id=exam_id,
            card_id=card_id,
            rating=rating,
            idempotency_key=idempotency_key,
        )
        return ReviewServiceResult(
            review_id=a.review_id,
            card_id=a.card_id,
            rating=a.rating,
            exam_state=exam.state,
            topic_proficiency=a.topic_proficiency,
            due_at=a.due_at,
            interval_days=a.interval_days,
            ease=a.ease,
            idempotent_replay=a.idempotent_replay,
        )
