import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

# Configure dedicated test database before importing app modules.
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="phase4_api_planner_"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_ROOT / 'meta.sqlite').as_posix()}"
os.environ["VECTOR_BACKEND"] = "numpy"

from app.api.endpoints import app
from app.data.db_engine import drop_all_tables, init_db
from app.data.db_repository import DBRepository


class ApiPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        drop_all_tables()
        init_db()
        self.client = TestClient(app)
        self.repo = DBRepository(_TEST_ROOT / "meta.sqlite")
        self.user_id = "phase4-user"
        self.repo.ensure_user(self.user_id)
        self.exam_id = self.repo.create_exam(user_id=self.user_id, title="Phase 4 Exam")
        self.repo.update_exam_lifecycle(exam_id=self.exam_id, state="active_learning")
        self.doc_id = "doc-source-1"
        self.source_path = _TEST_ROOT / "source.txt"
        self.source_path.write_text("source body", encoding="utf-8")
        self.repo.add_document(
            doc_id=self.doc_id,
            path=self.source_path.as_posix(),
            title="source.txt",
            info={},
        )
        self.repo.attach_documents_to_exam(exam_id=self.exam_id, doc_ids=[self.doc_id])

        self.repo.upsert_topic(topic_id="t1", exam_id=self.exam_id, label="Topic 1")
        self.repo.upsert_topic(topic_id="t2", exam_id=self.exam_id, label="Topic 2")
        self.repo.upsert_card(
            card_id="c1",
            exam_id=self.exam_id,
            topic_id="t1",
            question="Q1?",
            answer="A1",
            difficulty=1,
            card_type="learning",
            status="active",
            info={},
        )
        self.repo.replace_card_topics(
            card_id="c1",
            topics=[{"topic_id": "t1", "role": "primary", "weight": 1.0}],
        )
        self.repo.upsert_card(
            card_id="c2",
            exam_id=self.exam_id,
            topic_id="t2",
            question="Q2?",
            answer="A2",
            difficulty=2,
            card_type="learning",
            status="active",
            info={},
        )
        self.repo.replace_card_topics(
            card_id="c2",
            topics=[{"topic_id": "t2", "role": "primary", "weight": 1.0}],
        )
        now = datetime.now(timezone.utc)
        self.repo.upsert_card_scheduling(
            card_id="c1",
            due_at=now - timedelta(minutes=10),
            state="review",
            interval_days=1.0,
            ease=2.5,
            reps=1,
            lapses=0,
            last_reviewed_at=now - timedelta(days=1),
        )
        self.repo.upsert_card_scheduling(
            card_id="c2",
            due_at=now + timedelta(days=2),
            state="new",
            interval_days=1.0,
            ease=2.5,
            reps=0,
            lapses=0,
            last_reviewed_at=now - timedelta(days=1),
        )

    def test_next_previous_and_presented_history(self) -> None:
        n1 = self.client.get(
            f"/exams/{self.exam_id}/session/next-card",
            params={"user_id": self.user_id},
        )
        self.assertEqual(n1.status_code, 200)
        p1 = n1.json()
        self.assertFalse(p1["no_cards_available"])
        self.assertEqual(p1["reason"], "overdue")
        self.assertEqual(p1["card"]["card_id"], "c1")

        n2 = self.client.get(
            f"/exams/{self.exam_id}/session/next-card",
            params={"user_id": self.user_id},
        )
        self.assertEqual(n2.status_code, 200)

        prev = self.client.get(
            f"/exams/{self.exam_id}/session/previous-card",
            params={"user_id": self.user_id},
        )
        self.assertEqual(prev.status_code, 200)
        self.assertFalse(prev.json()["no_cards_available"])
        self.assertEqual(prev.json()["reason"], "previous")

        history = self.client.get(
            f"/exams/{self.exam_id}/cards/presented-history",
            params={"user_id": self.user_id},
        )
        self.assertEqual(history.status_code, 200)
        self.assertGreaterEqual(history.json()["total"], 2)

    def test_active_learning_review_uses_unified_orchestration(self) -> None:
        res = self.client.post(
            f"/exams/{self.exam_id}/cards/c1/review",
            data={"user_id": self.user_id, "rating": "learned_now"},
            headers={"Idempotency-Key": "phase4-active-review"},
        )
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["card_id"], "c1")
        self.assertEqual(payload["exam_state"], "active_learning")
        self.assertIn("due_at", payload)
        self.assertIn("interval_days", payload)
        self.assertIn("ease", payload)

    def test_progress_contract_endpoint(self) -> None:
        # Seed one proficiency row through review.
        review = self.client.post(
            f"/exams/{self.exam_id}/cards/c1/review",
            data={"user_id": self.user_id, "rating": "almost_knew"},
            headers={"Idempotency-Key": "phase4-progress-review"},
        )
        self.assertEqual(review.status_code, 200)
        progress = self.client.get(
            f"/exams/{self.exam_id}/progress",
            params={"user_id": self.user_id},
        )
        self.assertEqual(progress.status_code, 200)
        payload = progress.json()
        self.assertEqual(payload["exam_id"], self.exam_id)
        self.assertEqual(payload["user_id"], self.user_id)
        self.assertIn("topics", payload)
        self.assertGreaterEqual(len(payload["topics"]), 1)

        # Reviewed topic should appear with updated proficiency metrics.
        t1_row = next((t for t in payload["topics"] if t["topic_id"] == "t1"), None)
        self.assertIsNotNone(t1_row)
        assert t1_row is not None
        self.assertEqual(t1_row["topic_label"], "Topic 1")
        self.assertGreaterEqual(float(t1_row["proficiency"]), 0.0)
        self.assertLessEqual(float(t1_row["proficiency"]), 1.0)
        self.assertGreaterEqual(int(t1_row["n_reviews"]), 1)

        # Overall proficiency should be computed when at least one topic row exists.
        self.assertIn("overall_proficiency", payload)
        self.assertIsNotNone(payload["overall_proficiency"])
        self.assertGreaterEqual(float(payload["overall_proficiency"]), 0.0)
        self.assertLessEqual(float(payload["overall_proficiency"]), 1.0)

    def test_document_source_endpoint_returns_exam_document(self) -> None:
        res = self.client.get(
            f"/documents/{self.doc_id}/source",
            params={"exam_id": self.exam_id, "user_id": self.user_id},
        )
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/plain", (res.headers.get("content-type") or ""))
        self.assertIn("source body", res.text)

    def test_diagnostic_exam_serves_unanswered_diagnostic_cards(self) -> None:
        self.repo.update_exam_lifecycle(exam_id=self.exam_id, state="diagnostic")
        self.repo.upsert_card(
            card_id="d1",
            exam_id=self.exam_id,
            topic_id="t1",
            question="Diagnostic Q1?",
            answer="Diagnostic A1",
            difficulty=1,
            card_type="diagnostic",
            status="active",
            info={},
        )
        self.repo.replace_card_topics(
            card_id="d1",
            topics=[{"topic_id": "t1", "role": "primary", "weight": 1.0}],
        )
        self.repo.upsert_card(
            card_id="d2",
            exam_id=self.exam_id,
            topic_id="t2",
            question="Diagnostic Q2?",
            answer="Diagnostic A2",
            difficulty=1,
            card_type="diagnostic",
            status="active",
            info={},
        )
        self.repo.replace_card_topics(
            card_id="d2",
            topics=[{"topic_id": "t2", "role": "primary", "weight": 1.0}],
        )

        first = self.client.get(
            f"/exams/{self.exam_id}/session/next-card",
            params={"user_id": self.user_id},
        )
        self.assertEqual(first.status_code, 200)
        first_payload = first.json()
        self.assertFalse(first_payload["no_cards_available"])
        self.assertEqual(first_payload["reason"], "diagnostic")
        self.assertIn(first_payload["card"]["card_id"], {"d1", "d2"})

        first_card_id = first_payload["card"]["card_id"]
        first_review = self.client.post(
            f"/exams/{self.exam_id}/cards/{first_card_id}/review",
            data={"user_id": self.user_id, "rating": "almost_knew"},
            headers={"Idempotency-Key": "phase4-diagnostic-first"},
        )
        self.assertEqual(first_review.status_code, 200)
        self.assertEqual(first_review.json()["exam_state"], "diagnostic")

        second = self.client.get(
            f"/exams/{self.exam_id}/session/next-card",
            params={"user_id": self.user_id},
        )
        self.assertEqual(second.status_code, 200)
        second_payload = second.json()
        self.assertFalse(second_payload["no_cards_available"])
        self.assertEqual(second_payload["reason"], "diagnostic")
        self.assertNotEqual(second_payload["card"]["card_id"], first_card_id)


if __name__ == "__main__":
    unittest.main()
