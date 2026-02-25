import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import inspect

# Configure dedicated test database before importing app modules.
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="phase1_schema_contract_"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_ROOT / 'meta.sqlite').as_posix()}"
os.environ["VECTOR_BACKEND"] = "numpy"

from app.data.db_engine import drop_all_tables, init_db, engine, SessionLocal
from app.data.models import (
    User,
    Exam,
    Topic,
    Card,
    CardTopic,
    CardScheduling,
    TopicProficiency,
    ExamSessionState,
    CardPresentationLog,
    StudentKnowledgeState,
)


class SchemaContractTests(unittest.TestCase):
    def setUp(self) -> None:
        drop_all_tables()
        init_db()

    def test_phase1_tables_exist(self) -> None:
        table_names = set(inspect(engine).get_table_names())
        required = {
            "card_topics",
            "exam_session_state",
            "card_presentation_log",
            "student_knowledge_state",
        }
        for table_name in required:
            self.assertIn(table_name, table_names)

    def test_phase1_columns_exist(self) -> None:
        schema_expectations = {
            "exams": {
                "state",
                "diagnostic_total",
                "diagnostic_answered",
                "diagnostic_started_at",
                "diagnostic_completed_at",
            },
            "cards": {"card_type", "retired_at", "supersedes_card_id"},
            "card_scheduling": {"state"},
            "topic_proficiency": {
                "current_difficulty",
                "streak_up",
                "streak_down",
                "seen_count",
                "correctish_count",
            },
        }
        inspector = inspect(engine)
        for table_name, expected_columns in schema_expectations.items():
            existing = {col["name"] for col in inspector.get_columns(table_name)}
            for col in expected_columns:
                self.assertIn(col, existing, f"Missing column '{col}' in table '{table_name}'")

    def test_phase1_defaults_roundtrip(self) -> None:
        now = datetime.now(timezone.utc)
        with SessionLocal() as db:
            db.add(User(user_id="phase1-user"))
            db.add(Exam(exam_id="phase1-exam", user_id="phase1-user", title="Phase 1"))
            db.add(Topic(topic_id="phase1-topic", exam_id="phase1-exam", label="Topic"))
            db.add(
                Card(
                    card_id="phase1-card",
                    exam_id="phase1-exam",
                    topic_id="phase1-topic",
                    question="Q?",
                    answer="A",
                )
            )
            db.flush()

            db.add(
                CardTopic(
                    card_id="phase1-card",
                    topic_id="phase1-topic",
                    role="primary",
                    weight=1.0,
                )
            )
            db.add(CardScheduling(card_id="phase1-card", due_at=now))
            db.add(
                TopicProficiency(
                    user_id="phase1-user",
                    exam_id="phase1-exam",
                    topic_id="phase1-topic",
                )
            )
            db.add(ExamSessionState(user_id="phase1-user", exam_id="phase1-exam"))
            db.add(
                CardPresentationLog(
                    user_id="phase1-user",
                    exam_id="phase1-exam",
                    sequence_no=1,
                    card_id="phase1-card",
                )
            )
            db.add(
                StudentKnowledgeState(
                    user_id="phase1-user",
                    exam_id="phase1-exam",
                    topic_id="phase1-topic",
                )
            )
            db.commit()

            exam = db.query(Exam).filter(Exam.exam_id == "phase1-exam").one()
            card = db.query(Card).filter(Card.card_id == "phase1-card").one()
            scheduling = db.query(CardScheduling).filter(CardScheduling.card_id == "phase1-card").one()
            proficiency = (
                db.query(TopicProficiency)
                .filter(
                    TopicProficiency.user_id == "phase1-user",
                    TopicProficiency.exam_id == "phase1-exam",
                    TopicProficiency.topic_id == "phase1-topic",
                )
                .one()
            )

            self.assertEqual(exam.state, "diagnostic")
            self.assertEqual(exam.diagnostic_total, 0)
            self.assertEqual(exam.diagnostic_answered, 0)
            self.assertEqual(card.card_type, "learning")
            self.assertEqual(scheduling.state, "new")
            self.assertEqual(proficiency.current_difficulty, 1)
            self.assertEqual(proficiency.streak_up, 0)
            self.assertEqual(proficiency.streak_down, 0)
            self.assertEqual(proficiency.seen_count, 0)
            self.assertEqual(proficiency.correctish_count, 0)

    def test_repository_exam_lifecycle_contract_roundtrip(self) -> None:
        from app.data.db_repository import DBRepository

        repo = DBRepository(_TEST_ROOT / "meta.sqlite")
        with SessionLocal() as db:
            db.add(User(user_id="phase1-u2"))
            db.commit()
        exam_id = repo.create_exam(user_id="phase1-u2", title="Lifecycle contract")
        now = datetime.now(timezone.utc)
        repo.update_exam_lifecycle(
            exam_id=exam_id,
            state="active_learning",
            diagnostic_total=3,
            diagnostic_answered=2,
            diagnostic_started_at=now,
        )
        updated = repo.get_exam(exam_id)
        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertEqual(updated.state, "active_learning")
        self.assertEqual(updated.diagnostic_total, 3)
        self.assertEqual(updated.diagnostic_answered, 2)
        self.assertTrue(updated.diagnostic_started_at)


if __name__ == "__main__":
    unittest.main()
