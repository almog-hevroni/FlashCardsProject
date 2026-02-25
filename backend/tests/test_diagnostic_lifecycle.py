import os
import tempfile
import unittest
from pathlib import Path

# Configure dedicated test database before importing app modules.
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="phase3_diagnostic_lifecycle_"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_ROOT / 'meta.sqlite').as_posix()}"
os.environ["VECTOR_BACKEND"] = "numpy"

from fastapi.testclient import TestClient

from app.api.endpoints import app
from app.data.db_engine import drop_all_tables, init_db
from app.data.vector_store import VectorStore
from app.services.exams import create_exam
from app.services.graph import GeneratedCard


def _write_text_doc(name: str, body: str) -> str:
    path = _TEST_ROOT / name
    path.write_text(body, encoding="utf-8")
    return str(path)


class Phase3DiagnosticLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        drop_all_tables()
        init_db()
        self.client = TestClient(app)

    def test_bootstrap_and_atomic_transition(self) -> None:
        from app.services import diagnostic_lifecycle as dl_mod

        original_build_topics = dl_mod.build_topics_for_exam
        original_generate = dl_mod.generate_starter_cards_v2

        def fake_build_topics_for_exam(*, exam_id, store, overwrite=True):
            store.db.upsert_topic(topic_id="tA", exam_id=exam_id, label="Topic A", info={"n_chunks": 1})
            store.db.upsert_topic(topic_id="tB", exam_id=exam_id, label="Topic B", info={"n_chunks": 1})
            return [
                {"topic_id": "tA", "label": "Topic A"},
                {"topic_id": "tB", "label": "Topic B"},
            ]

        def fake_generate_starter_cards_v2(*, exam_id, user_id, n, difficulty, card_type="learning", store=None, max_workers=5):
            assert store is not None
            cards = []
            for i, tid in enumerate(["tA", "tB"][:n], start=1):
                cid = f"diag_card_{i}"
                store.db.upsert_card(
                    card_id=cid,
                    exam_id=exam_id,
                    topic_id=tid,
                    question=f"Q{i}",
                    answer=f"A{i}",
                    difficulty=1,
                    card_type="diagnostic",
                    status="active",
                    info={"source": "test"},
                )
                store.db.replace_card_topics(
                    card_id=cid,
                    topics=[{"topic_id": tid, "role": "primary", "weight": 1.0}],
                )
                cards.append(
                    GeneratedCard(
                        card_id=cid,
                        exam_id=exam_id,
                        topic_id=tid,
                        topic_label=f"Topic {tid}",
                        question=f"Q{i}",
                        answer=f"A{i}",
                        difficulty=1,
                        proofs=[],
                    )
                )
            return cards

        dl_mod.build_topics_for_exam = fake_build_topics_for_exam
        dl_mod.generate_starter_cards_v2 = fake_generate_starter_cards_v2
        try:
            d1 = _write_text_doc("p3_bootstrap_1.txt", "alpha beta gamma")
            resp = self.client.post(
                "/exams/from-upload",
                data={"user_id": "u1", "title": "Exam 1", "mode": "mastery"},
                files=[("files", ("p3_bootstrap_1.txt", Path(d1).read_text(encoding="utf-8"), "text/plain"))],
            )
            self.assertEqual(resp.status_code, 200)
            payload = resp.json()
            exam_id = payload["exam_id"]
            self.assertEqual(payload["state"], "diagnostic")
            self.assertEqual(payload["diagnostic_total"], 2)
            self.assertEqual(payload["diagnostic_answered"], 0)

            exam_resp = self.client.get(f"/exams/{exam_id}", params={"user_id": "u1"})
            self.assertEqual(exam_resp.status_code, 200)
            self.assertEqual(exam_resp.json()["state"], "diagnostic")

            r1 = self.client.post(
                f"/exams/{exam_id}/cards/diag_card_1/review",
                data={"user_id": "u1", "rating": "almost_knew"},
                headers={"Idempotency-Key": "diag-r1"},
            )
            self.assertEqual(r1.status_code, 200)
            self.assertEqual(r1.json()["diagnostic_answered"], 1)
            self.assertEqual(r1.json()["exam_state"], "diagnostic")

            r1_replay = self.client.post(
                f"/exams/{exam_id}/cards/diag_card_1/review",
                data={"user_id": "u1", "rating": "almost_knew"},
                headers={"Idempotency-Key": "diag-r1"},
            )
            self.assertEqual(r1_replay.status_code, 200)
            self.assertTrue(r1_replay.json()["idempotent_replay"])
            self.assertEqual(r1_replay.json()["diagnostic_answered"], 1)

            r2 = self.client.post(
                f"/exams/{exam_id}/cards/diag_card_2/review",
                data={"user_id": "u1", "rating": "i_knew_it"},
                headers={"Idempotency-Key": "diag-r2"},
            )
            self.assertEqual(r2.status_code, 200)
            self.assertEqual(r2.json()["diagnostic_answered"], 2)
            self.assertEqual(r2.json()["exam_state"], "active_learning")

            exam_resp2 = self.client.get(f"/exams/{exam_id}", params={"user_id": "u1"})
            self.assertEqual(exam_resp2.status_code, 200)
            self.assertEqual(exam_resp2.json()["state"], "active_learning")
            self.assertEqual(exam_resp2.json()["diagnostic_answered"], 2)
        finally:
            dl_mod.build_topics_for_exam = original_build_topics
            dl_mod.generate_starter_cards_v2 = original_generate

    def test_failed_bootstrap_cleans_up_exam_data(self) -> None:
        from app.services import diagnostic_lifecycle as dl_mod

        original_build_topics = dl_mod.build_topics_for_exam
        original_generate = dl_mod.generate_starter_cards_v2

        def fake_build_topics_for_exam(*, exam_id, store, overwrite=True):
            store.db.upsert_topic(topic_id="tA", exam_id=exam_id, label="Topic A", info={"n_chunks": 1})
            store.db.upsert_topic(topic_id="tB", exam_id=exam_id, label="Topic B", info={"n_chunks": 1})
            return [
                {"topic_id": "tA", "label": "Topic A"},
                {"topic_id": "tB", "label": "Topic B"},
            ]

        def fake_generate_partial(*, exam_id, user_id, n, difficulty, card_type="learning", store=None, max_workers=5):
            assert store is not None
            # Generate fewer cards than topics to force failure.
            cid = "diag_card_partial"
            store.db.upsert_card(
                card_id=cid,
                exam_id=exam_id,
                topic_id="tA",
                question="Q?",
                answer="A",
                difficulty=1,
                card_type="diagnostic",
                status="active",
                info={"source": "test"},
            )
            store.db.replace_card_topics(
                card_id=cid,
                topics=[{"topic_id": "tA", "role": "primary", "weight": 1.0}],
            )
            return [
                GeneratedCard(
                    card_id=cid,
                    exam_id=exam_id,
                    topic_id="tA",
                    topic_label="Topic A",
                    question="Q?",
                    answer="A",
                    difficulty=1,
                    proofs=[],
                )
            ]

        dl_mod.build_topics_for_exam = fake_build_topics_for_exam
        dl_mod.generate_starter_cards_v2 = fake_generate_partial
        try:
            d1 = _write_text_doc("p3_bootstrap_fail.txt", "alpha beta gamma")
            resp = self.client.post(
                "/exams/from-upload",
                data={"user_id": "u2", "title": "Exam fail", "mode": "mastery"},
                files=[("files", ("p3_bootstrap_fail.txt", Path(d1).read_text(encoding="utf-8"), "text/plain"))],
            )
            self.assertEqual(resp.status_code, 422)
            payload = resp.json()
            self.assertEqual(payload.get("error"), "diagnostic_bootstrap_failed")

            store = VectorStore(basepath=str(_TEST_ROOT / "store"))
            exams = store.db.list_exams(user_id="u2")
            self.assertEqual(len(exams), 0)
        finally:
            dl_mod.build_topics_for_exam = original_build_topics
            dl_mod.generate_starter_cards_v2 = original_generate


if __name__ == "__main__":
    unittest.main()

