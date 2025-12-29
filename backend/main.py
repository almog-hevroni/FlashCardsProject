import argparse, json, logging
from dotenv import load_dotenv
load_dotenv()

# For logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)

from app.services.ingestion import ingest_documents
from app.services.retrieval import retrieve_with_proofs
from app.data.vector_store import VectorStore
from app.services.qa import generate_answer
from app.services.graph import run_generate_qa
from app.services.exams import create_exam, load_exam, attach_documents, log_event
from app.services.topics import build_topics_for_exam, list_topics_for_exam
from app.services.routing import answer_in_exam, route_question_to_topic

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ingest", nargs="+", help="one or more document paths to ingest")
    p.add_argument("--create_exam", action="store_true", help="create a new exam workspace")
    p.add_argument("--exam_title", default="New Exam", help="title for --create_exam")
    p.add_argument("--exam_mode", default="mastery", help="mastery|exam (for --create_exam)")
    p.add_argument("--user_id", default="local_user", help="local-first user id")
    p.add_argument("--exam_id", help="existing exam id (to attach documents or inspect)")
    p.add_argument("--build_topics", action="store_true", help="build topics for --exam_id")
    p.add_argument("--topic_merge_threshold", type=float, default=0.88, help="merge topics if centroid similarity >= threshold (default 0.88)")
    p.add_argument("--topic_id", help="optional: restrict --ask/--answer to this topic_id")
    p.add_argument("--auto_topic", action="store_true", help="auto-route question to topic within --exam_id (for --answer)")
    p.add_argument("--ask", help="question: retrieve proofs only (no LLM answer)")
    p.add_argument("--answer", help="question: retrieve + generate answer with citations")
    p.add_argument("--qa_n", type=int, default=5, help="number of auto-generated QA pairs after ingest")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--min_score", type=float, default=0.4)
    args = p.parse_args()

    store = VectorStore()

    if args.build_topics:
        if not args.exam_id:
            raise SystemExit("--build_topics requires --exam_id")
        topics = build_topics_for_exam(
            exam_id=args.exam_id,
            store=store,
            overwrite=True,
            merge_threshold=args.topic_merge_threshold,
        )
        print(f"Built {len(topics)} topic(s) for exam_id={args.exam_id}")
        for t in list_topics_for_exam(exam_id=args.exam_id, store=store):
            print(f"- {t.topic_id}: {t.label}")
        return

    if args.create_exam:
        eid = create_exam(
            store=store,
            user_id=args.user_id,
            title=args.exam_title,
            mode=args.exam_mode,
            info={"created_via": "cli"},
        )
        print(f"Created exam_id={eid}")
        return

    if args.ingest:
        results = ingest_documents(args.ingest, store)
        for res in results:
            print(f"Ingested doc_id={res.doc_id} chunks={res.num_chunks}")
        doc_ids = [res.doc_id for res in results]
        if args.exam_id and doc_ids:
            attach_documents(store=store, exam_id=args.exam_id, doc_ids=doc_ids)
            log_event(
                store=store,
                user_id=args.user_id,
                exam_id=args.exam_id,
                type="documents_attached",
                payload={"doc_ids": doc_ids},
            )
            ws = load_exam(store=store, exam_id=args.exam_id)
            print(f"Attached documents to exam_id={args.exam_id} (now {len(ws.doc_ids)} doc(s))")
        if doc_ids and args.qa_n > 0:
            qa_out = run_generate_qa(doc_ids=doc_ids, num_questions=args.qa_n, store_basepath=str(store.base))
            print("\n=== AUTO-GENERATED QA ===\n")
            print(qa_out["report"])

    if args.ask:
        allowed_chunk_ids = None
        if args.topic_id:
            allowed_chunk_ids = store.db.list_chunk_ids_for_topic(topic_id=args.topic_id)
        proofs = retrieve_with_proofs(
            args.ask,
            k=args.k,
            store=store,
            allowed_chunk_ids=allowed_chunk_ids,
        )
        print(json.dumps([p.model_dump() for p in proofs], ensure_ascii=False, indent=2))

    if args.answer:
        if args.auto_topic:
            if not args.exam_id:
                raise SystemExit("--auto_topic requires --exam_id")
            ans, routes = answer_in_exam(
                exam_id=args.exam_id,
                question=args.answer,
                store=store,
                k=args.k,
                min_score=args.min_score,
                top_n_topics=1,
            )
            if routes:
                r0 = routes[0]
                print(f"\n=== ROUTED TOPIC ===\n{r0.topic_id}: {r0.label} (score={r0.score:.3f})\n")
            print("\n=== ANSWER ===\n")
            print(ans.answer)
            print("\n=== PROOFS ===\n")
            for i, p in enumerate(ans.proofs, 1):
                print(f"S{i} | doc={p.doc_id} page={p.page} score={p.score:.2f}")
                text = " ".join(p.text.split())
                print(f"\"{text}\"")
                print()
            return

        allowed_chunk_ids = None
        if args.topic_id:
            allowed_chunk_ids = store.db.list_chunk_ids_for_topic(topic_id=args.topic_id)
        ans = generate_answer(
            args.answer,
            k=args.k,
            min_score=args.min_score,
            store=store,
            allowed_chunk_ids=allowed_chunk_ids,
        )
        print("\n=== ANSWER ===\n")
        print(ans.answer)
        print("\n=== PROOFS ===\n")
        for i, p in enumerate(ans.proofs, 1):
            print(f"S{i} | doc={p.doc_id} page={p.page} score={p.score:.2f}")
            text = " ".join(p.text.split())
            print(f"\"{text}\"")
            print()

if __name__ == "__main__":
    main()
