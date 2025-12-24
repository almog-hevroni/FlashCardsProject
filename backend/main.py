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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ingest", nargs="+", help="one or more document paths to ingest")
    p.add_argument("--create_exam", action="store_true", help="create a new exam workspace")
    p.add_argument("--exam_title", default="New Exam", help="title for --create_exam")
    p.add_argument("--exam_mode", default="mastery", help="mastery|exam (for --create_exam)")
    p.add_argument("--user_id", default="local_user", help="local-first user id")
    p.add_argument("--exam_id", help="existing exam id (to attach documents or inspect)")
    p.add_argument("--ask", help="question: retrieve proofs only (no LLM answer)")
    p.add_argument("--answer", help="question: retrieve + generate answer with citations")
    p.add_argument("--qa_n", type=int, default=5, help="number of auto-generated QA pairs after ingest")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--min_score", type=float, default=0.4)
    args = p.parse_args()

    store = VectorStore()

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
        proofs = retrieve_with_proofs(args.ask, k=args.k, store=store)
        print(json.dumps([p.model_dump() for p in proofs], ensure_ascii=False, indent=2))

    if args.answer:
        ans = generate_answer(args.answer, k=args.k, min_score=args.min_score)
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
