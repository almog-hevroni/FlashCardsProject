import argparse, json, logging, time
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
from app.services.exams import create_exam, load_exam, attach_documents, log_event
from app.services.topics import build_topics_for_exam, list_topics_for_exam
from app.services.routing import answer_in_exam, route_question_to_topic
from app.services.cards import generate_starter_cards

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
    p.add_argument("--gen_starter_cards", action="store_true", help="generate starter flashcards for --exam_id (requires topics)")
    p.add_argument("--ask", help="question: retrieve proofs only (no LLM answer)")
    p.add_argument("--answer", help="question: retrieve + generate answer with citations")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--min_score", type=float, default=0.4)
    p.add_argument("--demo", nargs="+", help="Full demo: ingest docs, create exam, build topics, generate cards")
    args = p.parse_args()

    store = VectorStore()

    try:
        _run(args, store)
    finally:
        store.db.close()


def _run(args, store):
    # === DEMO MODE: Full automated flow ===
    if args.demo:
        print("\n" + "="*50)
        print("FLASHCARDS DEMO - Full Automated Flow")
        print("="*50 + "\n")
        
        # 1. Create exam
        print("[Step 1/4] Creating exam workspace...")
        exam_id = create_exam(
            store=store,
            user_id=args.user_id,
            title=args.exam_title,
            mode=args.exam_mode,
            info={"created_via": "demo"},
        )
        print(f"   Created exam_id={exam_id}\n")
        
        # 2. Ingest documents
        print("[Step 2/4] Ingesting documents...")
        results = ingest_documents(args.demo, store)
        for res in results:
            print(f"   Ingested {res.doc_id} ({res.num_chunks} chunks)")
        doc_ids = [res.doc_id for res in results]
        attach_documents(store=store, exam_id=exam_id, doc_ids=doc_ids)
        print(f"   Attached {len(doc_ids)} document(s) to exam\n")
        
        # 3. Build topics
        print("[Step 3/4] Building topics (clustering content)...")
        topics = build_topics_for_exam(
            exam_id=exam_id,
            store=store,
            overwrite=True,
            merge_threshold=args.topic_merge_threshold,
        )
        topic_list = list_topics_for_exam(exam_id=exam_id, store=store)
        print(f"   Created {len(topic_list)} topic(s):")
        for t in topic_list:
            print(f"      - {t.label}")
        print()
        
        # 4. Generate starter flashcards
        print("[Step 4/4] Generating flashcards...")
        start_time = time.time()
        cards = generate_starter_cards(
            exam_id=exam_id,
            user_id=args.user_id,
            store=store,
            n=5,
            difficulty=1,
        )
        elapsed = time.time() - start_time
        print(f"   Generated {len(cards)} flashcard(s) in {elapsed:.2f} seconds")
        if cards:
            print(f"   Average: {elapsed/len(cards):.2f} seconds per card\n")
        else:
            print()
        
        # 5. Display cards with topics, answers, and proofs
        print("="*50)
        print("GENERATED FLASHCARDS WITH PROOFS")
        print("="*50)
        for i, c in enumerate(cards, 1):
            print(f"\n{'-'*50}")
            print(f"Card {i} | Topic: {c.topic_label}")
            print(f"{'-'*50}")
            print(f"Q: {c.question}")
            print(f"\nA: {c.answer}")
            
            # Show proofs (source evidence)
            if c.proofs:
                print(f"\nProofs ({len(c.proofs)} source(s)):")
                for j, proof in enumerate(c.proofs[:3], 1):  # Show top 3 proofs
                    doc_id = proof.get('doc_id', 'unknown')
                    page = proof.get('page', '?')
                    score = proof.get('score', 0)
                    text = proof.get('text', '')
                    # Truncate long text
                    text_preview = " ".join(text.split())[:300]
                    if len(text) > 300:
                        text_preview += "..."
                    print(f"  [{j}] doc={doc_id} page={page} score={score:.2f}")
                    print(f"      \"{text_preview}\"")
        
        print(f"\n{'='*50}")
        print(f"Demo complete! exam_id={exam_id}")
        print(f"{'='*50}\n")
        return

    if args.gen_starter_cards:
        if not args.exam_id:
            raise SystemExit("--gen_starter_cards requires --exam_id")
        print(f"Generating starter cards for exam_id={args.exam_id}...")
        start_time = time.time()
        cards = generate_starter_cards(
            exam_id=args.exam_id,
            user_id=args.user_id,
            store=store,
            n=5,
            difficulty=1,
        )
        elapsed = time.time() - start_time
        print(f"\nGenerated {len(cards)} starter card(s) in {elapsed:.2f} seconds")
        if cards:
            print(f"Average: {elapsed/len(cards):.2f} seconds per card\n")
        for c in cards:
            print(f"\n{'='*60}")
            print(f"TOPIC: {c.topic_label}")
            print(f"Q: {c.question}")
            print(f"\nA: {c.answer}")
            if c.proofs:
                print(f"\nPROOFS ({len(c.proofs)} sources):")
                for j, proof in enumerate(c.proofs[:3], 1):
                    doc_id = proof.get('doc_id', 'unknown')
                    page = proof.get('page', '?')
                    score = proof.get('score', 0)
                    text = proof.get('text', '')[:150]
                    print(f"  [{j}] doc={doc_id} page={page} score={score:.2f}")
                    print(f"      \"{text}...\"")
        print(f"\n{'='*60}")
        print(f"Total time: {elapsed:.2f} seconds")
        return

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
