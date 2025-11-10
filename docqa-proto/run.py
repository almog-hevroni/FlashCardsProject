import argparse, json
from app.api import ingest_document, retrieve_with_proofs
from app.store.storage import VectorStore

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ingest", help="path to document to ingest")
    p.add_argument("--ask", help="question to ask")
    p.add_argument("--k", type=int, default=4)
    args = p.parse_args()

    store = VectorStore()

    if args.ingest:
        res = ingest_document(args.ingest, store)
        print(f"Ingested doc_id={res.doc_id} chunks={res.num_chunks}")

    if args.ask:
        proofs = retrieve_with_proofs(args.ask, k=args.k, store=store)
        print(json.dumps([p.model_dump() for p in proofs], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
