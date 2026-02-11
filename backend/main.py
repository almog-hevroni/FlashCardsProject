import argparse, json, logging, time, sys
from dotenv import load_dotenv
load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# PRESENTATION UTILITIES - Beautiful Terminal Output
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'
    
    @classmethod
    def disable(cls):
        """Disable colors for non-supporting terminals"""
        cls.HEADER = cls.BLUE = cls.CYAN = cls.GREEN = ''
        cls.YELLOW = cls.RED = cls.BOLD = cls.DIM = cls.RESET = ''

# Check if terminal supports colors
if sys.platform == 'win32':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except:
        Colors.disable()

def print_banner():
    """Print beautiful ASCII art banner"""
    banner = f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  {Colors.BOLD}{Colors.YELLOW}    ███████╗██╗      █████╗ ███████╗██╗  ██╗ ██████╗ █████╗ ██████╗ ██████╗   {Colors.CYAN}║
║  {Colors.BOLD}{Colors.YELLOW}    ██╔════╝██║     ██╔══██╗██╔════╝██║  ██║██╔════╝██╔══██╗██╔══██╗██╔══██╗  {Colors.CYAN}║
║  {Colors.BOLD}{Colors.YELLOW}    █████╗  ██║     ███████║███████╗███████║██║     ███████║██████╔╝██║  ██║  {Colors.CYAN}║
║  {Colors.BOLD}{Colors.YELLOW}    ██╔══╝  ██║     ██╔══██║╚════██║██╔══██║██║     ██╔══██║██╔══██╗██║  ██║  {Colors.CYAN}║
║  {Colors.BOLD}{Colors.YELLOW}    ██║     ███████╗██║  ██║███████║██║  ██║╚██████╗██║  ██║██║  ██║██████╔╝  {Colors.CYAN}║
║  {Colors.BOLD}{Colors.YELLOW}    ╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝   {Colors.CYAN}║
║                                                                              ║
║  {Colors.DIM}         AI-Powered Flashcard Generation System                            {Colors.CYAN}║
║  {Colors.DIM}         Intelligent Learning Through Document Analysis                    {Colors.CYAN}║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
    print(banner)

def print_step_header(step_num, total_steps, title, icon=""):
    """Print a beautiful step header"""
    progress = "█" * step_num + "░" * (total_steps - step_num)
    print(f"""
{Colors.CYAN}┌──────────────────────────────────────────────────────────────────────────────┐
│  {Colors.BOLD}{Colors.YELLOW}{icon} STEP {step_num}/{total_steps}: {title.upper():<55}{Colors.CYAN}│
│  {Colors.DIM}[{progress}] {int(step_num/total_steps*100):>3}%{Colors.CYAN}                                                    │
└──────────────────────────────────────────────────────────────────────────────┘{Colors.RESET}
""")

def print_success(message):
    """Print success message"""
    print(f"  {Colors.GREEN}✓{Colors.RESET} {message}")

def print_info(message):
    """Print info message"""
    print(f"  {Colors.CYAN}ℹ{Colors.RESET} {message}")

def print_item(message, indent=2):
    """Print list item"""
    spaces = " " * indent
    print(f"{spaces}{Colors.DIM}•{Colors.RESET} {message}")

def print_section_divider():
    """Print a section divider"""
    print(f"\n{Colors.DIM}{'─' * 80}{Colors.RESET}\n")

def print_card(card_num, topic, question, answer, proofs):
    """Print a beautifully formatted flashcard"""
    print(f"""
{Colors.CYAN}╭──────────────────────────────────────────────────────────────────────────────╮
│  {Colors.BOLD}{Colors.YELLOW}📚 FLASHCARD #{card_num:<64}{Colors.CYAN}│
│  {Colors.DIM}Topic: {topic[:66]:<66}{Colors.CYAN}│
╰──────────────────────────────────────────────────────────────────────────────╯{Colors.RESET}
""")
    
    # Question
    print(f"  {Colors.BOLD}{Colors.GREEN}❓ QUESTION:{Colors.RESET}")
    wrapped_q = _wrap_text(question, 72)
    for line in wrapped_q:
        print(f"     {line}")
    
    # Answer
    print(f"\n  {Colors.BOLD}{Colors.BLUE}💡 ANSWER:{Colors.RESET}")
    wrapped_a = _wrap_text(answer, 72)
    for line in wrapped_a:
        print(f"     {line}")
    
    # Proofs
    if proofs:
        print(f"\n  {Colors.BOLD}{Colors.DIM}📎 SOURCES ({len(proofs)} reference(s)):{Colors.RESET}")
        for i, proof in enumerate(proofs[:2], 1):
            doc_id = proof.get('doc_id', 'unknown')[:15]
            page = proof.get('page', '?')
            score = proof.get('score', 0)
            text = proof.get('text', '')
            text_preview = " ".join(text.split())[:120]
            if len(text) > 120:
                text_preview += "..."
            print(f"     {Colors.DIM}[{i}] {doc_id} (p.{page}) score: {score:.2f}{Colors.RESET}")
            print(f"         {Colors.DIM}\"{text_preview}\"{Colors.RESET}")

def _wrap_text(text, width):
    """Wrap text to specified width"""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= width:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
    
    if current_line:
        lines.append(" ".join(current_line))
    
    return lines if lines else [""]

def print_final_summary(exam_id, num_cards, elapsed_time):
    """Print beautiful final summary"""
    print(f"""
{Colors.GREEN}╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   {Colors.BOLD}✨ DEMO COMPLETED SUCCESSFULLY! ✨{Colors.GREEN}                                       ║
║                                                                              ║
║   {Colors.RESET}{Colors.GREEN}📋 Exam ID:        {exam_id:<55}║
║   📚 Cards Created:  {num_cards:<55}║
║   ⏱️  Time Elapsed:   {elapsed_time:.2f} seconds{' ' * 46}║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.RESET}
""")

def print_topics_box(topics):
    """Print topics in a nice box"""
    print(f"\n{Colors.CYAN}  ┌─────────────────────────────────────────────────────────────────────┐{Colors.RESET}")
    print(f"{Colors.CYAN}  │ {Colors.BOLD}{Colors.YELLOW}📂 DISCOVERED TOPICS{Colors.CYAN}                                               │{Colors.RESET}")
    print(f"{Colors.CYAN}  ├─────────────────────────────────────────────────────────────────────┤{Colors.RESET}")
    for t in topics:
        label = t.label[:63] if len(t.label) > 63 else t.label
        print(f"{Colors.CYAN}  │{Colors.RESET}  • {label:<64}{Colors.CYAN}│{Colors.RESET}")
    print(f"{Colors.CYAN}  └─────────────────────────────────────────────────────────────────────┘{Colors.RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Default logging - will be adjusted for demo mode
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
    p.add_argument("--quiet", action="store_true", help="Suppress logging output for clean demo")
    args = p.parse_args()

    store = VectorStore()

    try:
        _run(args, store)
    finally:
        store.db.close()


def _run(args, store):
    # === DEMO MODE: Full automated flow with beautiful output ===
    if args.demo:
        # Suppress all logging for clean presentation
        logging.getLogger().setLevel(logging.CRITICAL)
        for name in ['app.services.ingestion', 'app.services.graph', 'app.services.llm', 
                     'app.services.topics', 'app.services.cards', 'app.services.retrieval',
                     'app.data.vector_store', 'app.data.db', 'httpx', 'openai']:
            logging.getLogger(name).setLevel(logging.CRITICAL)
        
        demo_start_time = time.time()
        
        # Clear screen and show banner
        print("\033[2J\033[H", end="")  # Clear screen
        print_banner()
        
        time.sleep(0.5)  # Dramatic pause
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 1: Create Exam Workspace
        # ─────────────────────────────────────────────────────────────────────
        print_step_header(1, 4, "Creating Exam Workspace", "🎯")
        
        exam_id = create_exam(
            store=store,
            user_id=args.user_id,
            title=args.exam_title,
            mode=args.exam_mode,
            info={"created_via": "demo"},
        )
        
        print_success(f"Exam workspace created successfully!")
        print_info(f"Exam ID: {Colors.BOLD}{exam_id}{Colors.RESET}")
        print_info(f"Title: {args.exam_title}")
        print_info(f"Mode: {args.exam_mode}")
        
        time.sleep(0.3)
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 2: Ingest Documents
        # ─────────────────────────────────────────────────────────────────────
        print_step_header(2, 4, "Ingesting Documents", "📄")
        
        print_info("Processing documents...")
        results = ingest_documents(args.demo, store, user_id=args.user_id, exam_id=exam_id)
        
        for res in results:
            print_success(f"Ingested: {res.doc_id}")
            print_item(f"Chunks created: {res.num_chunks}", indent=4)
        
        doc_ids = [res.doc_id for res in results]
        attach_documents(store=store, exam_id=exam_id, doc_ids=doc_ids)
        
        print()
        print_success(f"Attached {len(doc_ids)} document(s) to exam")
        
        time.sleep(0.3)
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 3: Build Topics (Clustering)
        # ─────────────────────────────────────────────────────────────────────
        print_step_header(3, 4, "Analyzing & Clustering Content", "🧠")
        
        print_info("Running AI-powered topic extraction...")
        print_info("Clustering document content...")
        
        topics = build_topics_for_exam(
            exam_id=exam_id,
            store=store,
            overwrite=True,
            merge_threshold=args.topic_merge_threshold,
        )
        topic_list = list_topics_for_exam(exam_id=exam_id, store=store)
        
        print_success(f"Identified {len(topic_list)} distinct topic(s)")
        print_topics_box(topic_list)
        
        time.sleep(0.3)
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 4: Generate Flashcards
        # ─────────────────────────────────────────────────────────────────────
        print_step_header(4, 4, "Generating AI Flashcards", "✨")
        
        print_info("Generating intelligent flashcards with RAG...")
        print_info("This may take a moment...")
        print()
        
        card_start_time = time.time()
        cards = generate_starter_cards(
            exam_id=exam_id,
            user_id=args.user_id,
            store=store,
            n=5,
            difficulty=1,
        )
        card_elapsed = time.time() - card_start_time
        
        print_success(f"Generated {len(cards)} flashcard(s)")
        print_info(f"Generation time: {card_elapsed:.2f} seconds")
        if cards:
            print_info(f"Average: {card_elapsed/len(cards):.2f} seconds per card")
        
        # ─────────────────────────────────────────────────────────────────────
        # Display Generated Flashcards
        # ─────────────────────────────────────────────────────────────────────
        print_section_divider()
        
        print(f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║  {Colors.BOLD}{Colors.YELLOW}📚 GENERATED FLASHCARDS{Colors.CYAN}                                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.RESET}
""")
        
        for i, c in enumerate(cards, 1):
            print_card(
                card_num=i,
                topic=c.topic_label,
                question=c.question,
                answer=c.answer,
                proofs=c.proofs
            )
            if i < len(cards):
                print(f"\n{Colors.DIM}{'─' * 80}{Colors.RESET}")
        
        # ─────────────────────────────────────────────────────────────────────
        # Final Summary
        # ─────────────────────────────────────────────────────────────────────
        total_elapsed = time.time() - demo_start_time
        print_final_summary(exam_id, len(cards), total_elapsed)
        
        return

    # ═══════════════════════════════════════════════════════════════════════════════
    # NON-DEMO MODES (Original functionality preserved)
    # ═══════════════════════════════════════════════════════════════════════════════

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
        if not args.exam_id:
            raise SystemExit("--ingest requires --exam_id when using Pinecone backend")
        results = ingest_documents(args.ingest, store, user_id=args.user_id, exam_id=args.exam_id)
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
        if store.vector_backend == "pinecone":
            if not args.exam_id:
                raise SystemExit("--ask requires --exam_id when using Pinecone backend")
            ex = store.db.get_exam(args.exam_id)
            if ex is None:
                raise SystemExit(f"Exam not found: {args.exam_id}")
            store.set_namespace(f"u:{ex.user_id}|e:{args.exam_id}")
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

        if store.vector_backend == "pinecone":
            if not args.exam_id:
                raise SystemExit("--answer requires --exam_id when using Pinecone backend")
            ex = store.db.get_exam(args.exam_id)
            if ex is None:
                raise SystemExit(f"Exam not found: {args.exam_id}")
            store.set_namespace(f"u:{ex.user_id}|e:{args.exam_id}")
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
