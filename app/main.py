import argparse
import shlex
import sys

from app import config
from app.database.sessions import Message, Session, SessionStore
from app.embeddings.embedder import Embedder, EmbeddingError
from app.ingestion.loader import load_documents, save_document
from app.llm.ollama_client import OllamaClient, OllamaError
from app.retrieval.retriever import SearchResult, VectorIndex

MAX_HISTORY_MESSAGES = 12

SYSTEM_PROMPT = (
    "You are a helpful, normal conversational AI assistant. Answer every "
    "user question naturally and completely from your own knowledge, just "
    "like a normal chat model. You may also receive retrieved excerpts from "
    "the user's documents and relevant earlier chat messages. Use the "
    "document excerpts only as supporting evidence to verify and enrich your "
    "answer; they are not a restriction on what you may say. If the excerpts "
    "contain relevant information, incorporate it and cite the source file "
    "name. If they do not, simply answer normally and never mention that the "
    "documents lack the answer. Never say things like 'I don't know based on "
    "the provided documents' or 'this is not available in the documents'. "
    "Never claim that a document says something it does not. Keep answers "
    "concise unless the user asks for more detail."
)


def ensure_index(embedder: Embedder, rebuild: bool = False) -> VectorIndex:
    documents = load_documents(config.DOCUMENTS_DIR)

    if not documents:
        print(
            "No documents loaded yet. The assistant will answer normally; "
            "use /add <path> to give it document support."
        )
        return VectorIndex(records=[])

    if not rebuild and config.INDEX_FILE.exists():
        try:
            index = VectorIndex.load(config.INDEX_FILE)

            if index.matches(
                documents,
                embedder,
                config.CHUNK_SIZE,
                config.CHUNK_OVERLAP,
            ):
                print(
                    f"Loaded existing index: {len(index.records)} chunks "
                    f"from {len(documents)} document(s)."
                )
                return index
        except (KeyError, ValueError, OSError):
            pass

    print(
        f"Indexing {len(documents)} document(s) with "
        f"{embedder.backend}/{embedder.model_name}..."
    )
    index = VectorIndex.build(
        documents,
        embedder,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    index.save(config.INDEX_FILE)
    print(f"Created index: {len(index.records)} chunks.")

    return index


def build_context(results: list[SearchResult]) -> str:
    blocks = []

    for position, result in enumerate(results, start=1):
        record = result.record
        blocks.append(
            f"[{position}] source: {record.source} "
            f"(chunk {record.chunk_index}, score {result.score:.3f})\n"
            f"{record.text}"
        )

    return "\n\n".join(blocks)


def build_user_message(question: str, results: list[SearchResult]) -> str:
    if not results:
        return f"User question: {question}"

    return (
        "Document excerpts for verification and support:\n"
        f"{build_context(results)}\n\n"
        f"User question: {question}"
    )


def memory_text(memory: list[Message]) -> str:
    return "\n".join(
        f"{message.role.capitalize()}: {message.content}"
        for message in memory
    )


def answer_question(
    question: str,
    index: VectorIndex,
    embedder: Embedder,
    client: OllamaClient,
    top_k: int,
    history: list[Message] | None = None,
    memory: list[Message] | None = None,
    query_embedding: list[float] | None = None,
) -> tuple[str, list[SearchResult], list[float]]:
    if query_embedding is None:
        query_embedding = embedder.embed(question)

    results = index.search(
        query_embedding,
        top_k=top_k,
        min_score=config.MIN_DOC_SCORE,
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if memory:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Potentially relevant earlier conversation for context:\n"
                    f"{memory_text(memory)}"
                ),
            }
        )

    if history:
        messages.extend(
            {"role": message.role, "content": message.content}
            for message in history
        )

    messages.append(
        {
            "role": "user",
            "content": build_user_message(question, results),
        }
    )

    answer = client.chat(messages)

    return answer, results, query_embedding


def run_turn(
    question: str,
    index: VectorIndex,
    embedder: Embedder,
    client: OllamaClient,
    store: SessionStore,
    session: Session,
    top_k: int,
) -> tuple[str, list[SearchResult]]:
    query_embedding = embedder.embed(question)
    history = store.load_messages(
        session.id,
        limit=MAX_HISTORY_MESSAGES,
    )
    recent_ids = {message.id for message in history}
    memory = store.search_messages(
        session.id,
        query_embedding,
        limit=config.MEMORY_MESSAGES,
        min_score=config.MIN_DOC_SCORE,
        exclude_ids=recent_ids,
    )

    answer, results, _ = answer_question(
        question,
        index,
        embedder,
        client,
        top_k,
        history=history,
        memory=memory,
        query_embedding=query_embedding,
    )

    try:
        answer_embedding = embedder.embed(answer)
    except EmbeddingError:
        answer_embedding = None

    store.add_message(session.id, "user", question, query_embedding)
    store.add_message(session.id, "assistant", answer, answer_embedding)

    return answer, results


def print_sources(results: list[SearchResult]) -> None:
    if not results:
        return

    sources = sorted({result.record.source for result in results})
    print("\nSupporting documents: " + ", ".join(sources))


def print_session(session: Session) -> None:
    if session.name:
        print(f"Session: {session.name} (id: {session.id})")
    else:
        print(f"Session: {session.id}")


def print_chat_help() -> None:
    print(
        "Commands:\n"
        "  /add <path>     copy a .txt/.md/.pdf document into documents/ and reindex\n"
        "  /docs           list the documents currently loaded\n"
        "  /reindex        rebuild the document index\n"
        "  /session        show the current session\n"
        "  /rename <name>  give the current session a name\n"
        "  /new [name]     start a new session\n"
        "  /sessions       list recent sessions\n"
        "  /help           show this help\n"
        "  /exit           quit"
    )


def add_document(source: str, embedder: Embedder) -> VectorIndex:
    destination = save_document(source, config.DOCUMENTS_DIR)
    print(f"Saved document: {destination.relative_to(config.DOCUMENTS_DIR)}")

    return ensure_index(embedder, rebuild=True)


def list_documents() -> None:
    documents = load_documents(config.DOCUMENTS_DIR)

    if not documents:
        print("No documents loaded.")
        return

    for document in documents:
        print(f"- {document.source} ({len(document.text)} characters)")


def list_sessions(store: SessionStore) -> None:
    sessions = store.list_sessions()

    if not sessions:
        print("No sessions yet.")
        return

    for session in sessions:
        print(
            f"- {session.label} (id: {session.id}, "
            f"{session.message_count} messages, updated {session.updated_at})"
        )


def handle_chat_command(
    line: str,
    index: VectorIndex,
    session: Session,
    embedder: Embedder,
    store: SessionStore,
) -> tuple[bool, VectorIndex, Session]:
    try:
        parts = shlex.split(line)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return False, index, session

    command = parts[0].lower()
    arguments = parts[1:]

    try:
        if command in {"/exit", "/quit"}:
            return True, index, session

        if command == "/help":
            print_chat_help()
        elif command == "/add":
            if not arguments:
                print("Usage: /add /path/to/document.md or /path/to/document.pdf")
            else:
                index = add_document(arguments[0], embedder)
        elif command == "/docs":
            list_documents()
        elif command == "/reindex":
            index = ensure_index(embedder, rebuild=True)
            print("Index rebuilt.")
        elif command == "/session":
            print_session(session)
        elif command == "/rename":
            if not arguments:
                print("Usage: /rename session name")
            else:
                session = store.rename_session(
                    session.id,
                    " ".join(arguments),
                )
                print(f"Session renamed to: {session.label}")
        elif command == "/new":
            name = " ".join(arguments).strip() or None
            session = store.create_session(name=name)
            print("Started a new session.")
            print_session(session)
        elif command == "/sessions":
            list_sessions(store)
        else:
            print(f"Unknown command: {command}. Type /help for options.")
    except (EmbeddingError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)

    return False, index, session


def run_chat(
    index: VectorIndex,
    embedder: Embedder,
    client: OllamaClient,
    store: SessionStore,
    session: Session,
    top_k: int,
) -> None:
    print("Local RAG chat. Ask anything, or use /add to give the assistant a document.")
    print_session(session)
    print_chat_help()

    while True:
        try:
            line = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line.lower() in {"exit", "quit", "q"}:
            break

        if line.startswith("/"):
            should_exit, index, session = handle_chat_command(
                line,
                index,
                session,
                embedder,
                store,
            )

            if should_exit:
                break

            continue

        try:
            answer, results = run_turn(
                line,
                index,
                embedder,
                client,
                store,
                session,
                top_k,
            )
        except (EmbeddingError, OllamaError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            continue

        print(f"\nAssistant: {answer}")
        print_sources(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local RAG chat app with persistent sessions.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="chat",
        choices=["chat", "ask", "add", "ingest", "reset", "sessions"],
        help=(
            "chat (default), ask a single question, add a document, ingest, "
            "reset the index, or list sessions."
        ),
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Question for 'ask', or document path for 'add'.",
    )
    parser.add_argument(
        "--document",
        action="append",
        default=[],
        metavar="PATH",
        help="Add a .txt/.md/.pdf document before starting. Can be repeated.",
    )
    parser.add_argument(
        "--session",
        metavar="ID_OR_NAME",
        help=(
            "Resume a session by id or name, or create a named session. "
            "Defaults to a new auto-generated session id."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=config.TOP_K,
        help=f"Number of chunks to retrieve (default: {config.TOP_K}).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the index even if the cached index is up to date.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "reset":
        if config.INDEX_FILE.exists():
            config.INDEX_FILE.unlink()
            print(f"Removed index: {config.INDEX_FILE}")
        else:
            print("No index to remove.")
        return 0

    if args.command == "sessions":
        store = SessionStore(config.SESSIONS_DB)
        list_sessions(store)
        return 0

    try:
        embedder = Embedder()
        document_paths = list(args.document)

        if args.command == "add":
            path = " ".join(args.question).strip()

            if not path:
                print(
                    "Error: provide a document path, e.g. "
                    './scripts/run.sh add "/path/to/file.md"',
                    file=sys.stderr,
                )
                return 2

            document_paths.append(path)

        saved_any = False

        for path in document_paths:
            try:
                destination = save_document(path, config.DOCUMENTS_DIR)
                print(
                    "Saved document: "
                    f"{destination.relative_to(config.DOCUMENTS_DIR)}"
                )
                saved_any = True
            except (FileNotFoundError, OSError, ValueError) as exc:
                print(f"Error: {exc}", file=sys.stderr)

        if document_paths and not saved_any:
            return 1

        index = ensure_index(
            embedder,
            rebuild=args.rebuild or saved_any,
        )

        if args.command in {"ingest", "add"}:
            return 0

        store = SessionStore(config.SESSIONS_DB)
        session = store.resolve_session(args.session)
        client = OllamaClient()

        if args.command == "ask":
            question = " ".join(args.question).strip()

            if not question:
                print(
                    'Error: provide a question, e.g. '
                    './scripts/run.sh ask "What is MVCC?"',
                    file=sys.stderr,
                )
                return 2

            print_session(session)
            answer, results = run_turn(
                question,
                index,
                embedder,
                client,
                store,
                session,
                args.top_k,
            )
            print(f"\nAssistant: {answer}")
            print_sources(results)
            return 0

        run_chat(index, embedder, client, store, session, args.top_k)
        return 0
    except (EmbeddingError, FileNotFoundError, OllamaError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
