import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

from app.config import SUPPORTED_EXTENSIONS


@dataclass(frozen=True)
class Document:
    source: str
    text: str


def _read_pdf(file_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF support requires pypdf. Install it with "
            "'python -m pip install pypdf'."
        ) from exc

    try:
        reader = PdfReader(str(file_path))

        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF is password-protected")

        pages = []

        for page in reader.pages:
            text = (page.extract_text() or "").strip()

            if text:
                pages.append(text)

        return "\n\n".join(pages)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read PDF: {exc}") from exc


def _read_document_text(file_path: Path) -> str:
    if file_path.suffix.lower() == ".pdf":
        return _read_pdf(file_path)

    return file_path.read_text(encoding="utf-8")


def load_text_file(path: str) -> str:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    return _read_document_text(file_path)


def load_documents(directory: str | Path) -> list[Document]:
    documents_dir = Path(directory)

    if not documents_dir.exists():
        return []

    if not documents_dir.is_dir():
        raise ValueError(f"Path is not a directory: {documents_dir}")

    documents: list[Document] = []

    for file_path in sorted(documents_dir.rglob("*")):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            text = _read_document_text(file_path).strip()
        except (OSError, RuntimeError, ValueError) as exc:
            warnings.warn(f"Skipping {file_path.name}: {exc}")
            continue

        if not text:
            warnings.warn(
                f"Skipping {file_path.name}: no extractable text "
                "(scanned PDFs need OCR)."
            )
            continue

        documents.append(
            Document(
                source=str(file_path.relative_to(documents_dir)),
                text=text,
            )
        )

    return documents


def save_document(source: str | Path, documents_dir: str | Path) -> Path:
    source_path = Path(source).expanduser().resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"File not found: {source_path}")

    if not source_path.is_file():
        raise ValueError(f"Path is not a file: {source_path}")

    if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported document type {source_path.suffix!r}. "
            f"Supported types: {supported}"
        )

    destination_dir = Path(documents_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source_path.name

    if destination.exists():
        if destination.resolve() == source_path:
            return destination

        if destination.read_bytes() == source_path.read_bytes():
            return destination

        counter = 1
        while destination.exists():
            destination = destination_dir / (
                f"{source_path.stem}-{counter}{source_path.suffix}"
            )
            counter += 1

    shutil.copy2(source_path, destination)

    return destination

