from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from app.embeddings.embedder import Embedder
from app.ingestion.chunker import chunk_text
from app.ingestion.loader import Document


@dataclass(frozen=True)
class ChunkRecord:
    text: str
    source: str
    chunk_index: int
    embedding: list[float]

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChunkRecord":
        return cls(
            text=data["text"],
            source=data["source"],
            chunk_index=data["chunk_index"],
            embedding=data["embedding"],
        )


@dataclass(frozen=True)
class SearchResult:
    score: float
    record: ChunkRecord


class VectorIndex:
    def __init__(
        self,
        records: Sequence[ChunkRecord],
        metadata: dict | None = None,
    ) -> None:
        self.records = list(records)
        self.metadata = metadata or {}
        self._matrix: np.ndarray | None = None

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            if not self.records:
                self._matrix = np.empty((0, 0), dtype=np.float32)
            else:
                self._matrix = np.asarray(
                    [record.embedding for record in self.records],
                    dtype=np.float32,
                )
                norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self._matrix = self._matrix / norms

        return self._matrix

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        if not self.records:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)
        norm = float(np.linalg.norm(query))

        if norm == 0:
            return []

        query = query / norm
        scores = self.matrix @ query

        results = [
            SearchResult(score=float(scores[index]), record=self.records[index])
            for index in np.argsort(-scores)
            if float(scores[index]) >= min_score
        ]

        return results[:max(1, top_k)]

    def save(self, path: str | Path) -> None:
        index_path = Path(path)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": self.metadata,
            "records": [record.to_dict() for record in self.records],
        }

        index_path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        return cls(
            records=[
                ChunkRecord.from_dict(record)
                for record in data.get("records", [])
            ],
            metadata=data.get("metadata", {}),
        )

    @staticmethod
    def fingerprint(
        documents: Sequence[Document],
        embedder: Embedder,
        chunk_size: int,
        chunk_overlap: int,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(b"local-rag-index-v1\0")
        digest.update(embedder.backend.encode("utf-8"))
        digest.update(b"\0")
        digest.update(embedder.model_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(f"{chunk_size}:{chunk_overlap}".encode("utf-8"))

        for document in documents:
            digest.update(b"\0")
            digest.update(document.source.encode("utf-8"))
            digest.update(b"\0")
            digest.update(document.text.encode("utf-8"))

        return digest.hexdigest()

    def matches(
        self,
        documents: Sequence[Document],
        embedder: Embedder,
        chunk_size: int,
        chunk_overlap: int,
    ) -> bool:
        expected = self.fingerprint(
            documents,
            embedder,
            chunk_size,
            chunk_overlap,
        )

        return self.metadata.get("fingerprint") == expected

    @classmethod
    def build(
        cls,
        documents: Sequence[Document],
        embedder: Embedder,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int = 64,
    ) -> "VectorIndex":
        pending: list[tuple[str, str, int]] = []

        for document in documents:
            chunks = chunk_text(
                document.text,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
            )

            for chunk_index, text in enumerate(chunks):
                pending.append((document.source, text, chunk_index))

        records: list[ChunkRecord] = []

        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            embeddings = embedder.embed_many([text for _, text, _ in batch])

            for (source, text, chunk_index), embedding in zip(batch, embeddings):
                records.append(
                    ChunkRecord(
                        text=text,
                        source=source,
                        chunk_index=chunk_index,
                        embedding=embedding,
                    )
                )

        metadata = {
            "fingerprint": cls.fingerprint(
                documents,
                embedder,
                chunk_size,
                chunk_overlap,
            ),
            "embedding_backend": embedder.backend,
            "embedding_model": embedder.model_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "document_count": len(documents),
            "chunk_count": len(records),
        }

        return cls(records=records, metadata=metadata)
