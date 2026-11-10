from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Session:
    id: str
    name: str | None
    created_at: str
    updated_at: str
    message_count: int = 0

    @property
    def label(self) -> str:
        return self.name or self.id


@dataclass(frozen=True)
class Message:
    id: int
    role: str
    content: str
    created_at: str


def _session_from_row(row: sqlite3.Row) -> Session:
    keys = row.keys()

    return Session(
        id=row["id"],
        name=row["name"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        message_count=row["message_count"] if "message_count" in keys else 0,
    )


def _message_from_row(row: sqlite3.Row) -> Message:
    return Message(
        id=row["id"],
        role=row["role"],
        content=row["content"],
        created_at=row["created_at"],
    )


class SessionStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row

        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _init_schema(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    embedding BLOB,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, id)
                """
            )

    def create_session(self, name: str | None = None) -> Session:
        session_id = uuid.uuid4().hex[:12]
        normalized_name = name.strip() if name else None
        now = _now()

        with self._connection() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO sessions (id, name, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, normalized_name, now, now),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"Session name already in use: {normalized_name}"
                ) from exc

        session = self.get_session(session_id)

        if session is None:
            raise RuntimeError("Failed to create session")

        return session

    def get_session(self, session_id: str) -> Session | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT id, name, created_at, updated_at
                FROM sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()

        return _session_from_row(row) if row else None

    def get_session_by_name(self, name: str) -> Session | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT id, name, created_at, updated_at
                FROM sessions
                WHERE name = ?
                """,
                (name,),
            ).fetchone()

        return _session_from_row(row) if row else None

    def resolve_session(self, identifier: str | None = None) -> Session:
        if identifier:
            session = self.get_session(identifier) or self.get_session_by_name(
                identifier
            )

            if session is not None:
                return session

            return self.create_session(name=identifier)

        return self.create_session()

    def rename_session(self, session_id: str, name: str) -> Session:
        normalized_name = name.strip()

        if not normalized_name:
            raise ValueError("Session name cannot be empty")

        with self._connection() as connection:
            try:
                connection.execute(
                    """
                    UPDATE sessions
                    SET name = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (normalized_name, _now(), session_id),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"Session name already in use: {normalized_name}"
                ) from exc

        session = self.get_session(session_id)

        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        return session

    def list_sessions(self, limit: int = 20) -> list[Session]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    s.id,
                    s.name,
                    s.created_at,
                    s.updated_at,
                    (
                        SELECT COUNT(*)
                        FROM messages m
                        WHERE m.session_id = s.id
                    ) AS message_count
                FROM sessions s
                ORDER BY s.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [_session_from_row(row) for row in rows]

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        embedding: Sequence[float] | None = None,
    ) -> int:
        embedding_blob = None

        if embedding is not None:
            embedding_blob = np.asarray(
                embedding,
                dtype=np.float32,
            ).tobytes()

        now = _now()

        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO messages
                    (session_id, role, content, created_at, embedding)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, role, content, now, embedding_blob),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            message_id = int(cursor.lastrowid)

        return message_id

    def load_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[Message]:
        with self._connection() as connection:
            if limit is None:
                rows = connection.execute(
                    """
                    SELECT id, role, content, created_at
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY id ASC
                    """,
                    (session_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT id, role, content, created_at
                    FROM (
                        SELECT id, role, content, created_at
                        FROM messages
                        WHERE session_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                    )
                    ORDER BY id ASC
                    """,
                    (session_id, limit),
                ).fetchall()

        return [_message_from_row(row) for row in rows]

    def search_messages(
        self,
        session_id: str,
        query_embedding: Sequence[float],
        limit: int = 3,
        min_score: float = 0.0,
        exclude_ids: set[int] | None = None,
        max_candidates: int = 500,
    ) -> list[Message]:
        query = np.asarray(query_embedding, dtype=np.float32)
        query_norm = float(np.linalg.norm(query))

        if query_norm == 0:
            return []

        query = query / query_norm
        excluded = exclude_ids or set()

        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT id, role, content, created_at, embedding
                FROM messages
                WHERE session_id = ? AND embedding IS NOT NULL
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, max_candidates),
            ).fetchall()

        candidates: list[tuple[float, sqlite3.Row]] = []

        for row in rows:
            if row["id"] in excluded:
                continue

            vector = np.frombuffer(row["embedding"], dtype=np.float32)
            norm = float(np.linalg.norm(vector))

            if norm == 0:
                continue

            score = float((vector / norm) @ query)

            if score >= min_score:
                candidates.append((score, row))

        candidates.sort(key=lambda item: item[0], reverse=True)

        return [
            _message_from_row(row)
            for _, row in candidates[:max(1, limit)]
        ]
