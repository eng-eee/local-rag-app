from typing import Sequence

import requests

from app import config


class EmbeddingError(RuntimeError):
    pass


class Embedder:
    def __init__(
        self,
        backend: str | None = None,
        model: str | None = None,
    ) -> None:
        self.backend = (backend or config.EMBEDDING_BACKEND).lower()

        if self.backend == "ollama":
            self.model_name = model or config.OLLAMA_EMBEDDING_MODEL
        elif self.backend in {"sentence-transformers", "sentence_transformers", "st"}:
            self.backend = "sentence-transformers"
            self.model_name = model or config.SENTENCE_TRANSFORMER_MODEL
        else:
            raise ValueError(
                f"Unknown embedding backend: {backend!r}. "
                "Use 'ollama' or 'sentence-transformers'."
            )

        self._sentence_transformer = None

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        if self.backend == "ollama":
            return self._embed_ollama(texts)

        return self._embed_sentence_transformers(texts)

    def _embed_ollama(self, texts: Sequence[str]) -> list[list[float]]:
        url = f"{config.OLLAMA_BASE_URL}/api/embed"
        payload = {"model": self.model_name, "input": list(texts)}

        try:
            response = requests.post(url, json=payload, timeout=180)
        except requests.RequestException as exc:
            raise EmbeddingError(
                f"Could not reach Ollama at {config.OLLAMA_BASE_URL}. "
                "Start it with 'ollama serve'."
            ) from exc

        if response.status_code == 404:
            return self._embed_ollama_legacy(texts)

        if not response.ok:
            raise EmbeddingError(
                f"Ollama embedding request failed ({response.status_code}): "
                f"{response.text.strip()}"
            )

        embeddings = response.json().get("embeddings")

        if not embeddings or len(embeddings) != len(texts):
            raise EmbeddingError(
                f"Ollama returned an unexpected embedding response for model "
                f"{self.model_name!r}. Is the model installed?"
            )

        return embeddings

    def _embed_ollama_legacy(self, texts: Sequence[str]) -> list[list[float]]:
        url = f"{config.OLLAMA_BASE_URL}/api/embeddings"
        embeddings: list[list[float]] = []

        for text in texts:
            try:
                response = requests.post(
                    url,
                    json={"model": self.model_name, "prompt": text},
                    timeout=180,
                )
            except requests.RequestException as exc:
                raise EmbeddingError(
                    f"Could not reach Ollama at {config.OLLAMA_BASE_URL}. "
                    "Start it with 'ollama serve'."
                ) from exc

            if not response.ok:
                raise EmbeddingError(
                    f"Ollama embedding request failed ({response.status_code}): "
                    f"{response.text.strip()}"
                )

            embedding = response.json().get("embedding")

            if not embedding:
                raise EmbeddingError(
                    f"Ollama returned no embedding for model {self.model_name!r}. "
                    "Is the model installed?"
                )

            embeddings.append(embedding)

        return embeddings

    def _embed_sentence_transformers(
        self, texts: Sequence[str]
    ) -> list[list[float]]:
        if self._sentence_transformer is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is not installed. Either install it "
                    "or use EMBEDDING_BACKEND=ollama."
                ) from exc

            self._sentence_transformer = SentenceTransformer(
                self.model_name,
                device="cpu",
            )

        vectors = self._sentence_transformer.encode(
            list(texts),
            normalize_embeddings=True,
        )

        return vectors.tolist()
