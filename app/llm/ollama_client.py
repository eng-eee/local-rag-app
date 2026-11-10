import re

import requests

from app import config


class OllamaError(RuntimeError):
    pass


def _strip_reasoning(text: str) -> str:
    for closing_tag in ("</think>", "</think>"):
        index = text.rfind(closing_tag)

        if index != -1:
            text = text[index + len(closing_tag):]
            break

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    return text.strip()


class OllamaClient:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int = 300,
    ) -> None:
        self.model = model or config.OLLAMA_MODEL
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        return self.chat(messages)

    def chat(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.2},
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise OllamaError(
                f"Could not reach Ollama at {self.base_url}. "
                "Start it with 'ollama serve'."
            ) from exc

        if not response.ok:
            raise OllamaError(
                f"Ollama request failed ({response.status_code}): "
                f"{response.text.strip()}"
            )

        message = response.json().get("message", {})
        content = message.get("content") or message.get("thinking") or ""

        answer = _strip_reasoning(content)

        if not answer:
            raise OllamaError(
                f"Ollama returned an empty answer for model {self.model!r}. "
                "Check that the model is installed with 'ollama list'."
            )

        return answer
