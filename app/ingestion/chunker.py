import re


def _split_long_text(text: str, max_length: int) -> list[str]:
    pieces: list[str] = []
    current = ""

    for word in text.split():
        while len(word) > max_length:
            if current:
                pieces.append(current)
                current = ""

            pieces.append(word[:max_length])
            word = word[max_length:]

        if not current:
            current = word
        elif len(current) + 1 + len(word) <= max_length:
            current = f"{current} {word}"
        else:
            pieces.append(current)
            current = word

    if current:
        pieces.append(current)

    return pieces


def _overlap_tail(text: str, overlap: int) -> str:
    if overlap <= 0:
        return ""

    tail = text[-overlap:]
    first_space = tail.find(" ")

    if first_space == -1:
        return tail

    return tail[first_space + 1:].strip()


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    normalized = re.sub(r"\s+", " ", text).strip()

    if not normalized:
        return []

    # Split into sentences while keeping the sentence-ending punctuation.
    sentences = re.split(r"(?<=[.!?])\s+", normalized)

    pieces: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if len(sentence) <= chunk_size:
            pieces.append(sentence)
        else:
            pieces.extend(_split_long_text(sentence, chunk_size))

    chunks: list[str] = []
    current = ""

    for piece in pieces:
        if not current:
            current = piece
            continue

        if len(current) + 1 + len(piece) <= chunk_size:
            current = f"{current} {piece}"
            continue

        chunks.append(current)

        tail = _overlap_tail(current, overlap)

        if tail and len(tail) + 1 + len(piece) <= chunk_size:
            current = f"{tail} {piece}"
        else:
            current = piece

    if current:
        chunks.append(current)

    return chunks
