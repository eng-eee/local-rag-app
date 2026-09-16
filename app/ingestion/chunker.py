import re


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

    # Split into sentences while keeping the sentence-ending punctuation.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        # If adding the sentence still fits, keep building the chunk.
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
            continue

        # Store the current chunk.
        if current:
            chunks.append(current)

        # Start the next chunk.
        current = sentence

    if current:
        chunks.append(current)

    return chunks