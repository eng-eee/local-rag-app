from app.ingestion.loader import load_text_file
from app.ingestion.chunker import chunk_text
from app.embeddings.embedder import Embedder


def main():
    document = load_text_file("documents/postgresql_intro.md")
    chunks = chunk_text(document)

    print(f"Document characters: {len(document)}")
    print(f"Number of chunks: {len(chunks)}")

    embedder = Embedder()

    for index, chunk in enumerate(chunks):
        print()
        print(f"===== CHUNK {index} =====")
        print(chunk)

        vector = embedder.embed(chunk)

        print()
        print(f"Vector dimensions: {len(vector)}")
        print(f"First 10 values: {vector[:10]}")


if __name__ == "__main__":
    main()