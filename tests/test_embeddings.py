from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
)

texts = [
    "How does PostgreSQL handle concurrent transactions?",
    "What is PostgreSQL MVCC?",
    "How do I bake a chocolate cake?",
]

vectors = model.encode(
    texts,
    normalize_embeddings=True,
)

print("Vector shape:", vectors.shape)

print()
print("Similarity:")
print("transaction ↔ MVCC:",
      vectors[0] @ vectors[1])

print("transaction ↔ cake:",
      vectors[0] @ vectors[2])

print("MVCC ↔ cake:",
      vectors[1] @ vectors[2])