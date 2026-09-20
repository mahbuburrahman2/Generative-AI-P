from sentence_transformers import SentenceTransformer

# Load local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "Dhaka is the capital of Bangladesh.",
    "Python is a programming language.",
    "Machine learning is a part of artificial intelligence.",
    "Iraq is located in the Middle East.",
    "Deep learning uses neural networks."
]

# Create embeddings
embeddings = model.encode(texts)

print("Embedding completed!")
print("Number of texts:", len(embeddings))
print("Vector dimension:", embeddings.shape[1])

for i, embedding in enumerate(embeddings):
    print(f"\nText {i + 1}: {texts[i]}")
    print("First 5 values:", embedding[:5])