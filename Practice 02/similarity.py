from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

text1 = "I love programming."
text2 = "I enjoy coding."

# Create embeddings
embedding1 = model.encode([text1])
embedding2 = model.encode([text2])

# Calculate similarity
similarity = cosine_similarity(embedding1, embedding2)

print("Text 1:", text1)
print("Text 2:", text2)

print("\nSimilarity score:", similarity[0][0])