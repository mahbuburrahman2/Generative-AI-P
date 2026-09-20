import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

print("API key found:", bool(api_key))

client = genai.Client(api_key=api_key)

texts = [
    "Dhaka is the capital of Bangladesh.",
    "Python is a programming language.",
    "Machine learning is a part of artificial intelligence.",
    "Iraq is located in the Middle East.",
    "Deep learning uses neural networks."
]

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=texts
)

print("\nBatch embedding completed!\n")

for i, embedding in enumerate(result.embeddings):
    print(f"Text {i + 1}: {texts[i]}")
    print("Vector dimension:", len(embedding.values))
    print("First 5 values:", embedding.values[:5])
    print("-" * 50)