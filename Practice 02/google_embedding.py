import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

print("API key found:", bool(api_key))

client = genai.Client(api_key=api_key)

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents="Dhaka is the capital of Bangladesh."
)

embedding = result.embeddings[0].values

print("Embedding created successfully!")
print("Vector dimension:", len(embedding))
print("First 10 values:", embedding[:10])