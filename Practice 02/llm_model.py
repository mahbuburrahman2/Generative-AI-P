import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

api_key = os.getenv("GEMINI_API_KEY")

print("API key found:", bool(api_key))

llm = GoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key
)

result = llm.invoke("What is the capital of Bangladesh?")

print(result)