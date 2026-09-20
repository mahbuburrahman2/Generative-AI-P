import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

print("API key found:", bool(api_key))

chat_model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key
)

result = chat_model.invoke(
    "What is the capital of Bangladesh?"
)

print(result.content)