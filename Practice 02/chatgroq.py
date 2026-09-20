import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

print("API key found:", bool(api_key))

chat_model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=api_key
)

result = chat_model.invoke(
    "What is the capital of Bangladesh?"
)

print(result.content)