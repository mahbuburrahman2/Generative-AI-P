import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

print("API key found:", bool(api_key))

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    huggingfacehub_api_token=api_key,
    task="text-generation"
)

chat_model = ChatHuggingFace(llm=llm)

result = chat_model.invoke(
    "Write for me a poem about love in bangla language"
)

print(result.content)