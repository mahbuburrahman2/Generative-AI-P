from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env
load_dotenv()

# LLM
model = ChatGroq(
    model="openai/gpt-oss-120b"
)

# Prompt
prompt_template = PromptTemplate.from_template(
    "Explain {topic} in simple terms"
)

# Output parser
parser = StrOutputParser()

# Chain
chain = prompt_template | model | parser

# Invoke
result = chain.invoke({
    "topic": "Machine Learning"
})

print(result)