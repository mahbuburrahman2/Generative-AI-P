from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

prompt = PromptTemplate(
    template="""
    You are an AI Tutor.
    Explain the following topic to a student in a simple and beginner-friendly way: {topic}
    """,
    input_variables=["topic"]
)

model = ChatGroq(model="openai/gpt-oss-120b")
parser = StrOutputParser()

chain = prompt | model | parser

chain.invoke({"topic": "Machine Learning"})

result = chain.invoke({"topic": "Machine Learning"})
print(result)

chain.get_graph().print_ascii()