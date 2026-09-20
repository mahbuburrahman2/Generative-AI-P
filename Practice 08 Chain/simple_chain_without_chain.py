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

# Step 1: Invoke the prompt with the topic
prompt_value = prompt.invoke({"topic": "Machine Learning"})

# Step 2: Pass the prompt value to the model to get a response
model_response = model.invoke(prompt_value)

# Step 3: Parse the model's response into a clean string
model_output = parser.invoke(model_response)

print(model_output)