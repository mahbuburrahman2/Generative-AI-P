from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers.structured import (
    StructuredOutputParser,
    ResponseSchema
)

load_dotenv()

# Define the model
model = ChatGroq(
    model="openai/gpt-oss-120b"
)

# Define the response schema
response_schema = [
    ResponseSchema(
        name="fact_1",
        description="The first fact about the topic"
    ),
    ResponseSchema(
        name="fact_2",
        description="The second fact about the topic"
    ),
    ResponseSchema(
        name="fact_3",
        description="The third fact about the topic"
    ),
    ResponseSchema(
        name="fact_4",
        description="The fourth fact about the topic"
    ),
    ResponseSchema(
        name="fact_5",
        description="The fifth fact about the topic"
    )
]

# Create parser
parser = StructuredOutputParser.from_response_schemas(
    response_schema
)

# Define prompt
template = PromptTemplate(
    template="""
Give me 5 facts about {topic}.

{format_instruction}
""",
    input_variables=["topic"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)

# Create chain
chain = template | model | parser

# Invoke chain
result = chain.invoke(
    {"topic": "Machine Learning"}
)

print(result)