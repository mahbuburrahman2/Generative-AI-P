from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

# -------------------------
# Prompt 1: Detailed Evaluation
# -------------------------

prompt1 = PromptTemplate(
    template="""
Evaluate the following student's answer.

Question: {question}

Student's Answer:
{answer}

Provide a detailed evaluation including:
1. Correctness
2. Strengths
3. Weaknesses
4. Suggestions for improvement
""",
    input_variables=["question", "answer"],
)


# -------------------------
# Prompt 2: Concise Feedback
# -------------------------

prompt2 = PromptTemplate(
    template="""
Convert the following detailed evaluation into concise 5-point feedback.

Detailed Evaluation:
{evaluation}

Provide exactly 5 clear and useful feedback points.
""",
    input_variables=["evaluation"],
)


# -------------------------
# Model and Parser
# -------------------------

model = ChatGroq(model="openai/gpt-oss-120b")

parser = StrOutputParser()


# -------------------------
# Chain
# -------------------------

chain = prompt1 | model | parser | prompt2 | model | parser


# -------------------------
# Invoke Chain
# -------------------------

result = chain.invoke({
    "question": "What is Machine Learning?",
    "answer": """
Machine learning is a subset of artificial intelligence where computers
learn patterns from data and make predictions without being explicitly
programmed for every step.
"""
})


# -------------------------
# Print Result
# -------------------------

print(result)


# -------------------------
# Display Chain Graph
# -------------------------

chain.get_graph().print_ascii()