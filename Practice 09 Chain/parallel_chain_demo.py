from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_groq import ChatGroq

load_dotenv()

# Model
model = ChatGroq(model="openai/gpt-oss-120b")

# Parser
parser = StrOutputParser()


# Prompt 1: Correctness
correctness_prompt = PromptTemplate(
    template="""
Evaluate the correctness of the following student's answer.

Question: {question}

Student's Answer:
{answer}

Focus only on correctness.
""",
    input_variables=["question", "answer"],
)


# Prompt 2: Strengths
strength_prompt = PromptTemplate(
    template="""
Identify the strengths of the following student's answer.

Question: {question}

Student's Answer:
{answer}

Focus only on strengths.
""",
    input_variables=["question", "answer"],
)


# Prompt 3: Weaknesses
weakness_prompt = PromptTemplate(
    template="""
Identify the weaknesses of the following student's answer.

Question: {question}

Student's Answer:
{answer}

Focus only on weaknesses.
""",
    input_variables=["question", "answer"],
)


# Create individual chains
correctness_chain = correctness_prompt | model | parser
strength_chain = strength_prompt | model | parser
weakness_chain = weakness_prompt | model | parser


# Parallel Chain
parallel_chain = RunnableParallel(
    correctness=correctness_chain,
    strengths=strength_chain,
    weaknesses=weakness_chain
)


# Invoke
result = parallel_chain.invoke({
    "question": "What is Machine Learning?",
    "answer": """
    Machine learning is a subset of artificial intelligence where computers
    learn patterns from data and make predictions without being explicitly
    programmed for every step.
    """
})


print("CORRECTNESS:")
print(result["correctness"])

print("\nSTRENGTHS:")
print(result["strengths"])

print("\nWEAKNESSES:")
print(result["weaknesses"])

# Graph
parallel_chain.get_graph().print_ascii()