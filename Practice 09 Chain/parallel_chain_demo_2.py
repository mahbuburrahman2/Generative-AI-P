from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_groq import ChatGroq

load_dotenv()


# -------------------------
# Models
# -------------------------

model1 = ChatGroq(
    model="openai/gpt-oss-120b"
)

model2 = ChatGroq(
    model="openai/gpt-oss-20b"
)


# -------------------------
# Prompt 1: Notes
# -------------------------

prompt1 = PromptTemplate(
    template="""
Generate short and simple notes from the following text:

{text}
""",
    input_variables=["text"],
)


# -------------------------
# Prompt 2: Quiz
# -------------------------

prompt2 = PromptTemplate(
    template="""
Create a short quiz from the following text.

{text}
""",
    input_variables=["text"],
)


# -------------------------
# Prompt 3: Merge
# -------------------------

prompt3 = PromptTemplate(
    template="""
Merge the provided notes and quiz into a single document.

Notes:
{notes}

Quiz:
{quiz}
""",
    input_variables=["notes", "quiz"],
)


# -------------------------
# Parser
# -------------------------

parser = StrOutputParser()


# -------------------------
# Parallel Chain
# -------------------------

parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | model1 | parser,
        "quiz": prompt2 | model2 | parser
    }
)


# -------------------------
# Merge Chain
# -------------------------

merge_chain = prompt3 | model1 | parser


# -------------------------
# Complete Chain
# -------------------------

chain = parallel_chain | merge_chain


# -------------------------
# Input
# -------------------------

text = """
Deep learning is a subset of machine learning powered by multilayer
artificial neural networks that can automatically learn complex patterns
and representations from large amounts of data.
"""


# -------------------------
# Invoke
# -------------------------

result = chain.invoke({
    "text": text
})


# -------------------------
# Output
# -------------------------

print(result)


# -------------------------
# Graph
# -------------------------

chain.get_graph().print_ascii()