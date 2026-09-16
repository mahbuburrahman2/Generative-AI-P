from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Summarize the following text in 3 short lines:\n{text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Write 3 simple questions from the following text:\n{text}",
    input_variables=["text"]
)

parallel_chain = RunnableParallel({
    "summary": prompt1 | model | parser,
    "questions": prompt2 | model | parser
})

text = """
Machine learning is a subset of artificial intelligence where computers learn patterns from data and make predictions.
"""

result = parallel_chain.invoke({"text": text})

print("Summary:\n", result["summary"])
print("\nQuestions:\n", result["questions"])