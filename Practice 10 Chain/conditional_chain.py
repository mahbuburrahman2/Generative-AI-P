from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch
from langchain_groq import ChatGroq

load_dotenv()

# Model
model = ChatGroq(model="openai/gpt-oss-120b")

parser = StrOutputParser()

positive_chain = ChatPromptTemplate.from_template(
    "Replay to this positive movie review in a friend way: \n {review}\n",
)
negative_chain = ChatPromptTemplate.from_template(
    "Reply to this negative movie review in a friend way: \n {review}\n",
)

positive_chain = positive_chain | model | parser
negative_chain = negative_chain | model | parser

conditional_chain = RunnableBranch(
    (
        lambda  x : "good" in x["review"].lower(),positive_chain
    ),
    negative_chain
)

result1 = conditional_chain.invoke({
    "review": "The movie was really good and I enjoyed every moment"
})

result2 = conditional_chain.invoke({
    "review": "The movie was really bad"
})
print(result1)
print(result2)