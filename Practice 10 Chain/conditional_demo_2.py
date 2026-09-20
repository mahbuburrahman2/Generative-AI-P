from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_groq import ChatGroq

load_dotenv()

# Model
model = ChatGroq(model="openai/gpt-oss-120b")

# Output parser
parser = StrOutputParser()


# --------------------------------------------------
# 1. Sentiment Classifier Prompt
# --------------------------------------------------

classifier_prompt = PromptTemplate(
    template="""
You are a movie review classifier.

Classify the following review as either:
- positive
- negative

Return only one word: positive or negative.

Review: {review}
""",
    input_variables=["review"]
)

classifier_chain = classifier_prompt | model | parser


# --------------------------------------------------
# 2. Positive Response Chain
# --------------------------------------------------

positive_prompt = PromptTemplate.from_template(
    """
Reply to this positive movie review in a friendly way.

Review:
{review}
"""
)

positive_chain = positive_prompt | model | parser


# --------------------------------------------------
# 3. Negative Response Chain
# --------------------------------------------------

negative_prompt = PromptTemplate.from_template(
    """
Reply to this negative movie review in a friendly and helpful way.

Review:
{review}
"""
)

negative_chain = negative_prompt | model | parser


# --------------------------------------------------
# 4. Conditional Branch
# --------------------------------------------------

conditional_chain = RunnableBranch(
    (
        lambda x: x["sentiment"].strip().lower() == "positive",
        positive_chain
    ),
    negative_chain
)


# --------------------------------------------------
# 5. Movie Review
# --------------------------------------------------

review = "The movie was absolutely fantastic. I loved every minute of it."


# --------------------------------------------------
# 6. Classify Sentiment
# --------------------------------------------------

sentiment = classifier_chain.invoke({
    "review": review
})

print("Sentiment:", sentiment)


# --------------------------------------------------
# 7. Run Conditional Chain
# --------------------------------------------------

result = conditional_chain.invoke({
    "review": review,
    "sentiment": sentiment
})

print("Response:", result)