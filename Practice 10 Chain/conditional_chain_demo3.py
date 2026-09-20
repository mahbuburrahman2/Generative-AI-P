from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_groq import ChatGroq

load_dotenv()

# --------------------------------------------------
# Model
# --------------------------------------------------

model = ChatGroq(model="openai/gpt-oss-120b")

parser = StrOutputParser()


# --------------------------------------------------
# 1. Classifier Prompt
# --------------------------------------------------

classifier_prompt = PromptTemplate(
    template="""
You are a movie review classifier.

Classify the following review as exactly one of:
- positive
- negative
- neutral

Return only one word: positive, negative, or neutral.

Review:
{review}
""",
    input_variables=["review"]
)

classifier_chain = classifier_prompt | model | parser


# --------------------------------------------------
# 2. Positive Prompt
# --------------------------------------------------

positive_prompt = PromptTemplate.from_template(
    """
Reply to this positive movie review in a friendly and enthusiastic way.

Review:
{review}
"""
)

positive_chain = positive_prompt | model | parser


# --------------------------------------------------
# 3. Negative Prompt
# --------------------------------------------------

negative_prompt = PromptTemplate.from_template(
    """
Reply to this negative movie review in a friendly and understanding way.
Acknowledge the viewer's criticism without being defensive.

Review:
{review}
"""
)

negative_chain = negative_prompt | model | parser


# --------------------------------------------------
# 4. Neutral Prompt
# --------------------------------------------------

neutral_prompt = PromptTemplate.from_template(
    """
Reply to this neutral movie review in a balanced and conversational way.
Do not assume that the viewer strongly liked or disliked the movie.

Review:
{review}
"""
)

neutral_chain = neutral_prompt | model | parser


# --------------------------------------------------
# 5. Default Prompt
# --------------------------------------------------

default_prompt = PromptTemplate.from_template(
    """
Reply to the following movie review in a general, polite, and helpful way.
Do not assume a particular sentiment.

Review:
{review}
"""
)

default_chain = default_prompt | model | parser


# --------------------------------------------------
# 6. Conditional Branch
# --------------------------------------------------

conditional_chain = RunnableBranch(
    (
        lambda x: x["sentiment"].strip().lower() == "positive",
        positive_chain
    ),
    (
        lambda x: x["sentiment"].strip().lower() == "negative",
        negative_chain
    ),
    (
        lambda x: x["sentiment"].strip().lower() == "neutral",
        neutral_chain
    ),
    default_chain
)


# --------------------------------------------------
# 7. Review
# --------------------------------------------------

review = "The movie was absolutely fantastic. I loved every minute of it."


# --------------------------------------------------
# 8. Classify the Review
# --------------------------------------------------

sentiment = classifier_chain.invoke({
    "review": review
})


# --------------------------------------------------
# 9. Run the Appropriate Branch
# --------------------------------------------------

result = conditional_chain.invoke({
    "review": review,
    "sentiment": sentiment
})


# --------------------------------------------------
# 10. Output
# --------------------------------------------------

print("Review:", review)
print("Sentiment:", sentiment)
print("Response:", result)