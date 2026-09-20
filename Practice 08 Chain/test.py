student_answer = """
   Python is a high-level programming language.
   It is widely used in web development, data science,
   and artificial intelligence.
"""


# Step 1: Clean the answer
def clean_answer(answer: str) -> str:
    return answer.strip()


# Step 2: Count the words
def count_words(answer: str) -> int:
    return len(answer.split())


# Step 3: Generate feedback
def generate_feedback(answer: str, word_count: int) -> str:
    return f"""
    The answer contains {word_count} words.
    The student correctly explained Python and mentioned some important application areas.
    """


# Step 4: Format the feedback
def format_feedback(feedback: str) -> str:
    return feedback.strip().upper()


# Execution pipeline
cleaned_answer = clean_answer(student_answer)
word_count = count_words(cleaned_answer)
raw_feedback = generate_feedback(cleaned_answer, word_count)
final_feedback = format_feedback(raw_feedback)

# Output the final result
print(final_feedback)