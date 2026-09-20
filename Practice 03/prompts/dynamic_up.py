import json
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Force loading .env file from any parent folder
load_dotenv(find_dotenv())

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ `GROQ_API_KEY` was not detected in environment variables. Check your `.env` file.")
    st.stop()

# Initialize ChatGroq explicitly passing the key
model = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=groq_api_key
)

st.header("Research Tool")

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis",
    ],
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"],
)

length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)",
    ],
)

current_dir = Path(__file__).parent
template_path = current_dir / "template.json"

with open(template_path, "r", encoding="utf-8") as f:
    data = json.load(f)

template = PromptTemplate.from_template(data["template"])

if st.button("summarize"):
    prompt = template.invoke(
        {
            "paper_input": paper_input,
            "style_input": style_input,
            "length_input": length_input,
        }
    )

    try:
        result = model.invoke(prompt)
        st.write(result.content)
    except Exception as e:
        st.error(f"Error calling model: {e}")