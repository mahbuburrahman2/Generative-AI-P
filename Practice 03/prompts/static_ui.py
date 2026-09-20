from langchain_groq import ChatGroq

from dotenv import load_dotenv
import streamlit as st


load_dotenv()

model = ChatGroq(model='openai/gpt-oss-120b')

st.header("Research Tool")

user_input = st.text_input("Enter your prompt")

if st.button("Summarize"):

    #model invoke -> with user prompt

    result = model.invoke(user_input)


    #show result -> st.write
    st.write(result.content)
