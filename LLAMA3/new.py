from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os

os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", "")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert in data science and a helpful assistant. Your task is to provide clear, concise, and easy-to-understand responses to user queries. Always be polite, encouraging, and patient, especially with beginners. Ensure that your explanations are step-by-step and avoid using jargon unless it is explained in simple terms. Provide examples wherever possible to illustrate concepts.",
        ),
        ("user", "Question:{question}"),
    ]
)

st.title("LLM basic using Groq + Llama3 🚀")
input_text = st.text_input("Search the topic you want🙌")

llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0.7)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))