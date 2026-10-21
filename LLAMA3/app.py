import streamlit as st
from langchain_groq import ChatGroq
import os


os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", "")

llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0.7, streaming=True)

st.title("💬 Llama3 Chatbot (Groq)")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(
        msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"
    ).write(msg["content"])


def generate_response():
    prompt = st.session_state.messages[-1]["content"]
    response = llm.stream(prompt)
    full_response = ""
    for chunk in response:
        token = chunk.content
        full_response += token
        yield token
    st.session_state["full_message"] = full_response


if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="🧑‍💻").write(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        response = st.write_stream(generate_response())

    st.session_state.messages.append(
        {"role": "assistant", "content": st.session_state["full_message"]}
    )