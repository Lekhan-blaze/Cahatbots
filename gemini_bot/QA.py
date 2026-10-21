from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

import os
import streamlit as st
from google import genai
from google.genai.errors import ServerError, APIError

st.set_page_config(page_title="GEMINI CHATBOT DEMO", page_icon="🤖")
st.header("Gemini Application")

# Sidebar for model selection
st.sidebar.title("Model Settings")
model_options = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-flash-latest",
    "gemini-3.8-flash",
]
selected_model = st.sidebar.selectbox("Choose Model", model_options, index=0)

# Clear chat history button
if st.sidebar.button("Clear Chat History"):
    st.session_state["chat_history"] = []
    st.session_state["chat"] = None
    st.rerun()

# Cache the Gemini client so Streamlit doesn't close HTTP connections across reruns
@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

client = get_gemini_client()

# Recreate chat if model changed or not initialized
if "current_model" not in st.session_state or st.session_state["current_model"] != selected_model:
    st.session_state["current_model"] = selected_model
    st.session_state["chat"] = client.chats.create(model=selected_model)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_input = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")

if submit and user_input:
    st.session_state["chat_history"].append(("YOU", user_input))
    
    st.subheader("The Response is")
    response_placeholder = st.empty()
    full_response = ""
    
    try:
        response = st.session_state["chat"].send_message_stream(user_input)
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                response_placeholder.write(full_response)
        
        st.session_state["chat_history"].append(("BOT", full_response))

    except ServerError as e:
        if "503" in str(e):
            st.warning(f"⚠️ {selected_model} is currently experiencing high demand on Google's servers. Trying fallback model (gemini-3.6-flash)...")
            try:
                fallback_chat = client.chats.create(model="gemini-3.6-flash")
                fallback_resp = fallback_chat.send_message(user_input)
                full_response = fallback_resp.text
                response_placeholder.write(full_response)
                st.session_state["chat_history"].append(("BOT", full_response))
            except Exception as e2:
                st.error(f"Models are currently busy. Please try again in a few moments. Error: {e2}")
        else:
            st.error(f"Server Error: {e}")
    except APIError as e:
        st.error(f"API Error: {e}")

st.subheader("Chat History")
for role, text in st.session_state["chat_history"]:
    st.write(f"**{role}:** {text}")