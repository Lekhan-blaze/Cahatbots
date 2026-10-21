from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

import os
import streamlit as st
from PIL import Image
from google import genai
from google.genai.errors import ServerError, APIError

st.set_page_config(page_title="Gemini Vision Bot Demo", page_icon="🖼️")
st.header("Gemini Vision Application")

# Sidebar for model selection
st.sidebar.title("Model Settings")
model_options = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-flash-latest",
    "gemini-3.8-flash",
]
selected_model = st.sidebar.selectbox("Choose Model", model_options, index=0)

# Cache the client so it persists across reruns
@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

client = get_gemini_client()

## Function to load Gemini model and get response
def get_gemini_response(input_prompt, img, model_name):
    contents = []
    if input_prompt:
        contents.append(input_prompt)
    if img:
        contents.append(img)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
        )
        return response.text
    except ServerError as e:
        if "503" in str(e):
            st.warning(f"⚠️ {model_name} is currently experiencing high demand. Falling back to gemini-3.6-flash...")
            fallback = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=contents,
            )
            return fallback.text
        raise e


input_prompt = st.text_input("Input Prompt: ", key="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_container_width=True)

submit = st.button("Tell me about the image")

## If ask button is clicked
if submit:
    if image is not None:
        try:
            with st.spinner("Analyzing image..."):
                response = get_gemini_response(input_prompt, image, selected_model)
                st.subheader("The Response is")
                st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please upload an image first.")