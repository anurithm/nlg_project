import streamlit as st
from transformers import pipeline

# Page configuration
st.set_page_config(page_title="NLG Text Generator", page_icon="🤖")

# Title
st.title("🤖 Natural Language Generation Project")
st.write("Generate human-like text using AI (GPT-2 model)")

# Load model (cached to improve speed)
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="gpt2")

generator = load_model()

# User input
user_input = st.text_area(
    "Enter a prompt:",
    placeholder="Example: Artificial Intelligence is transforming the world because...",
    height=150
)

# Generate text
if st.button("Generate Text"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        with st.spinner("Generating text..."):
            output = generator(
                user_input,
                max_length=150,
                num_return_sequences=1
            )
            st.success("Generated Text:")
            st.write(output[0]["generated_text"])
