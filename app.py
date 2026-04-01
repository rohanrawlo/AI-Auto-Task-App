import streamlit as st
from transformers import pipeline

st.title("AI Document Summarizer")
st.write("Paste your text below and get a short summary instantly.")

# Use caching so the model only loads once, not on every refresh
@st.cache_resource
def load_model():
    return pipeline("summarization", model="google/flan-t5-base")

summarizer = load_model()

text = st.text_area("Paste your text here")

if st.button("Generate Summary"):
    if text:
        # T5 models usually don't need the prompt prepended for the "summarization" task
        # as the pipeline handles the prefix automatically.
        with st.spinner("Summarizing..."):
            result = summarizer(
                text, 
                max_length=100, 
                min_length=30, 
                do_sample=False
            )
            st.write(result[0]['summary_text'])
            st.success("Summary generated successfully!")
    else:
        st.warning("Please enter some text")