import streamlit as st
from transformers import pipeline

# Title of app
st.title("AI Document Summarizer")
st.write("Paste your text below and get a short summary instantly.")
# Input text box
text = st.text_area("Paste your text here")

# Load Google AI model
summarizer = pipeline("text2text-generation", model="google/flan-t5-base")

# Button
if st.button("Generate Summary"):
    if text:
        prompt = "Summarize the following text in 2-3 clear sentences, focusing on the main idea and key points:\n" + text

        result = summarizer(
            prompt,
            max_length=100,
            do_sample=False
        )

        st.write(result[0]['generated_text'])
        st.success("Summary generated successfully!")
    else:
        st.write("Please enter some text")
