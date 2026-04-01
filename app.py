import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

st.title("AI Document Summarizer")

@st.cache_resource
def load_summarizer():
    model_name = "google/flan-t5-base"
    # Explicitly load the model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Pass the loaded objects directly into the pipeline
    return pipeline("summarization", model=model, tokenizer=tokenizer)

summarizer = load_summarizer()

text = st.text_area("Paste your text here")

if st.button("Generate Summary"):
    if text:
        with st.spinner("Summarizing..."):
            # Flan-T5 usually likes a prefix like "summarize: "
            input_text = "summarize: " + text
            result = summarizer(input_text, max_length=100, min_length=30, do_sample=False)
            st.write(result[0]['summary_text'])
    else:
        st.warning("Please enter some text")