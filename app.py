import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import plotly.express as px
import pandas as pd

# Set page config
st.set_page_config(page_title="Fake News Detector", layout="wide")

@st.cache_resource
def load_model():
    # Using a smaller BERT model that's faster to load
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities).item()
    confidence = torch.max(probabilities).item()
    return "Reliable" if predicted_class == 1 else "Unreliable", confidence

def main():
    st.title("🔍 Advanced Fake News Detector")
    st.markdown("""
    This tool uses a fine-tuned BERT model to detect potentially fake or misleading news articles.
    Enter the text of a news article below to analyze its reliability.
    """)
    
    # Load model
    with st.spinner("Loading model (this may take a minute)..."):
        model, tokenizer = load_model()
    
    # Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter News Article")
        # Initialize session state for input_text if it doesn't exist
        if 'input_text' not in st.session_state:
            st.session_state.input_text = ""
            
        text = st.text_area(
            "Paste the article text here:",
            value=st.session_state.input_text,
            height=300,
            key="input_text"
        )
        
        if st.button("Analyze Article"):
            if text.strip():
                with st.spinner("Analyzing..."):
                    prediction, confidence = predict(text, model, tokenizer)
                    
                    st.subheader("Analysis Results")
                    if prediction == "Reliable":
                        st.success(f"✅ {prediction} (Confidence: {confidence:.1%})")
                    else:
                        st.error(f"⚠️ {prediction} (Confidence: {confidence:.1%})")
                    
                    # Simple progress bar for confidence
                    st.progress(int(confidence * 100))
                    
            else:
                st.warning("Please enter some text to analyze.")
    
    with col2:
        st.subheader("Example Articles")
        examples = [
            ("Reliable Example", "The government announced new climate policies today, aiming to reduce carbon emissions by 50% by 2030. The plan includes significant investments in renewable energy and electric vehicle infrastructure."),
            ("Unreliable Example", "Breaking: Miracle cure discovered! Doctors hate this one trick to cure all diseases! Government doesn't want you to know about this simple remedy!"),
        ]
        
        for title, example in examples:
            if st.button(title, key=title):
                st.session_state.input_text = example
                st.rerun()
        
        st.markdown("""
        ### About
        This tool uses a pre-trained BERT model fine-tuned on sentiment analysis to assess the reliability of news content.
        
        *Note: Always verify information from multiple sources.*
        """)

if __name__ == "__main__":
    main()
