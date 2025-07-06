import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model
import re


# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = text.lower().strip()  # Convert to lowercase and remove extra spaces
    return text

# Load models
@st.cache_resource
def load_models():
    sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
    classifier = load_model('comment_classifier.h5')
    return sentence_transformer, classifier

sentence_transformer, classifier = load_models()

# Define label columns
label_columns = ['IsToxic', 'IsAbusive', 'IsThreat', 'IsProvocative', 'IsObscene', 
                 'IsHatespeech', 'IsRacist', 'IsNationalist', 'IsSexist', 
                 'IsHomophobic', 'IsReligiousHate', 'IsRadicalism']

# Streamlit app
st.title("Comment Toxicity Classifier")
st.write("Enter a comment to classify its toxicity across multiple labels.")

# Text input
user_input = st.text_area("Enter your comment:", height=100)

if st.button("Classify"):
    if user_input:
        # Clean and encode the input text
        cleaned_text = clean_text(user_input)
        embedding = sentence_transformer.encode([cleaned_text])

        # Predict probabilities
        probabilities = classifier.predict(embedding)[0]

        # Display results
        st.subheader("Classification Results")
        results = pd.DataFrame({
            'Label': label_columns,
            'Probability': probabilities
        })
        results['Prediction'] = results['Probability'].apply(lambda x: 'Positive' if x > 0.3 else 'Negative')

        # Show results as a table
        st.table(results[['Label', 'Probability', 'Prediction']])

        
    else:
        st.warning("Please enter a comment to classify.")