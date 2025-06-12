import streamlit as st
import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Define sentiment labels
LABELS = ["Negative", "Neutral", "Positive"]

# Load Naive Bayes model
@st.cache_resource
def load_nb_model():
    return joblib.load("naive_bayes_model.pkl")

# Load SVM model
@st.cache_resource
def load_svm_model():
    return joblib.load("svm_model.pkl")

# Load TF-IDF Vectorizer
@st.cache_resource
def load_vectorizer():
    return joblib.load("tfidf_vectorizer.pkl")

# Load BERT model and tokenizer locally
@st.cache_resource
def load_bert_model():
    model = BertForSequenceClassification.from_pretrained(
        "./bert-finetuned-superapp", local_files_only=True
    )
    tokenizer = BertTokenizer.from_pretrained(
        "./bert-finetuned-superapp", local_files_only=True
    )
    return model, tokenizer

# Prediction function
def predict_sentiment(text, model_name):
    if model_name in ["Naive Bayes", "SVM"]:
        vectorizer = load_vectorizer()
        vec_text = vectorizer.transform([text])
        if model_name == "Naive Bayes":
            model = load_nb_model()
        else:
            model = load_svm_model()
        pred = model.predict(vec_text)[0]
        return LABELS[pred]
    elif model_name == "BERT":
        model, tokenizer = load_bert_model()
        model.eval()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        return LABELS[pred]
    return "Unknown"

# Streamlit App UI
st.title("🧠 Superapp Review Sentiment Analyzer")
st.markdown("Analyze customer feedback using Naive Bayes, SVM, or BERT.")

# Input text
text_input = st.text_area("✍️ Enter your superapp review:")

# Model selection
model_choice = st.selectbox("📦 Choose a model to predict with:", ["Naive Bayes", "SVM", "BERT"])

# Analyze button
if st.button("🔍 Analyze"):
    if text_input.strip() == "":
        st.warning("Please enter a review first.")
    else:
        try:
            prediction = predict_sentiment(text_input, model_choice)
            # Show result with color-coded output
            if prediction == "Negative":
                st.error(f"Predicted Sentiment: {prediction}")
            elif prediction == "Neutral":
                st.warning(f"Predicted Sentiment: {prediction}")
            else:
                st.success(f"Predicted Sentiment: {prediction}")
        except Exception as e:
            st.exception(f"🚫 Model loading or prediction failed. Error: {e}")
