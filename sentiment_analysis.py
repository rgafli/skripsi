import streamlit as st
import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification

LABELS = ["Negative", "Neutral", "Positive"]

# Load Naive Bayes model
@st.cache_resource
def load_nb_model():
    return joblib.load("naive_bayes_model.pkl")

# Load SVM model
@st.cache_resource
def load_svm_model():
    return joblib.load("svm_model.pkl")

# Load TF-IDF vectorizer
@st.cache_resource
def load_vectorizer():
    return joblib.load("tfidf_vectorizer.pkl")

# Load BERT model and tokenizer from local folder
@st.cache_resource
def load_bert_model():
    model = BertForSequenceClassification.from_pretrained(
        "bert-finetuned-superapp", local_files_only=True
    )
    tokenizer = BertTokenizer.from_pretrained(
        "bert-finetuned-superapp", local_files_only=True
    )
    return model, tokenizer

# Predict sentiment
def predict_sentiment(text, model_choice):
    if model_choice in ["Naive Bayes", "SVM"]:
        vectorizer = load_vectorizer()
        vec_text = vectorizer.transform([text])
        model = load_nb_model() if model_choice == "Naive Bayes" else load_svm_model()
        pred = model.predict(vec_text)[0]
        return LABELS[pred]
    elif model_choice == "BERT":
        model, tokenizer = load_bert_model()
        model.eval()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        return LABELS[pred]
    return "Unknown"

# Streamlit App
st.title("🧠 Superapp Review Sentiment Analyzer")
st.markdown("Analyze customer feedback using Naive Bayes, SVM, or BERT.")

text_input = st.text_area("✍️ Enter your superapp review:")

model_choice = st.selectbox("📦 Choose a model to predict with:", ["Naive Bayes", "SVM", "BERT"])

if st.button("🔍 Analyze"):
    if text_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        try:
            result = predict_sentiment(text_input, model_choice)
            if result == "Positive":
                st.success(f"🟢 Sentiment: {result}")
            elif result == "Neutral":
                st.warning(f"🟡 Sentiment: {result}")
            else:
                st.error(f"🔴 Sentiment: {result}")
        except Exception as e:
            st.error(f"🚫 Model loading or prediction failed.\n\nError: {str(e)}")
