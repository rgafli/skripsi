import streamlit as st
import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# --- Load Models and Vectorizer ---
@st.cache_resource
def load_nb_model():
    return joblib.load("naive_bayes_model.pkl")

@st.cache_resource
def load_svm_model():
    return joblib.load("svm_model.pkl")

@st.cache_resource
def load_vectorizer():
    return joblib.load("tfidf_vectorizer.pkl")

@st.cache_resource
def load_bert_model():
    model = BertForSequenceClassification.from_pretrained("bert-finetuned-superapp")
    tokenizer = BertTokenizer.from_pretrained("bert-finetuned-superapp")
    return model, tokenizer

# --- Prediction Functions ---
def predict_with_nb(text, model, vectorizer):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return pred

def predict_with_svm(text, model, vectorizer):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return pred

def predict_with_bert(text, model, tokenizer):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return pred

# --- Streamlit App UI ---
st.set_page_config(page_title="Superapp Sentiment Analyzer", layout="centered")
st.title("🧠 Superapp Review Sentiment Analyzer")
st.markdown("Analyze customer feedback using **Naive Bayes**, **SVM**, or **BERT**.")

user_input = st.text_area("✍️ Enter your superapp review:", height=150)
model_choice = st.selectbox("📦 Choose a model to predict with:", ["Naive Bayes", "SVM", "BERT"])
predict_button = st.button("🔍 Analyze")

if predict_button and user_input.strip():
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

    try:
        if model_choice == "Naive Bayes":
            model = load_nb_model()
            vectorizer = load_vectorizer()
            label = predict_with_nb(user_input, model, vectorizer)

        elif model_choice == "SVM":
            model = load_svm_model()
            vectorizer = load_vectorizer()
            label = predict_with_svm(user_input, model, vectorizer)

        elif model_choice == "BERT":
            model, tokenizer = load_bert_model()
            label = predict_with_bert(user_input, model, tokenizer)

        st.success(f"🧾 **Predicted Sentiment:** {sentiment_map[label]}")
    except Exception as e:
        st.error("🚫 Model loading or prediction failed. Please check your model files.")
        st.exception(e)
