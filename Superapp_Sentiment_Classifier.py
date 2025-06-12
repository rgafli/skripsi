# 🧠 SUPERAPP SENTIMENT CLASSIFIER
# Google Colab script to train and export Naive Bayes, SVM, and BERT models for sentiment analysis.

# STEP 1: INSTALL DEPENDENCIES
# pip install transformers datasets scikit-learn pandas joblib torch

# STEP 2: UPLOAD CSV FILE
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from google.colab import files
uploaded = files.upload()  # Upload 'superapp_reviews.csv'

# STEP 3: DATA PREPROCESSING
df = pd.read_csv("superapp_reviews.csv")
df.dropna(inplace=True)
le = LabelEncoder()
df["label"] = le.fit_transform(df["Sentiment"])
df = df[["Translated_Review", "label"]]

# STEP 4: TRAIN & SAVE NAIVE BAYES AND SVM
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import joblib

X_train, X_test, y_train, y_test = train_test_split(df["Translated_Review"], df["label"], test_size=0.2)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
joblib.dump(nb_model, "naive_bayes_model.pkl")

svm_model = LinearSVC()
svm_model.fit(X_train_tfidf, y_train)
joblib.dump(svm_model, "svm_model.pkl")

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Download models if needed
files.download("naive_bayes_model.pkl")
files.download("svm_model.pkl")
files.download("tfidf_vectorizer.pkl")

# STEP 5: FINE-TUNE BERT
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import torch

hf_dataset = Dataset.from_pandas(df)
hf_dataset = hf_dataset.train_test_split(test_size=0.2)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(example["Translated_Review"], padding="max_length", truncation=True)

tokenized_dataset = hf_dataset.map(tokenize, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

training_args = TrainingArguments(
    output_dir="./bert-finetuned-superapp",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_dir="./logs",
    save_total_limit=1,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)

trainer.train()
model.save_pretrained("bert-finetuned-superapp")
tokenizer.save_pretrained("bert-finetuned-superapp")

# Zip and download the BERT model
import os
os.system("zip -r bert-finetuned-superapp.zip bert-finetuned-superapp")
files.download("bert-finetuned-superapp.zip")
