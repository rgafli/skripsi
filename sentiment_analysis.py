import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import torch
from transformers import BertTokenizer, BertForSequenceClassification, GPT2Tokenizer, GPT2ForSequenceClassification, AdamW

# Load the dataset
data = pd.read_csv("googleplaystore_user_reviews.csv")
data = data.dropna(subset=['Translated_Review', 'Sentiment'])

# Split dataset into features and labels
X = data['Translated_Review']
y = data['Sentiment']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# TF-IDF Vectorization for traditional models
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_pred = nb_model.predict(X_test_tfidf)

# SVM Model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)
svm_pred = svm_model.predict(X_test_tfidf)

# Evaluation for traditional models
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print("Naive Bayes Classification Report:\n", classification_report(y_test, nb_pred))
print("SVM Classification Report:\n", classification_report(y_test, svm_pred))

# BERT Implementation
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

def encode_text_bert(texts, tokenizer, max_length=128):
    return tokenizer.batch_encode_plus(
        texts.tolist(), max_length=max_length, padding=True, truncation=True, return_tensors="pt"
    )

X_train_encoded_bert = encode_text_bert(X_train, bert_tokenizer)
X_test_encoded_bert = encode_text_bert(X_test, bert_tokenizer)

# Optimizer and Training for BERT
optimizer_bert = AdamW(bert_model.parameters(), lr=1e-5)
epochs = 3

for epoch in range(epochs):
    bert_model.train()
    optimizer_bert.zero_grad()
    outputs = bert_model(**X_train_encoded_bert, labels=torch.tensor(y_train))
    loss = outputs.loss
    loss.backward()
    optimizer_bert.step()

# Evaluation for BERT
bert_model.eval()
with torch.no_grad():
    outputs = bert_model(**X_test_encoded_bert)
    predictions_bert = torch.argmax(outputs.logits, dim=-1)

bert_accuracy = (predictions_bert == torch.tensor(y_test)).float().mean().item()
print(f"BERT Test Accuracy: {bert_accuracy}")

# GPT Implementation
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=3)

def encode_text_gpt(texts, tokenizer, max_length=128):
    return tokenizer.batch_encode_plus(
        texts.tolist(), max_length=max_length, padding=True, truncation=True, return_tensors="pt"
    )

X_train_encoded_gpt = encode_text_gpt(X_train, gpt_tokenizer)
X_test_encoded_gpt = encode_text_gpt(X_test, gpt_tokenizer)

# Optimizer and Training for GPT
optimizer_gpt = AdamW(gpt_model.parameters(), lr=1e-5)
for epoch in range(epochs):
    gpt_model.train()
    optimizer_gpt.zero_grad()
    outputs = gpt_model(**X_train_encoded_gpt, labels=torch.tensor(y_train))
    loss = outputs.loss
    loss.backward()
    optimizer_gpt.step()

# Evaluation for GPT
gpt_model.eval()
with torch.no_grad():
    outputs_gpt = gpt_model(**X_test_encoded_gpt)
    predictions_gpt = torch.argmax(outputs_gpt.logits, dim=-1)

gpt_accuracy = (predictions_gpt == torch.tensor(y_test)).float().mean().item()
print(f"GPT Test Accuracy: {gpt_accuracy}")

# Results Comparison
print("\nResults Comparison:")
print(f"Naive Bayes Accuracy: {accuracy_score(y_test, nb_pred)}")
print(f"SVM Accuracy: {accuracy_score(y_test, svm_pred)}")
print(f"BERT Accuracy: {bert_accuracy}")
print(f"GPT Accuracy: {gpt_accuracy}")
