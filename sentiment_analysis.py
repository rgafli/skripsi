import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

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

# -------------------
# BERT Implementation
# -------------------

# Tokenization
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

def encode_text_bert(texts, tokenizer, max_length=128):
    return tokenizer.batch_encode_plus(
        texts.tolist(),
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

# Encode train and test sets
X_train_encoded_bert = encode_text_bert(X_train, bert_tokenizer)
X_test_encoded_bert = encode_text_bert(X_test, bert_tokenizer)

# Optimizer
optimizer_bert = AdamW(bert_model.parameters(), lr=1e-5)

# Training loop
epochs = 3
bert_model.train()
for epoch in range(epochs):
    optimizer_bert.zero_grad()
    outputs = bert_model(**X_train_encoded_bert, labels=torch.tensor(y_train))
    loss = outputs.loss
    print(f"BERT - Epoch {epoch+1}, Loss: {loss.item():.4f}")
    loss.backward()
    optimizer_bert.step()

# Evaluation
bert_model.eval()
with torch.no_grad():
    outputs = bert_model(**X_test_encoded_bert)
    predictions_bert = torch.argmax(outputs.logits, dim=-1)

bert_accuracy = (predictions_bert == torch.tensor(y_test)).float().mean().item()
print(f"BERT Test Accuracy: {bert_accuracy:.4f}")

# ----------------------
# Results Comparison
# ----------------------
print("\n📊 Results Comparison:")
print(f"Naive Bayes Accuracy: {accuracy_score(y_test, nb_pred):.4f}")
print(f"SVM Accuracy: {accuracy_score(y_test, svm_pred):.4f}")
print(f"BERT Accuracy: {bert_accuracy:.4f}")
