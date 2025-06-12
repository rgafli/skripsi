import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("googleplaystore_user_reviews.csv")
df.dropna(subset=["Translated_Review", "Sentiment"], inplace=True)

# Encode Sentiment
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["Sentiment"])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df["Translated_Review"], df["label"], test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)

# Train models
nb_model = MultinomialNB().fit(X_train_vec, y_train)
svm_model = SVC(kernel='linear').fit(X_train_vec, y_train)

# Save models
joblib.dump(nb_model, "naive_bayes_model.pkl")
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Models and vectorizer saved successfully!")
