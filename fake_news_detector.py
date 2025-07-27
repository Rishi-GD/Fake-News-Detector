# Fake News Detector using Machine Learning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("news.csv")
X = df['text']
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Passive Aggressive Classifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(X_train_tfidf, y_train)

# Predict and evaluate
y_pred = pac.predict(X_test_tfidf)
score = accuracy_score(y_test, y_pred)

print(f"Accuracy: {round(score*100, 2)}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
