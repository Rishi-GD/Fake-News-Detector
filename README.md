# 📰 Fake News Detector

A machine learning project that detects fake news using a TF-IDF Vectorizer and PassiveAggressiveClassifier.

## 🧰 Tech Stack
- Python
- scikit-learn
- pandas
- TfidfVectorizer
- PassiveAggressiveClassifier

## 📊 Dataset
The dataset should be named `news.csv` and must contain two columns:
- `text`: The news content
- `label`: Either `FAKE` or `REAL`

You can find such datasets on Kaggle: [Fake News Dataset](https://www.kaggle.com/c/fake-news/data)

## 📈 Model
- Vectorizer: TF-IDF
- Model: PassiveAggressiveClassifier
- Accuracy ~92% on test data

## 🚀 How to Run
1. Place `news.csv` in the project folder
2. Run `python fake_news_detector.py`

## 📁 Files
- `fake_news_detector.py`: Main model code
- `README.md`: Documentation

👨‍💻 Created by Rishi Verma
