# 🎬 Movie Review Sentiment Analyzer

A machine learning web application that predicts whether a movie review expresses **positive or negative sentiment**.

The project uses a custom **PyTorch RNN**, **TF-IDF vectorization**, and **NLP preprocessing**, with **Streamlit** for deployment.

## 🚀 Features

- Text preprocessing using NLP techniques
- URL, punctuation, and HTML removal
- Stopword removal
- Porter stemming
- TF-IDF feature extraction
- Custom RNN implemented using PyTorch
- Positive/negative sentiment prediction
- Prediction confidence score
- Interactive Streamlit web interface

## 🧠 Machine Learning Pipeline

```text
User Review
     ↓
Lowercase
     ↓
Remove URLs
     ↓
Remove HTML
     ↓
Remove Punctuation
     ↓
Remove Stopwords
     ↓
Porter Stemming
     ↓
TF-IDF Vectorization
     ↓
PyTorch RNN
     ↓
Sigmoid
     ↓
Positive / Negative

🛠️ Technologies Used
Python
PyTorch
Scikit-learn
NLTK
Pandas
Joblib
Streamlit

sentiment-analyser/
│
├── app.py
├── requirements.txt
│
├── models/
│   ├── sentiment_rnn.pth
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
│
├── notebook/
│   └── SENTIMENT_ANALYSER.ipynb
│
└── README.md

.

📌 Model

The trained model is a custom recurrent neural network built using PyTorch.

The model receives a 5000-dimensional TF-IDF representation of the processed review and produces a binary sentiment prediction.

🔮 Future Improvements
Improve model architecture using LSTM/GRU
Add probability visualization
Experiment with word embeddings
Improve preprocessing pipeline
Deploy the application publicly
Compare RNN performance with traditional ML models
👨‍💻 Author

Laik Ahmad

B.Tech CSE (Data Science)
```
