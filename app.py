
import re
import joblib
import torch
import torch.nn as nn
import streamlit as st
import nltk



from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# =========================================================
# 1. RNN MODEL
# =========================================================

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        h0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size
        )

        out, _ = self.rnn(x, h0)

        out = self.fc(out[:, -1, :])

        return out


# =========================================================
# 2. LOAD SAVED FILES
# =========================================================

@st.cache_resource
def load_model():

    # Load TF-IDF vectorizer
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")

    # Load label encoder
    label_encoder = joblib.load("models/label_encoder.pkl")

    # Create model
    model = RNN(input_size=5000)

    # Load trained weights
    model.load_state_dict(
        torch.load(
            "models/sentiment_rnn.pth",
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model, tfidf, label_encoder


model, tfidf, label_encoder = load_model()


# =========================================================
# 3. TEXT PREPROCESSING
# =========================================================

def preprocess_text(text):

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+", "", text)

    # 3. Remove punctuation and additional symbols
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)

    # 4. Remove HTML
    text = re.sub(r"<.*?>", "", text)

    # 5. Remove stopwords
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")

    for word in tokens:
        if word in stop_words:
            text = text.replace(word, "")

    # 6. Stemming
    ps = PorterStemmer()
    stemmed_words = []

    tokens = word_tokenize(text)

    for token in tokens:
        stemmed_token = ps.stem(token)
        stemmed_words.append(stemmed_token)

    return " ".join(stemmed_words)


# =========================================================
# 4. STREAMLIT UI
# =========================================================

st.title("🎬 Movie Review Sentiment Analyzer")

st.write(
    "Enter a movie review and the trained RNN model "
    "will predict whether the sentiment is positive or negative."
)


review = st.text_area(
    "Enter your movie review:",
    placeholder="Example: This movie was absolutely amazing!"
)


# =========================================================
# 5. PREDICTION
# =========================================================

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        # Step 1: Apply same preprocessing used during training
        processed_review = preprocess_text(review)

        # Step 2: TF-IDF transformation
        vector = tfidf.transform([processed_review])

        # Step 3: Convert sparse matrix to dense array
        vector = vector.toarray()

        # Step 4: Convert to PyTorch tensor
        tensor = torch.tensor(
            vector,
            dtype=torch.float32
        )

        # Step 5: Add sequence dimension
        # Shape: (1, 5000) → (1, 1, 5000)
        tensor = tensor.unsqueeze(1)

        # Step 6: Model prediction
        with torch.no_grad():

            output = model(tensor)

            probability = torch.sigmoid(output)

            prediction = (probability >= 0.5).int().item()

        # Step 7: Convert encoded prediction back to label
        sentiment = label_encoder.inverse_transform([prediction])[0]

        # =================================================
        # 6. DISPLAY RESULT
        # =================================================

        st.subheader("Prediction")

        if sentiment.lower() == "positive":
            st.success(f"😊 {sentiment}")

        else:
            st.error(f"😞 {sentiment}")

        st.write(
            f"Confidence: {probability.item():.2%}"
        )
