# src/recommend.py

import os
import joblib

from preprocess import clean_text
from shared_features import build_features

# -----------------------------
# MODEL PATH
# -----------------------------
MODEL_PATH = "../outputs/models/therapy_chat_model.pkl"

# -----------------------------
# LOAD MODEL
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Train your model first.")

model = joblib.load(MODEL_PATH)

tfidf = joblib.load("tfidf.pkl")
bow = joblib.load("bow.pkl")

print("Model loaded successfully.")

# -----------------------------
# EXPLANATION ENGINE
# -----------------------------
def explain(pred):
    return {
        "CBT": "Detected depressive or negative thought patterns.",
        "DBT": "Detected emotional instability or distress signals.",
        "Mindfulness-Based Therapy": "Detected anxiety/stress-related patterns.",
        "No Therapy Needed": "No strong mental health risk detected."
    }.get(pred, "No explanation available.")

# -----------------------------
# THERAPY PREDICTION
# -----------------------------
def recommend_therapy(text):
    cleaned = clean_text(text)

    features = build_features([cleaned], tfidf, bow)
    prediction = model.predict(features)[0]

    print("\n==============================")
    print("INPUT:", text)
    print("THERAPY:", prediction)
    print("EXPLANATION:", explain(prediction))
    print("==============================\n")

# -----------------------------
# CHAT LOOP
# -----------------------------
print("\nTherapy Recommender Chatbot")
print("Type 'exit' to quit\n")

while True:
    user_input = input("User: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    recommend_therapy(user_input)