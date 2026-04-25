# src/lime_eval.py

import joblib
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer

from preprocess import clean_text
from shared_features import build_features

# -----------------------------
# LOAD MODELS
# -----------------------------
svm_model = joblib.load('../outputs/models/svem.pkl')
tfidf = joblib.load('../outputs/models/tfidf.pkl')
bow = joblib.load('../outputs/models/bow.pkl')

# -----------------------------
# LIME PREDICTION FUNCTION
# -----------------------------
def predict_proba_for_lime(texts):
    cleaned_texts = [clean_text(t) for t in texts]

    features = build_features(cleaned_texts, tfidf, bow)
    return svm_model.predict_proba(features)

# -----------------------------
# LIME EXPLAINER
# -----------------------------
explainer = LimeTextExplainer(class_names=['Non-Suicide', 'Suicide'])

# -----------------------------
# SAMPLE INPUTS
# -----------------------------
samples = [
    "I am so tired of everything I want to end it all tonight",
    "I feel sad but I know tomorrow will be better",
    "I have been making plans to say goodbye to everyone",
    "Today was okay I went for a walk in the park",
    "I cannot stop crying and I see no way out of this pain",
]

# -----------------------------
# GENERATE EXPLANATIONS
# -----------------------------
for i, text in enumerate(samples):

    exp = explainer.explain_instance(
        text,
        predict_proba_for_lime,
        num_features=10
    )

    fig = exp.as_pyplot_figure()
    plt.title(f"LIME Explanation - Sample {i + 1}")
    plt.tight_layout()
    plt.savefig(f"../outputs/figures/lime_sample_{i + 1}.png", dpi=150)
    plt.close()

    predicted_label = ['Non-Suicide', 'Suicide'][exp.predict_proba.argmax()]
    print(f"Sample {i + 1} done — predicted: {predicted_label}")

print("All LIME plots saved to outputs/figures/")