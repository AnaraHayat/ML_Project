# src/fw9_multilingual.py

import pandas as pd
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# MODEL (correct for embeddings)
# -----------------------------
MODEL_NAME = "bert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------------
# BERT EMBEDDINGS (FIXED)
# -----------------------------
def get_bert_embedding(texts, batch_size=16):
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = model(**enc)

        # Mean pooling (better than CLS-only)
        token_embeddings = output.last_hidden_state
        attention_mask = enc["attention_mask"]

        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)

        mean_pooled = summed / counts

        embeddings.append(mean_pooled.cpu().numpy())

        print(f"Embedded batch {i // batch_size + 1}")

    return np.vstack(embeddings)

# -----------------------------
# LOAD DATA
# -----------------------------
df_en = pd.read_csv('../data/Suicide_Detection.csv').dropna().sample(5000, random_state=42)
df_en['label'] = (df_en['class'] == 'suicide').astype(int)

df_ur = pd.read_csv('../data/urdu_translated_150.csv').dropna()
df_ur['label'] = (df_ur['label'] == 'suicide').astype(int)

# -----------------------------
# EMBEDDINGS
# -----------------------------
print("Getting English embeddings...")
X_en = get_bert_embedding(df_en['text'].tolist())
y_en = df_en['label'].values

print("Getting Urdu embeddings...")
X_ur = get_bert_embedding(df_ur['text'].tolist())
y_ur = df_ur['label'].values

# -----------------------------
# CLASSIFIER
# -----------------------------
clf = LogisticRegression(max_iter=1000)
clf.fit(X_en, y_en)

# -----------------------------
# PREDICTION
# -----------------------------
y_pred = clf.predict(X_ur)

# -----------------------------
# RESULTS
# -----------------------------
print("\nMultilingual BERT (Zero-shot Transfer) Results:")
print("Accuracy:", accuracy_score(y_ur, y_pred))

print(classification_report(
    y_ur,
    y_pred,
    target_names=["Non-Suicide", "Suicide"]
))