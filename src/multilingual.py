# src/multilingual.py

import pandas as pd
import torch
import numpy as np
import joblib

from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

# -----------------------------
# MODEL
# -----------------------------
MODEL_NAME = "bert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Using device: {device}")

# -----------------------------
# BERT EMBEDDINGS
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

        token_embeddings = output.last_hidden_state
        attention_mask = enc["attention_mask"]

        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / counts

        embeddings.append(mean_pooled.cpu().numpy())
        print(f"  Embedded batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")

    return np.vstack(embeddings)

# -----------------------------
# LOAD DATASETS
# -----------------------------
print("\nLoading datasets...")

# English — for training
df_en = pd.read_csv('../data/cleaned_suicide_detection.csv').dropna().sample(5000, random_state=42)
df_en['label'] = df_en['class']  # class column already contains 0/1 integers

# Roman Urdu
df_roman = pd.read_csv('../data/romanurdu_translated_100.csv').dropna()
df_roman['label'] = (df_roman['label'] == 'suicide').astype(int)
df_roman['lang'] = 'roman_urdu'

# Urdu — Handle missing file gracefully
try:
    df_ur = pd.read_csv('../data/urdu_translated_150.csv').dropna()
    df_ur['label'] = (df_ur['label'] == 'suicide').astype(int)
    df_ur['lang'] = 'urdu'
    urdu_loaded = True
except FileNotFoundError:
    print("⚠️  WARNING: urdu_translated_150.csv not found. Using only Roman Urdu data.")
    df_ur = None
    urdu_loaded = False

print(f"English samples   : {len(df_en)}")
print(f"Roman Urdu samples: {len(df_roman)}")
if urdu_loaded:
    print(f"Urdu samples      : {len(df_ur)}")
else:
    print(f"Urdu samples      : Not available")

# -----------------------------
# MERGE URDU + ROMAN URDU (or just Roman Urdu if Urdu unavailable)
# -----------------------------
def upsample(df, target):
    if len(df) < target:
        return resample(df, replace=True, n_samples=target, random_state=42)
    return df

if urdu_loaded:
    # Balance both to same size before merging
    target_size = max(len(df_roman), len(df_ur))
    df_roman_bal = upsample(df_roman, target_size)
    df_ur_bal    = upsample(df_ur, target_size)
    df_merged = pd.concat([df_roman_bal, df_ur_bal], ignore_index=True)
else:
    # Only use Roman Urdu
    df_merged = df_roman.copy()

df_merged  = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

print(f"\nMerged test dataset size: {len(df_merged)}")
print(f"Label distribution:\n{df_merged['label'].value_counts()}")

# -----------------------------
# EMBEDDINGS
# -----------------------------
print("\nGetting English embeddings (train)...")
X_en = get_bert_embedding(df_en['text'].tolist())
y_en = df_en['label'].values

print("\nGetting test dataset embeddings (Roman Urdu" + (" + Urdu" if urdu_loaded else "") + ")...")
X_merged = get_bert_embedding(df_merged['text'].tolist())
y_merged  = df_merged['label'].values
langs     = df_merged['lang'].values

# -----------------------------
# TRAIN ON ENGLISH
# -----------------------------
print("\nTraining classifier on English dataset...")
clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_en, y_en)

# Save the model
joblib.dump(clf, '../outputs/models/bert_lr.pkl')

# Save the model name so pipeline knows which BERT to load
with open('../outputs/models/bert_multilingual.txt', 'w') as f:
    f.write(MODEL_NAME)

# -----------------------------
# PREDICT ON MERGED URDU + ROMAN URDU
# -----------------------------
y_pred = clf.predict(X_merged)

# -----------------------------
# OVERALL RESULTS
# -----------------------------
result_title = "===== OVERALL RESULTS (Roman Urdu" + (" + Urdu" if urdu_loaded else "") + ") ====="
print("\n" + result_title)
print(f"Accuracy: {accuracy_score(y_merged, y_pred):.4f}")
print(classification_report(y_merged, y_pred, target_names=["Non-Suicide", "Suicide"]))

# -----------------------------
# PER-LANGUAGE RESULTS
# -----------------------------
print("\n===== PER-LANGUAGE RESULTS =====")

langs_to_check = ['roman_urdu']
if urdu_loaded:
    langs_to_check.append('urdu')

for lang in langs_to_check:
    mask = langs == lang
    if mask.sum() == 0:
        print(f"\n--- {lang.upper()} ---")
        print(f"No samples available for this language")
        continue
    
    y_true_lang = y_merged[mask]
    y_pred_lang = y_pred[mask]

    print(f"\n--- {lang.upper()} ---")
    print(f"Samples : {mask.sum()}")
    print(f"Accuracy: {accuracy_score(y_true_lang, y_pred_lang):.4f}")
    print(classification_report(
        y_true_lang, y_pred_lang,
        target_names=["Non-Suicide", "Suicide"],
        zero_division=0
    ))