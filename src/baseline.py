import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import clean_text

# ========================================
# MULTILINGUAL SUICIDE IDEATION DETECTOR
# ========================================
# This model uses TF-IDF features to maintain fast training/inference
# while supporting multilingual text through robust preprocessing
# (English, Urdu, Roman Urdu support via preprocess.clean_text)

# ========================================
# 1. LOAD DATA
# ========================================
print("\n" + "="*50)
print("PHASE 1: LOAD DATASET (Multilingual)")
print("="*50)
data = pd.read_csv("../data/cleaned_suicide_detection.csv")
print(f"[OK] Loaded {len(data)} samples from cleaned_suicide_detection.csv")

# Optimize: Sample for practical training (30K is large enough for good model quality)
SAMPLE_SIZE = 30000
data = data.sample(n=min(len(data), SAMPLE_SIZE), random_state=42)
print(f"[OK] Using sample of {len(data)} for practical training time")

# ========================================
# 2. PREPROCESSING (Multilingual Support)
# ========================================
print("\nPHASE 2: TEXT PREPROCESSING (Multilingual)")
print("="*50)
print("Cleaning text data with language auto-detection...")
print("  - English support: NLTK tokenization, lemmatization")
print("  - Urdu/Roman Urdu support: Character-aware preprocessing")
data['text'] = data['text'].apply(lambda x: clean_text(x, lang=None))
print("[OK] Text preprocessing complete")

# ========================================
# 3. SPLIT DATA
# ========================================
print("\nPHASE 3: TRAIN/TEST SPLIT (80/20)")
print("="*50)
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], 
    data['class'], 
    test_size=0.2, 
    random_state=42
)
print(f"[OK] Training samples: {len(X_train)}")
print(f"[OK] Test samples:     {len(X_test)}")
print(f"[OK] Class distribution - Train: {y_train.value_counts().to_dict()}")
print(f"[OK] Class distribution - Test:  {y_test.value_counts().to_dict()}")

# ========================================
# 4. FEATURE EXTRACTION (TF-IDF)
# ========================================
print("\nPHASE 4: FEATURE EXTRACTION")
print("="*50)
print("[4a] Extracting TF-IDF features (unigrams + bigrams)...")
print("     - Max features: 25,000")
print("     - N-gram range: (1, 2)")
print("     - Sublinear TF: enabled for better feature scaling")

tfidf = TfidfVectorizer(
    ngram_range=(1, 2), 
    max_features=25000,
    sublinear_tf=True,
    min_df=2,
    max_df=0.95
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print(f"[OK] TF-IDF train shape: {X_train_tfidf.shape}")
print(f"[OK] TF-IDF test shape:  {X_test_tfidf.shape}")

# ========================================
# 5. SOFT VOTING ENSEMBLE (SVEM)
# ========================================
print("\nPHASE 5: MODEL TRAINING")
print("="*50)
print("Training Soft-Voting Ensemble:")
print("  - Estimator 1: Logistic Regression (weight=1)")
print("  - Estimator 2: Random Forest 100 trees (weight=2)")
print("  - Voting: Soft (probability-based averaging)")

lr = LogisticRegression(
    class_weight='balanced', 
    max_iter=1000,
    random_state=42
)
rf = RandomForestClassifier(
    n_estimators=100, 
    n_jobs=-1,
    random_state=42,
    class_weight='balanced'
)

svem = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf)],
    voting='soft',
    weights=[1, 2]
)

print("\nTraining in progress...")
svem.fit(X_train_tfidf, y_train)
print("[OK] SVEM model trained successfully")

# ========================================
# 6. EVALUATION
# ========================================
print("\nPHASE 6: MODEL EVALUATION")
print("="*50)
y_pred = svem.predict(X_test_tfidf)
report = classification_report(
    y_test, 
    y_pred, 
    target_names=["Non-Suicide", "Suicide"],
    digits=4
)

print("\n" + "="*50)
print("CLASSIFICATION REPORT (Test Set)")
print("="*50)
print(report)

# Calculate accuracy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "="*50)
print("SUMMARY METRICS")
print("="*50)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# ========================================
# 7. SAVE ASSETS
# ========================================
print("\nPHASE 7: SAVE MODELS & ARTIFACTS")
print("="*50)

joblib.dump(svem, "../outputs/models/model.pkl")
print("[OK] Saved: model.pkl")

joblib.dump(tfidf, "../outputs/models/tfidf.pkl")
print("[OK] Saved: tfidf.pkl")

with open("../outputs/models/model_report.txt", "w") as f:
    f.write(report)
    f.write("\n\nSUMMARY METRICS\n")
    f.write("="*50 + "\n")
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1-Score:  {f1:.4f}\n")
print("[OK] Saved: model_report.txt")

with open("../outputs/models/training_config.txt", "w") as f:
    f.write("MULTILINGUAL SUICIDE IDEATION DETECTION MODEL\n")
    f.write("="*70 + "\n\n")
    f.write("MODEL ARCHITECTURE:\n")
    f.write("-" * 70 + "\n")
    f.write("Feature Engineering:\n")
    f.write("  - TF-IDF Vectorization (unigrams + bigrams)\n")
    f.write("  - Max features: 25,000\n")
    f.write("  - Sublinear TF scaling enabled\n")
    f.write("  - Min document frequency: 2\n")
    f.write("  - Max document frequency: 0.95\n\n")
    f.write("Ensemble Model:\n")
    f.write("  - Voting Classifier (Soft voting)\n")
    f.write("  - Estimator 1: Logistic Regression (weight=1)\n")
    f.write("  - Estimator 2: Random Forest 100 trees (weight=2)\n")
    f.write("  - Prediction: Average of probabilities\n\n")
    f.write("MULTILINGUAL SUPPORT:\n")
    f.write("-" * 70 + "\n")
    f.write("  - English: Full NLTK preprocessing support\n")
    f.write("  - Urdu: Character-aware tokenization\n")
    f.write("  - Roman Urdu: Latin script Urdu support\n")
    f.write("  - Language Detection: Automatic via Unicode ranges\n\n")
    f.write("WEIGHTING STRATEGY (Pipeline):\n")
    f.write("-" * 70 + "\n")
    f.write("  - Text Model (TF-IDF + SVEM): 60% of combined score\n")
    f.write("  - Demographic Model (Random Forest): 40% of combined score\n\n")
    f.write("DATASET STATISTICS:\n")
    f.write("-" * 70 + "\n")
    f.write(f"  - Total samples: {len(data)}\n")
    f.write(f"  - Training samples: {len(X_train)}\n")
    f.write(f"  - Test samples: {len(X_test)}\n")
    f.write(f"  - Class distribution (train): {y_train.value_counts().to_dict()}\n")
    f.write(f"  - Class distribution (test): {y_test.value_counts().to_dict()}\n\n")
    f.write("PERFORMANCE METRICS:\n")
    f.write("-" * 70 + "\n")
    f.write(f"  - Accuracy:  {accuracy:.4f}\n")
    f.write(f"  - Precision: {precision:.4f}\n")
    f.write(f"  - Recall:    {recall:.4f}\n")
    f.write(f"  - F1-Score:  {f1:.4f}\n\n")
    f.write("FUTURE ENHANCEMENT:\n")
    f.write("-" * 70 + "\n")
    f.write("  BERT Multilingual Embeddings can be integrated for improved\n")
    f.write("  semantic understanding (requires GPU for practical training time).\n")

print("[OK] Saved: training_config.txt")

print("\n" + "="*50)
print("[COMPLETE] TRAINING COMPLETE")
print("="*50)
print("\nModels are ready for inference in pipeline.py")
print("Weighting: 60% Text (TF-IDF) + 40% Demographics")
print("Languages: English, Urdu, Roman Urdu (supported)")
