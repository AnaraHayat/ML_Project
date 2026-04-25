# SVEM Model (RF + LR + SGDC with soft voting)

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report

from preprocess import clean_text
from features import get_hybrid_features


# -----------------------------
# 1. LOAD DATA
# -----------------------------
print('Loading data...')

df = pd.read_csv('../data/Suicide_Detection.csv')

print(f'Loaded {len(df)} rows')

# Drop nulls
df = df.dropna(subset=['text', 'class'])


# -----------------------------
# 2. CLEAN TEXT
# -----------------------------
print('\nCleaning text...')
df['clean'] = df['text'].apply(clean_text)
print('Text cleaning done')


# -----------------------------
# 3. HANDLE LABELS (ROBUST VERSION)
# -----------------------------
print("\nProcessing labels...")

df['class'] = df['class'].astype(str).str.lower().str.strip()

df['label'] = df['class'].apply(
    lambda x: 1 if ('suicide' in x and 'non' not in x) else 0
)

print("\nDataset distribution:")
print(df['label'].value_counts())


# Safety check
if df['label'].nunique() < 2:
    raise ValueError("Dataset has only ONE class. Fix dataset first.")


# -----------------------------
# 4. TRAIN-TEST SPLIT
# -----------------------------
print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    df['clean'],
    df['label'],
    test_size=0.30,
    random_state=42,
    stratify=df['label']
)

print("\nTrain distribution:")
print(y_train.value_counts())

print("\nTest distribution:")
print(y_test.value_counts())


# -----------------------------
# 5. FEATURE EXTRACTION
# -----------------------------
print('\nExtracting hybrid features...')

X_tr, X_te, tfidf, bow = get_hybrid_features(X_train, X_test)

print('Feature extraction complete')

print("Feature shape (train):", X_tr.shape)
print("Feature shape (test):", X_te.shape)


# -----------------------------
# 6. MODELS
# -----------------------------
print("\nInitializing models...")

lr = LogisticRegression(
    solver='liblinear',
    C=1.0,
    max_iter=1000,
    random_state=42
)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=80,
    n_jobs=-1,
    random_state=42
)

sgdc = SGDClassifier(
    loss='modified_huber',
    max_iter=100,
    tol=1e-3,
    random_state=42
)


# SVEM (Soft Voting Ensemble)
svem = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('rf', rf),
        ('sgdc', sgdc)
    ],
    voting='soft'
)


# -----------------------------
# 7. TRAIN MODEL
# -----------------------------
print('\nTraining SVEM model...')

svem.fit(X_tr, y_train)

print('Training completed')


# -----------------------------
# 8. EVALUATION
# -----------------------------
print('\nEvaluating model...')

y_pred = svem.predict(X_te)

accuracy = accuracy_score(y_test, y_pred)

print('\nAccuracy:', accuracy)

print('\nClassification Report:\n')

print(classification_report(
    y_test,
    y_pred,
    target_names=['Non-Suicide', 'Suicide']
))


# -----------------------------
# 9. SAVE MODEL
# -----------------------------
print('\nSaving model...')

joblib.dump(svem, '../outputs/models/svem.pkl')

joblib.dump(tfidf, '../outputs/models/tfidf.pkl')

joblib.dump(bow, '../outputs/models/bow.pkl')

joblib.dump(
    (X_train, X_test, y_train, y_test),
    '../outputs/models/splits.pkl'
)

print('Model saved successfully.')