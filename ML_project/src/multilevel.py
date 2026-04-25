# src/fw1_multilevel.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from preprocess import clean_text
from features import get_hybrid_features


# -----------------------------
# FILE PATH
# -----------------------------
DATA_PATH = '../data/Suicide_Detection.csv'


# -----------------------------
# URGENCY KEYWORDS
# -----------------------------
URGENT_WORDS = [
    'tonight','tomorrow','plan','method','decided','goodbye',
    'no way out','end it','last day','cant go on','want to die',
    'kill myself','rope','pills','jump','overdose'
]


# -----------------------------
# LABEL FUNCTION
# -----------------------------
def assign_4class(row):

    text  = str(row['text']).lower()
    label = str(row['class']).lower().strip()

    # Non-suicide → Low
    if 'non' in label:
        return 'Low'

    matches = sum(1 for w in URGENT_WORDS if w in text)

    if matches >= 2:
        return 'Critical'
    elif matches == 1:
        return 'High'
    else:
        return 'Medium'


# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading dataset...")

df = pd.read_csv(DATA_PATH).dropna(subset=['text','class'])

print(f"Dataset size: {len(df)}")


# -----------------------------
# CLEAN TEXT
# -----------------------------
print("Cleaning text...")

df['clean'] = df['text'].apply(clean_text)


# -----------------------------
# CREATE LABELS
# -----------------------------
print("Creating 4-class labels...")

df['risk_level'] = df.apply(assign_4class, axis=1)

print("\nClass distribution:")
print(df['risk_level'].value_counts())


# -----------------------------
# OPTIONAL SPEED SAMPLING
# (safe stratified version)
# -----------------------------
MAX_ROWS = 40000

if len(df) > MAX_ROWS:

    df = df.groupby('risk_level', group_keys=False)\
           .apply(lambda x: x.sample(
               min(len(x), MAX_ROWS // 4),
               random_state=42
           ))

    print("\nAfter stratified sampling:")
    print(df['risk_level'].value_counts())


# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['clean'],
    df['risk_level'],
    test_size=0.30,
    random_state=42,
    stratify=df['risk_level']
)


# -----------------------------
# FEATURES
# -----------------------------
print("\nExtracting features...")

X_tr, X_te, _, _ = get_hybrid_features(X_train, X_test)

print("Feature shape:", X_tr.shape)


# -----------------------------
# MODELS
# -----------------------------
models = {

    'Logistic Regression':
        LogisticRegression(
            solver='liblinear',
            max_iter=1000
        ),

    'Random Forest':
        RandomForestClassifier(
            n_estimators=200,
            max_depth=80,
            n_jobs=-1,
            random_state=42
        )
}


best_model = None
best_f1 = 0
best_name = None


# -----------------------------
# TRAIN MODELS
# -----------------------------
for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_tr, y_train)

    y_pred = model.predict(X_te)

    print(classification_report(y_test, y_pred))

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    f1 = report['weighted avg']['f1-score']

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_name = name


    # Save confusion matrix

    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_te,
        y_test
    )

    plt.title(f'Confusion Matrix - {name}')

    plt.tight_layout()

    plt.savefig(
        f'../outputs/figures/fw1_cm_{name.replace(" ","_")}.png',
        dpi=150
    )

    plt.close()


# -----------------------------
# SAVE BEST MODEL
# -----------------------------
print(f"\nBest model: {best_name} (F1={best_f1:.4f})")

joblib.dump(
    best_model,
    '../outputs/models/fw1_model.pkl'
)

print("Model saved successfully.")