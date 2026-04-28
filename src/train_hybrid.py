import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'prprocessedmental_health_diagnosis_treatment_.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'outputs', 'models')

# Use all 14 features for comprehensive demographic modeling
FEATURES = [
    'Age', 'Gender', 'Diagnosis', 'Symptom Severity (1-10)', 'Mood Score (1-10)',
    'Sleep Quality (1-10)', 'Physical Activity (hrs/week)', 'Medication',
    'Therapy Type', 'Treatment Duration (weeks)', 'Stress Level (1-10)',
    'Treatment Progress (1-10)', 'AI-Detected Emotional State'
]

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH).dropna(subset=FEATURES + ['Outcome'])

le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

le_state = LabelEncoder()
df['AI-Detected Emotional State'] = le_state.fit_transform(df['AI-Detected Emotional State'])

le_diagnosis = LabelEncoder()
df['Diagnosis'] = le_diagnosis.fit_transform(df['Diagnosis'])

le_medication = LabelEncoder()
df['Medication'] = le_medication.fit_transform(df['Medication'])

le_therapy = LabelEncoder()
df['Therapy Type'] = le_therapy.fit_transform(df['Therapy Type'])

le_outcome = LabelEncoder()
df['Outcome'] = le_outcome.fit_transform(df['Outcome'])

X = df[FEATURES]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test), zero_division=0))

joblib.dump(model, os.path.join(MODEL_DIR, 'hybrid_rf.pkl'))
joblib.dump(le_gender, os.path.join(MODEL_DIR, 'le_gender.pkl'))
joblib.dump(le_state, os.path.join(MODEL_DIR, 'le_state.pkl'))
joblib.dump(le_diagnosis, os.path.join(MODEL_DIR, 'le_diagnosis.pkl'))
joblib.dump(le_medication, os.path.join(MODEL_DIR, 'le_medication.pkl'))
joblib.dump(le_therapy, os.path.join(MODEL_DIR, 'le_therapy.pkl'))
joblib.dump(le_outcome, os.path.join(MODEL_DIR, 'le_outcome.pkl'))

print("✅ Hybrid model trained and saved.")
