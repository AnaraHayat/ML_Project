import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Load your demographic data
data_path = r"C:\Users\saraa\OneDrive\Desktop\ML_project\ML_project\data\prprocessedmental_health_diagnosis_treatment_.csv"
df = pd.read_csv(data_path)

# 2. Convert Gender and AI-State into numbers so the model understands
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

le_state = LabelEncoder()
df['AI-Detected Emotional State'] = le_state.fit_transform(df['AI-Detected Emotional State'])

# 3. Features: Age, Gender, Severity, Mood, Sleep, Stress, Emotional State
X = df[['Age', 'Gender', 'Symptom Severity (1-10)', 'Mood Score (1-10)', 
        'Sleep Quality (1-10)', 'Stress Level (1-10)', 'AI-Detected Emotional State']]
y = df['Outcome'] # Target: Improved, Deteriorated, etc.

# 4. Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 5. Save the final model and encoders
joblib.dump(model, "outputs/models/hybrid_rf.pkl")
joblib.dump(le_gender, "outputs/models/le_gender.pkl")
joblib.dump(le_state, "outputs/models/le_state.pkl")

print("✅ Hybrid model trained and saved in outputs/models/")