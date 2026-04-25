import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import clean_text

# 1. LOAD DATA
print("📂 Loading Dataset...")
data = pd.read_csv("data/cleaned_suicide_detection.csv")

# 2. PREPROCESSING
print("🧹 Cleaning text data (Preprocessing)...")
data['text'] = data['text'].apply(clean_text)

# 3. SPLIT DATA
# We split so we can test the model on "unseen" data
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['class'], test_size=0.2, random_state=42)

# 4. HYBRID FEATURE ENGINEERING
print("🧪 Extracting Bigram Features (TF-IDF)...")
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=25000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 5. SOFT VOTING ENSEMBLE (SVEM)
lr = LogisticRegression(class_weight='balanced', max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

svem = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf)], 
    voting='soft',
    weights=[1, 2]
)

print("🚀 Training SVEM Model (This may take a moment)...")
svem.fit(X_train_tfidf, y_train)

# 6. EVALUATION (The "Proper Procedure")
print("\n" + "="*30)
print("📊 MODEL PERFORMANCE REPORT")
print("="*30)
y_pred = svem.predict(X_test_tfidf)
report = classification_report(y_test, y_pred)
print(report)

# 7. SAVE ASSETS
joblib.dump(svem, "outputs/models/model.pkl")
joblib.dump(tfidf, "outputs/models/tfidf.pkl")
# Save the report text to show in the frontend later
with open("outputs/models/model_report.txt", "w") as f:
    f.write(report)

print("\n✅ Success: Enhanced Model and Metrics saved.")