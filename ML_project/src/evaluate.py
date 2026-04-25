import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load saved data
X_train, X_test, y_train, y_test = joblib.load('../outputs/models/splits.pkl')

# Load model
model = joblib.load('../outputs/models/svem.pkl')

# Load features
tfidf = joblib.load('../outputs/models/tfidf.pkl')
bow = joblib.load('../outputs/models/bow.pkl')

from features import get_hybrid_features

# Convert text → features again
X_tr, X_te, _, _ = get_hybrid_features(X_train, X_test)

# Predict
y_pred = model.predict(X_te)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))