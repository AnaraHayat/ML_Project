# src/fw2_risk_score.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import brier_score_loss

from shared_features import build_features

# -----------------------------
# LOAD MODELS
# -----------------------------
svm_model = joblib.load('../outputs/models/svem.pkl')
X_train, X_test, y_train, y_test = joblib.load('../outputs/models/splits.pkl')
tfidf = joblib.load('../outputs/models/tfidf.pkl')
bow = joblib.load('../outputs/models/bow.pkl')

# -----------------------------
# FEATURE BUILDING (FIXED)
# -----------------------------
X_te = build_features(X_test, tfidf, bow)

# -----------------------------
# RISK SCORE (0 → 1)
# -----------------------------
risk_scores = svm_model.predict_proba(X_te)[:, 1]

# -----------------------------
# SEVERITY MAPPING
# -----------------------------
def score_to_bucket(score):
    if score < 0.2:
        return 'Low'
    elif score < 0.4:
        return 'Medium'
    elif score < 0.7:
        return 'High'
    else:
        return 'Critical'

# -----------------------------
# RESULTS
# -----------------------------
results = pd.DataFrame({
    'text': X_test.values,
    'true_label': y_test.values,
    'risk_score': risk_scores,
    'bucket': [score_to_bucket(s) for s in risk_scores]
})

print(results.head(10))

# -----------------------------
# DISTRIBUTION PLOT
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 5))

for label, grp in results.groupby('true_label'):
    name = 'Suicide' if label == 1 else 'Non-Suicide'
    ax.hist(grp['risk_score'], bins=40, alpha=0.6, label=name)

ax.set_xlabel('Risk Score (0 = safe, 1 = critical)')
ax.set_ylabel('Number of posts')
ax.set_title('Risk Score Distribution')
ax.legend()

plt.tight_layout()
plt.savefig('../outputs/figures/fw2_score_dist.png', dpi=150)
plt.close()

# -----------------------------
# CALIBRATION CHECK
# -----------------------------
brier = brier_score_loss(y_test, risk_scores)
print(f'Brier Score: {brier:.4f}')

fig2, ax2 = plt.subplots(figsize=(6, 6))
CalibrationDisplay.from_predictions(y_test, risk_scores, ax=ax2, n_bins=10)
ax2.set_title('Calibration Curve')

plt.tight_layout()
plt.savefig('../outputs/figures/fw2_calibration.png', dpi=150)
plt.close()

print("Done.")