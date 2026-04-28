import os
import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from preprocess import clean_text, detect_language

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'outputs', 'models')

import pandas as pd

FEATURE_COLS = [
    'Age', 'Gender', 'Diagnosis', 'Symptom Severity (1-10)', 'Mood Score (1-10)',
    'Sleep Quality (1-10)', 'Physical Activity (hrs/week)', 'Medication',
    'Therapy Type', 'Treatment Duration (weeks)', 'Stress Level (1-10)',
    'Treatment Progress (1-10)', 'AI-Detected Emotional State'
]


class CrisisDetectionPipeline:
    def __init__(self):
        # --- TF-IDF + Voting Ensemble (English) ---
        self.text_model = joblib.load(os.path.join(MODEL_DIR, 'model.pkl'))
        self.tfidf      = joblib.load(os.path.join(MODEL_DIR, 'tfidf.pkl'))

        # --- Demographic Random Forest ---
        self.hybrid_model    = joblib.load(os.path.join(MODEL_DIR, 'hybrid_rf.pkl'))
        self.le_gender       = joblib.load(os.path.join(MODEL_DIR, 'le_gender.pkl'))
        self.le_state        = joblib.load(os.path.join(MODEL_DIR, 'le_state.pkl'))
        self.le_diagnosis    = joblib.load(os.path.join(MODEL_DIR, 'le_diagnosis.pkl'))
        self.le_medication   = joblib.load(os.path.join(MODEL_DIR, 'le_medication.pkl'))
        self.le_therapy      = joblib.load(os.path.join(MODEL_DIR, 'le_therapy.pkl'))
        self.le_outcome      = joblib.load(os.path.join(MODEL_DIR, 'le_outcome.pkl'))

        # --- BERT Multilingual (Urdu / Roman Urdu) ---
        self.bert_lr = None
        self.bert_tokenizer = None
        self.bert_model = None
        self._load_bert()

        print("[OK] All models loaded successfully")

    def _load_bert(self):
        """Load BERT classifier — gracefully skip if not trained yet."""
        bert_lr_path = os.path.join(MODEL_DIR, 'bert_lr.pkl')
        bert_name_path = os.path.join(MODEL_DIR, 'bert_multilingual.txt')

        if not os.path.exists(bert_lr_path):
            print("[WARNING] bert_lr.pkl not found — BERT disabled. Run multilingual.py first.")
            return

        try:
            self.bert_lr = joblib.load(bert_lr_path)

            model_name = "bert-base-multilingual-cased"
            if os.path.exists(bert_name_path):
                with open(bert_name_path) as f:
                    model_name = f.read().strip()

            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.bert_model = AutoModel.from_pretrained(model_name).to(self.device)
            self.bert_model.eval()
            print(f"[OK] BERT loaded ({model_name}) on {self.device}")

        except Exception as e:
            print(f"[WARNING] BERT load failed: {e} — falling back to TF-IDF only.")
            self.bert_lr = None

    def _get_bert_embedding(self, text: str) -> np.ndarray:
        """Get mean-pooled BERT embedding for a single text."""
        enc = self.bert_tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.bert_model(**enc)

        token_emb = output.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
        summed = torch.sum(token_emb * mask, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        return (summed / counts).cpu().numpy()

    def _get_text_risk(self, text: str, lang: str) -> float:
        """
        Route text through the right model based on language:
        - English          → TF-IDF + Voting Ensemble
        - Urdu/Roman Urdu  → BERT Multilingual (fallback to TF-IDF if unavailable)
        """
        cleaned = clean_text(text, lang=lang)

        if lang in ('urdu', 'roman_urdu') and self.bert_lr is not None:
            embedding = self._get_bert_embedding(cleaned)
            prob = self.bert_lr.predict_proba(embedding)[0][1]
            print(f"[INFO] Using BERT for {lang} input")
        else:
            tfidf_features = self.tfidf.transform([cleaned])
            prob = self.text_model.predict_proba(tfidf_features)[0][1]
            print(f"[INFO] Using TF-IDF for {lang} input")

        return float(prob)

    def get_recommendation(self, risk_level):
        if risk_level == "Critical":
            return "[CRITICAL] Contact emergency services immediately. You are not alone. National Suicide Prevention Lifeline: 988 (US), or reach out to a trusted person now."
        elif risk_level == "High":
            return "[HIGH RISK] Please reach out to a professional counselor, therapist, or trusted friend today. Consider calling a crisis helpline for immediate support."
        elif risk_level == "Medium":
            return "[MEDIUM RISK] Consider speaking to someone you trust. A professional mental health assessment is recommended."
        else:
            return "[LOW RISK] Consider practicing mindfulness, exercise, or speaking to someone. Maintain healthy coping strategies."

    def predict_full_report(self, text_input, age, gender, diagnosis, sleep, stress,
                            severity, mood_score, emotion, medication, therapy_type,
                            treatment_duration, treatment_progress, physical_activity):

        # ---------------------------
        # Part A: Text Model (language-aware)
        # ---------------------------
        lang = detect_language(text_input)
        text_prob = self._get_text_risk(text_input, lang)

        # ---------------------------
        # Part B: Demographic Model
        # ---------------------------
        try:
            gender_enc    = self.le_gender.transform([gender])[0]
            emotion_enc   = self.le_state.transform([emotion])[0]
            diagnosis_enc = self.le_diagnosis.transform([diagnosis])[0]
            medication_enc = self.le_medication.transform([medication])[0]
            therapy_enc   = self.le_therapy.transform([therapy_type])[0]
        except ValueError as e:
            print(f"[WARNING] Unseen label: {e}")
            gender_enc = emotion_enc = diagnosis_enc = medication_enc = therapy_enc = 0

        demo_features = pd.DataFrame([[
            age, gender_enc, diagnosis_enc, severity, mood_score,
            sleep, physical_activity, medication_enc, therapy_enc,
            treatment_duration, stress, treatment_progress, emotion_enc
        ]], columns=FEATURE_COLS)

        demo_proba    = self.hybrid_model.predict_proba(demo_features)[0]
        demo_risk_idx = int(np.argmax(demo_proba))
        demo_outcome  = str(self.hybrid_model.classes_[demo_risk_idx])
        demo_prob     = float(demo_proba[demo_risk_idx])

        # ---------------------------
        # Part C: Combined Score
        # ---------------------------
        combined_score = (0.6 * text_prob) + (0.4 * demo_prob)

        if combined_score > 0.7:
            risk = "Critical"
        elif combined_score > 0.5:
            risk = "High"
        elif combined_score > 0.3:
            risk = "Medium"
        else:
            risk = "Low"

        return {
            "Risk Level"      : risk,
            "Combined Score"  : float(round(combined_score, 4)),
            "Text Risk Score" : float(round(text_prob, 4)),
            "Clinical Outcome": demo_outcome,
            "Recommendation"  : self.get_recommendation(risk),
            "Language Detected": lang,   # bonus: expose this to frontend
        }