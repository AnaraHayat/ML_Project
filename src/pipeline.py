# src/pipeline.py

import joblib
from preprocess import clean_text
from shared_features import build_features

from recommend import get_recommendation, recommend_for_patient


class CrisisDetectionPipeline:

    def __init__(self):
        self.svem = joblib.load('../outputs/models/svem.pkl')
        self.tfidf = joblib.load('../outputs/models/tfidf.pkl')
        self.bow = joblib.load('../outputs/models/bow.pkl')
        self.fw1_model = joblib.load('../outputs/models/fw1_model.pkl')

        print("Pipeline loaded.")

    # -----------------------------
    # FEATURE PIPELINE (FIXED)
    # -----------------------------
    def _extract_features(self, text):
        cleaned = clean_text(text)
        return build_features([cleaned], self.tfidf, self.bow)

    # -----------------------------
    # MAIN PREDICTION
    # -----------------------------
    def predict(self, text,
                age=None, gender=None, diagnosis=None,
                symptom_severity=None, mood_score=None,
                sleep_quality=None, physical_activity=None,
                stress_level=None,
                treatment_progress=None,
                adherence=None):

        feats = self._extract_features(text)

        # -------------------------
        # Model 1: Binary risk
        # -------------------------
        binary_pred = self.svem.predict(feats)[0]
        risk_score = float(self.svem.predict_proba(feats)[0][1])

        binary_label = "Suicide" if binary_pred == 1 else "Non-Suicide"

        # -------------------------
        # Model 2: Risk level
        # -------------------------
        risk_level = self.fw1_model.predict(feats)[0]

        rec = get_recommendation(risk_level)

        # -------------------------
        # Derived psychological signals
        # -------------------------
        derived_severity = round(risk_score * 10, 1)

        risk_to_stress = {
            "Low": 3,
            "Medium": 5,
            "High": 7,
            "Critical": 9
        }

        risk_to_mood = {
            "Low": 7,
            "Medium": 5,
            "High": 3,
            "Critical": 1
        }

        derived_stress = risk_to_stress.get(risk_level, 5)
        derived_mood = risk_to_mood.get(risk_level, 5)

        # -------------------------
        # Patient-level model (optional)
        # -------------------------
        patient_rec = None

        if age is not None and gender is not None and diagnosis is not None:
            try:
                patient_rec = recommend_for_patient(
                    age=age,
                    gender=gender,
                    diagnosis=diagnosis,
                    symptom_severity=symptom_severity or derived_severity,
                    mood_score=mood_score or derived_mood,
                    stress_level=stress_level or derived_stress,
                    sleep_quality=sleep_quality or 5,
                    physical_activity=physical_activity or 3,
                    treatment_progress=treatment_progress or 5,
                    adherence=adherence or 70
                )
            except Exception as e:
                patient_rec = {"error": str(e)}

        # -------------------------
        # OUTPUT
        # -------------------------
        output = {
            "input_text": text,
            "binary_class": binary_label,
            "risk_score": round(risk_score, 4),
            "risk_level": risk_level,

            "action": rec["action"],
            "message": rec["message"],
            "resource": rec["resource"],

            "derived_severity": derived_severity,
            "derived_stress": derived_stress,
            "derived_mood": derived_mood,
        }

        if patient_rec and "error" not in patient_rec:
            output["recommended_therapy"] = patient_rec.get("recommended_therapy", "N/A")
            output["predicted_outcome"] = patient_rec.get("predicted_outcome", "N/A")

        return output


# -----------------------------
# PRINT FUNCTION
# -----------------------------
def print_result(result):
    print("=" * 60)
    print(f"Text         : {result['input_text'][:55]}")
    print(f"Binary Class : {result['binary_class']}")
    print(f"Risk Score   : {result['risk_score']}")
    print(f"Risk Level   : {result['risk_level']}")
    print(f"Action       : {result['action']}")
    print(f"Message      : {result['message']}")
    print(f"Resource     : {result['resource']}")

    print("--- Derived Signals ---")
    print(f"Severity     : {result['derived_severity']}/10")
    print(f"Stress       : {result['derived_stress']}/10")
    print(f"Mood         : {result['derived_mood']}/10")

    if "recommended_therapy" in result:
        print(f"Therapy      : {result['recommended_therapy']}")
        print(f"Outcome      : {result['predicted_outcome']}")

    print("=" * 60)


# -----------------------------
# TESTING
# -----------------------------
if __name__ == "__main__":
    pipeline = CrisisDetectionPipeline()

    print("\n--- Test 1 ---")
    r1 = pipeline.predict("I feel like ending everything tonight")
    print_result(r1)

    print("\n--- Test 2 ---")
    r2 = pipeline.predict(
        text="I feel like ending everything tonight",
        age=28,
        gender="Female",
        diagnosis="Major Depressive Disorder"
    )
    print_result(r2)

    print("\n--- Test 3 ---")
    r3 = pipeline.predict(
        text="I feel like ending everything tonight",
        age=28,
        gender="Female",
        diagnosis="Major Depressive Disorder",
        symptom_severity=9,
        mood_score=2,
        stress_level=9
    )
    print_result(r3)

    print("\n--- Test 4 ---")
    r4 = pipeline.predict(
        text="Went for a walk today feeling okay",
        age=30,
        gender="Male",
        diagnosis="Generalized Anxiety"
    )
    print_result(r4)