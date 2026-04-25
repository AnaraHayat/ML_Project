import streamlit as st
import pandas as pd
import joblib
from preprocess import clean_text

# --- 1. UI CONFIGURATION ---
st.set_page_config(page_title="MindGuard Pro AI", layout="wide")

# Custom CSS to ensure high visibility of text on black background
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #FFFFFF; }
    h1, h2, h3, label, p, .stMetric, .stMarkdown { color: #FFFFFF !important; }
    .stTextArea textarea { background-color: #1a1a1a; color: white; border: 1px solid #444; }
    .stButton>button { background-color: #cc0000; color: white; font-weight: bold; }
    .report-box { background-color: #111; padding: 15px; border-radius: 10px; border: 1px solid #333; font-family: monospace; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = joblib.load("outputs/models/model.pkl")
    tfidf = joblib.load("outputs/models/tfidf.pkl")
    try:
        with open("outputs/models/model_report.txt", "r") as f:
            report = f.read()
    except:
        report = "Run baseline.py to generate report."
    return model, tfidf, report

classifier, vectorizer, perf_report = load_assets()

# --- 3. TABS FOR ORGANIZATION ---
tab1, tab2 = st.tabs(["🔍 Patient Assessment", "📊 Model Intelligence"])

with tab1:
    st.title("🧠 MindGuard: Suicide Ideation Detection")
    user_story = st.text_area("Share your story/thoughts (English, Urdu, or Roman Urdu):", height=150)

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 1, 100, 25)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        height = st.number_input("Height (cm)", 100, 250, 170)
    with c2:
        weight = st.number_input("Weight (kg)", 30, 200, 70)
        sleep = st.slider("Sleep Quality (Hours)", 0, 12, 7)
        lifestyle = st.selectbox("Lifestyle", ["Sedentary", "Active", "Athlete"])

    if st.button("RUN ANALYSIS"):
        cleaned = clean_text(user_story)
        vec = vectorizer.transform([cleaned])
        prob = classifier.predict_proba(vec)[0][1]
        
        st.divider()
        
        # Result Logic
        if prob > 0.4:
            st.error(f"### DETECTED: Suicide Ideation Risk")
            st.write("🆘 **Recommendation:** Immediate Clinical Intervention Required.")
        else:
            st.success(f"### DETECTED: Non-Suicide / Stable")
            st.write("✅ **Recommendation:** Routine mental wellness checkup suggested.")
            
        st.metric("AI Confidence Level", f"{prob*100:.1f}%")
        
        # System Metric Graph
        chart_data = pd.DataFrame({
            "Metric": ["Risk Probability", "Sleep Health", "Activity Level"],
            "Value": [prob * 10, sleep, 8 if lifestyle == "Active" else 2]
        })
        st.bar_chart(chart_data.set_index("Metric"))

with tab2:
    st.header("📈 Training Procedure & Metrics")
    st.write("This section shows the mathematical 'proof' of the system's performance.")
    
    st.subheader("Classification Report (Precision/Recall/F1)")
    st.markdown(f"```\n{perf_report}\n```")
    
    st.subheader("Methodology Explanation")
    st.write("""
    1. **Preprocessing:** NLTK Tokenization and Lemmatization.
    2. **Features:** Hybrid Unigram & Bigram TF-IDF (25,000 features).
    3. **Ensemble:** Soft Voting (Logistic Regression + Random Forest).
    4. **Threshold:** 0.4 Sensitivity Adjustment for Clinical Safety.
    """)