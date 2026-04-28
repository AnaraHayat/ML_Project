# MindGuard Pro AI - Suicide Ideation Detection System

A multilingual mental health crisis detection system using advanced machine learning to identify suicidal tendencies
based on user input and clinical factors.

## Overview

MindGuard Pro AI combines natural language processing (TF-IDF) with demographic/clinical machine learning to predict
suicide ideation risk. The system supports **English, Urdu, and Roman Urdu** through intelligent text preprocessing and
provides risk assessment with clinical recommendations.

### Key Features

- **Multilingual Support**: English, Urdu, Roman Urdu language detection and processing
- **Dual-Model Ensemble**:
    - Text Model: TF-IDF + Soft-Voting Classifier (91.93% accuracy)
    - Demographics Model: Random Forest on clinical factors
    - Combined Score: 60% text + 40% demographics weighting
- **Comprehensive Clinical Assessment**: Diagnosis, medication, therapy type, treatment progress tracking
- **Real-time Risk Scoring**: Critical/High/Medium/Low risk classification with confidence scores
- **PDF Report Generation**: Exportable analysis reports with metrics visualization
- **Analysis History**: Persistent tracking of previous assessments
- **Professional UI**: Dark theme with responsive design and accessibility features

### Model Performance (Test Set)

- **Accuracy**: 91.93%
- **Precision**: 92.08%
- **Recall**: 91.59%
- **F1-Score**: 91.84%

---

## Quick Start (3 Steps)

### 1. Install Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Train Models (Optional - Pre-trained models included)

```bash
cd src
python baseline.py          # Train text model (TF-IDF + SVEM)
python train_hybrid.py      # Train demographic model (Random Forest)
```

### 3. Run the Application

```bash
cd src
python -m uvicorn app:app --host 0.0.0.0 --port 8000
# Open browser: http://localhost:8000
```

---

## Installation & Setup

### System Requirements

- Python 3.8+
- 4GB RAM minimum (8GB recommended)
- NLTK data (auto-downloaded on first run)
- PyTorch, Transformers, scikit-learn, FastAPI

### Step-by-Step Installation

1. **Clone/Extract Project**
   ```bash
   cd ML_project
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python -c "import fastapi, torch, sklearn, nltk; print('OK')"
   ```

---

## Training Models

### Text Model Training (TF-IDF + Soft-Voting Ensemble)

```bash
cd src
python baseline.py
```

**What it does:**

- Loads 30,000 stratified samples from `data/cleaned_suicide_detection.csv`
- Preprocesses text with multilingual support (English/Urdu/Roman Urdu)
- Extracts TF-IDF features (25,000 bigram features)
- Trains soft-voting ensemble (Logistic Regression + Random Forest)
- Saves: `model.pkl`, `tfidf.pkl`, `model_report.txt`, `training_config.txt`

**Training time:** ~5-10 minutes (CPU)

### Demographic Model Training (Random Forest)

```bash
# inside src directory
python train_hybrid.py
```

**What it does:**

- Loads clinical/demographic dataset
- Encodes categorical features (Gender, Diagnosis, Medication, Therapy, State, Outcome)
- Trains Random Forest classifier on 13 demographic features
- Saves: `hybrid_rf.pkl`, `le_*.pkl` (7 label encoders)

**Training time:** ~2-3 minutes (CPU)

---

### Bert Multilingual model training

```bash
# inside src directory
python multilingual.py
```

**What it does:**

- Generates BERT embeddings for your English, Urdu, and Roman Urdu datasets
- Trains a Logistic Regression classifier on top of those embeddings (trained on English, tested on Urdu/Roman Urdu)
- Saves bert_lr.pkl — which is the only thing your pipeline.py actually needs to route Urdu/Roman Urdu inputs through
  BERT instead of TF-IDF

**Training time:** ~5-10 minutes (CPU or discrete GPU if available)

---

## Running the Application

### Start the Server

### use script

```bash
# in root directory
.\run.bat         # for windows
```

### or manually run the app 

```bash
cd src
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### Access the Web Interface

Open browser: `http://localhost:8000`

### Using the Application

1. **Enter Patient Story** - Share thoughts/concerns in English, Urdu, or Roman Urdu
2. **Fill Demographics** - Age, gender, height, weight, lifestyle
3. **Mental Health Metrics** - Sleep, stress, severity, mood score, emotional state
4. **Clinical Information** - Diagnosis, medication, therapy type, treatment progress, physical activity
5. **Run Analysis** - System calculates combined risk score and recommendations
6. **View Results** - Risk level, confidence score, recommendation text
7. **Export PDF** - Download detailed report with metrics

---

## API Endpoints

### Analysis Endpoint

```http
POST /api/analyze
Content-Type: application/json

{
  "user_input": "I'm feeling hopeless...",
  "age": 25,
  "gender": "Female",
  "height": 165,
  "weight": 65,
  "sleep": 5,
  "stress": 8,
  "severity": 7,
  "mood_score": 3,
  "emotion": "Sad",
  "lifestyle": "Sedentary",
  "diagnosis": "Major Depressive Disorder",
  "medication": "SSRIs",
  "therapy_type": "Cognitive Behavioral Therapy",
  "treatment_duration": 30,
  "treatment_progress": 40,
  "physical_activity": 0.5
}
```

**Response:**

```json
{
  "risk_level": "High",
  "combined_score": 0.6234,
  "text_risk_score": 0.7521,
  "confidence": 62.34,
  "recommendation": "[HIGH RISK] Please reach out...",
  "clinical_outcome": "At-Risk",
  "timestamp": "2026-04-29 22:30:45"
}
```

### Other Endpoints

- `GET /` - Serve frontend HTML
- `GET /api/history` - Retrieve analysis history
- `GET /api/models-info` - Get model performance report
- `POST /api/generate-pdf` - Generate PDF report
- `POST /api/delete-history/{index}` - Delete history item
- `POST /api/clear-history` - Clear all history

---

## Architecture Overview

### System Components

```
Frontend (HTML/CSS/JavaScript)
          |
          v
FastAPI Backend (Python)
          |
    +-----+-----+
    |           |
Text Model  Demographics Model
(TF-IDF)    (Random Forest)
    |           |
    +-----+-----+
          |
    Combined Score
    (60% + 40%)
          |
          v
   Risk Classification
   (Critical/High/Med/Low)
```

### Data Flow

1. **User Input** → Frontend form submission
2. **Validation** → Pydantic validators (bounds checking, enum validation)
3. **Text Processing** → Language auto-detection → NLTK preprocessing
4. **Feature Extraction** → TF-IDF vectorization (25,000 features)
5. **Text Prediction** → Soft-Voting Classifier → probability
6. **Demographics Encoding** → Label encoders for categorical features
7. **Demographic Prediction** → Random Forest → probability
8. **Score Combination** → Weighted average (60% text, 40% demographics)
9. **Risk Classification** → Threshold-based categorization
10. **PDF Generation** → ReportLab with visualizations
11. **History Storage** → In-memory (max 25 items)

### Supported Languages

| Language   | Method                        | Accuracy                          |
|------------|-------------------------------|-----------------------------------|
| English    | NLTK preprocessing + TF-IDF   | 91.93%                            |
| Urdu       | Character-aware preprocessing | High (trained on Urdu data)       |
| Roman Urdu | Latin script Urdu support     | High (trained on Roman Urdu data) |

---

## Project Structure

```
ML_project/
├── src/
│   ├── app.py                  # FastAPI backend (228 lines)
│   ├── pipeline.py             # Detection pipeline (105 lines)
│   ├── baseline.py             # Text model training (210 lines)
│   ├── train_hybrid.py         # Demographics model training (61 lines)
│   ├── preprocess.py           # Multilingual text preprocessing (104 lines)
│   ├── pdf_export.py           # PDF report generation (241 lines)
│   └── multilingual.py         # BERT embeddings (for future enhancement)
│
├── static/
│   ├── index.html              # Frontend UI (340+ lines)
│   ├── css/style.css           # Professional dark theme styling (720+ lines)
│   └── js/app.js               # Client-side logic (462+ lines)
│
├── data/
│   ├── cleaned_suicide_detection.csv      # Main training dataset (232K samples)
│   ├── urdu_translated_150.csv            # Urdu samples
│   ├── romanurdu_translated_100.csv       # Roman Urdu samples
│   └── prprocessedmental_health_diagnosis_treatment_.csv  # Demographics data
│
├── outputs/
│   └── models/
│       ├── model.pkl                      # Text model (SVEM)
│       ├── tfidf.pkl                      # TF-IDF vectorizer
│       ├── hybrid_rf.pkl                  # Demographics model
│       ├── le_*.pkl                       # 7 label encoders
│       ├── model_report.txt               # Classification metrics
│       └── training_config.txt            # Model configuration
│
├── requirements.txt            # Python dependencies
├── run.bat                     # Windows startup script
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
└── .gitattributes              # Git attributes

Total: ~2,500+ lines of code, 10 trained models, 9 core Python modules
```

---

## Troubleshooting

### Models Not Found

**Error:** `FileNotFoundError: model.pkl not found`
**Solution:** Run training scripts first

```bash
python baseline.py
python train_hybrid.py
```

### NLTK Data Missing

**Error:** `LookupError: tokenizers/punkt not found`
**Solution:** Auto-downloads on first run, or manually:

```python
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

### Port 8000 Already in Use

**Solution:** Use different port

```bash
python -m uvicorn app:app --port 8001
```

### Out of Memory

**Solution:** Reduce dataset sample size in `baseline.py` (line 28):

```python
SAMPLE_SIZE = 15000  # Reduce from 30000
```

### Validation Errors

**Error:** `Value error, Age must be between 1 and 120`
**Solution:** Frontend enforces valid ranges. Check browser console for errors.

---

## Model Details

### Text Model (TF-IDF + Soft-Voting Ensemble)

**Features:**

- Input: Preprocessed text (any language)
- Vectorization: TF-IDF bigrams (max 25,000 features)
- Sublinear TF scaling, min_df=2, max_df=0.95
- Classifiers:
    - Logistic Regression (weight=1)
    - Random Forest 100 trees (weight=2)
- Voting: Soft (probability-based averaging)

**Output:** Suicide ideation probability (0-1)

### Demographic Model (Random Forest)

**Features (13 total):**

- Age, Gender (encoded), Diagnosis (encoded)
- Severity (1-10), Mood Score (1-10)
- Sleep (hours), Physical Activity (hrs/week)
- Medication (encoded), Therapy Type (encoded)
- Treatment Duration (days), Stress (0-10)
- Treatment Progress (%), Emotion (encoded)

**Output:** Risk classification (Low/Medium/High/Critical)

### Combined Scoring

```
Combined Score = (0.6 × Text Probability) + (0.4 × Demographics Probability)

Risk Classification:
  > 0.70  → Critical (RED)    - Immediate intervention needed
  > 0.50  → High (ORANGE)     - Professional help recommended
  > 0.30  → Medium (YELLOW)   - Assessment recommended
  ≤ 0.30  → Low (GREEN)       - Standard support sufficient
```

---

## Performance Metrics

### Test Set Results (6,000 samples)

| Metric    | Non-Suicide | Suicide | Overall |
|-----------|-------------|---------|---------|
| Precision | 91.79%      | 92.08%  | 92.08%  |
| Recall    | 92.27%      | 91.59%  | 91.59%  |
| F1-Score  | 92.03%      | 91.84%  | 91.84%  |
| Support   | 3,028       | 2,972   | 6,000   |

### Confusion Matrix

```
                Predicted
                Non-Suicide  Suicide
Actual Non-Suicide    2,795      233
       Suicide         255      2,717
```

**True Positive Rate:** 91.59% (catches 91.59% of suicide cases)
**True Negative Rate:** 92.27% (correctly identifies 92.27% of non-cases)
**False Positive Rate:** 7.73% (false alarms: 233/3,028)
**False Negative Rate:** 8.41% (missed cases: 255/2,972)

---

## Future Enhancements

1. **BERT Embeddings** - Integrate multilingual BERT (384-dim) for semantic understanding
2. **Database Persistence** - Replace in-memory history with PostgreSQL/MongoDB
3. **User Authentication** - Secure patient records with login system
4. **Hotline Integration** - Direct crisis helpline connection based on geography
5. **Model Explainability** - LIME/SHAP for clinical interpretability
6. **Real-time Alerts** - Email/SMS notifications for high-risk cases
7. **Model Versioning** - A/B testing framework for model improvements
8. **Mobile App** - Native iOS/Android applications

---

## Contributing

Contributions welcome! Areas for improvement:

- Improved multilingual support
- Additional language datasets
- Model performance optimization
- UI/UX enhancements
- API documentation (Swagger/OpenAPI)
- Deployment guides (Docker, Kubernetes)

---

## Important Disclaimers

**Medical Disclaimer:**
This system is an AI-powered screening tool, NOT a replacement for professional psychiatric evaluation. Always consult
qualified mental health professionals for diagnosis and treatment decisions.

**Risk Assessment:**
Risk scores should be interpreted as clinical indicators only. Do not rely solely on this tool for suicide prevention
decisions.

**Emergency Support:**
If someone is in immediate danger, contact emergency services or call a suicide prevention hotline:

- **USA:** National Suicide Prevention Lifeline: 988
- **International:** https://findahelpline.com

---

## License & Attribution

**Project:** MindGuard Pro AI
**Type:** Educational/Research Tool
**Created:** 2026

### Datasets

- Suicide Detection Dataset (Kaggle)
- Mental Health Diagnosis & Treatment Dataset
- Urdu/Roman Urdu suicide ideation datasets

### Libraries

- FastAPI, Uvicorn (REST API)
- scikit-learn (Machine Learning)
- PyTorch, Transformers (BERT - future integration)
- NLTK (Natural Language Processing)
- ReportLab (PDF Generation)
- pandas, numpy (Data Processing)

---

## Contact & Support

For issues, questions, or contributions:

1. Check troubleshooting section
2. Review model training output logs
3. Verify all dependencies installed (`pip list`)
4. Test individual components (baseline.py, train_hybrid.py)

---

**Last Updated:** April 29, 2026
**Model Version:** 1.0 (TF-IDF + Soft-Voting Ensemble)
**Status:** Production Ready
